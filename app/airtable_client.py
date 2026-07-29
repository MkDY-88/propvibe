"""
PropVibe - Airtable Client
==========================

Records every published post and its engagement in Airtable so the app can
*learn* which caption style performs best and lean future captions that way.

Three public functions:

    create_post_record(template_id, style_tag, caption, facebook_post_id,
                       tracking_id) -> str
        Create a row in the Posts table. Returns the new Airtable record id.

    log_engagement(post_record_id, clicks, likes, comments) -> str
        Create a row in the Engagement table, linked back to the Post record.
        Returns the new Airtable record id.

    get_style_performance() -> dict[str, float]
        Join Engagement onto Posts, group by style_tag, and return the AVERAGE
        engagement (clicks + likes + comments) for each style, e.g.
        {"Modern": 12.5, "Warm": 8.0, "Bold": 15.2}. Best-effort: returns an
        empty dict (with a logged warning) if Airtable isn't configured or the
        API errors, so callers can safely fall back to "no preference".

DESIGN NOTES
------------
* We talk to Airtable's REST API directly with ``httpx`` (already a dependency
  via the anthropic SDK and used by ``facebook_publisher.py``) rather than
  pulling in ``pyairtable`` - it isn't installed, and one more HTTP client would
  be dead weight for three small calls.

* Credentials (AIRTABLE_API_KEY, AIRTABLE_BASE_ID) are validated at *call time*,
  never at import - the rest of the app boots fine without them, exactly like
  ``copy_generator`` and ``facebook_publisher``. A write with no credentials
  raises ``AirtableError``; a read (get_style_performance) degrades to ``{}``.

* DEFENSIVE FIELD HANDLING. The real Airtable base was created with AI
  assistance, so the actual field (and table) names may differ from what we
  planned here. Two safeguards:
    1. Every table and field name has a sensible default AND an env override
       (AIRTABLE_POSTS_TABLE, AIRTABLE_FIELD_STYLE_TAG, ...), so a mismatch can
       be fixed with config, no code change.
    2. On a write, if Airtable rejects a single field - either as an unknown
       name, or as a value that column cannot hold (e.g. a linked-record
       column handed a plain name) - we log a clear warning, drop just that
       field, and retry. So one renamed, missing or wrongly-typed column never
       crashes the whole call. Reads look each value up by the expected name,
       then case-insensitively, then by keyword, and treat a genuinely absent
       field as a benign zero / skip.
"""

from __future__ import annotations

import logging
import os
import re
from urllib.parse import quote

import httpx

logger = logging.getLogger("propvibe.airtable")

API_BASE = "https://api.airtable.com/v0"

# Airtable calls are tiny JSON payloads; a short timeout keeps a hung request
# from stalling the publish/sync flow for long.
REQUEST_TIMEOUT_SECONDS = 30

# Airtable caps a list page at 100 records.
PAGE_SIZE = 100


class AirtableError(Exception):
    """A clean, user-safe failure from an Airtable write."""


# ---------------------------------------------------------------------------
# Configuration - table and field names (all overridable via the environment)
# ---------------------------------------------------------------------------
#
# Read at call time (not import) via these tiny helpers so that (a) the app
# boots without Airtable configured and (b) tests can set env vars after import.


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def _posts_table() -> str:
    return _env("AIRTABLE_POSTS_TABLE", "Posts")


def _engagement_table() -> str:
    return _env("AIRTABLE_ENGAGEMENT_TABLE", "Engagement")


def _leads_table() -> str:
    return _env("AIRTABLE_LEADS_TABLE", "Leads")


# Posts table columns.
def _posts_fields() -> dict[str, str]:
    return {
        "template_id": _env("AIRTABLE_FIELD_TEMPLATE_ID", "Template ID"),
        "style_tag": _env("AIRTABLE_FIELD_STYLE_TAG", "Style Tag"),
        "caption": _env("AIRTABLE_FIELD_CAPTION", "Caption"),
        "facebook_post_id": _env("AIRTABLE_FIELD_FACEBOOK_POST_ID", "Facebook Post ID"),
        "tracking_id": _env("AIRTABLE_FIELD_TRACKING_ID", "Tracking ID"),
        # Which room_listings.csv row this post came from (auto-ready flow
        # only; None/absent for posts created via the manual form). This is
        # what makes "next un-posted listing" durable across Railway redeploys
        # - see app.listings_source. Requires a Number column named "Listing
        # Row Index" (or the AIRTABLE_FIELD_LISTING_ROW_INDEX override) to
        # already exist on the Posts table: like every other field here, an
        # unrecognised name is dropped with a logged warning rather than
        # failing the write, but that also means posted-listing tracking
        # silently does nothing until the column exists.
        "listing_row_index": _env(
            "AIRTABLE_FIELD_LISTING_ROW_INDEX", "Listing Row Index"
        ),
    }


# Engagement table columns. "post_link" is the linked-record field pointing back
# at the Posts table (Airtable expects an array of record ids there).
def _engagement_fields() -> dict[str, str]:
    return {
        "post_link": _env("AIRTABLE_FIELD_POST_LINK", "Post"),
        "clicks": _env("AIRTABLE_FIELD_CLICKS", "Clicks"),
        "likes": _env("AIRTABLE_FIELD_LIKES", "Likes"),
        "comments": _env("AIRTABLE_FIELD_COMMENTS", "Comments"),
    }


# Leads table columns. The table also has "Lead ID" (an autonumber primary
# field Airtable fills in itself) plus "Name" and "Contact", which the chatbot
# deliberately never collects - upsert_lead_record leaves those untouched.
def _leads_fields() -> dict[str, str]:
    return {
        "tracking_id": _env("AIRTABLE_FIELD_LEAD_TRACKING_ID", "Tracking ID"),
        "interested_listing": _env("AIRTABLE_FIELD_LEAD_LISTING", "Interested Listing"),
        "source": _env("AIRTABLE_FIELD_LEAD_SOURCE", "Source"),
        "status": _env("AIRTABLE_FIELD_LEAD_STATUS", "Status"),
        "budget_signal": _env("AIRTABLE_FIELD_LEAD_BUDGET", "Budget Signal"),
        "timeline_signal": _env("AIRTABLE_FIELD_LEAD_TIMELINE", "Timeline Signal"),
        "transcript": _env("AIRTABLE_FIELD_LEAD_TRANSCRIPT", "Transcript"),
    }


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def _require_credentials() -> tuple[str, str]:
    """
    Read the API key and base id from the environment, or raise a clear error.

    Validated here (not at import) so the rest of the app boots without them.
    Mirrors the missing-credential guards in ``copy_generator`` and
    ``facebook_publisher``.
    """
    api_key = os.environ.get("AIRTABLE_API_KEY")
    base_id = os.environ.get("AIRTABLE_BASE_ID")

    if not api_key:
        raise AirtableError(
            "AIRTABLE_API_KEY is not set. Add it to your .env file "
            "(see .env.example), then restart the server."
        )
    if not base_id:
        raise AirtableError(
            "AIRTABLE_BASE_ID is not set. Add it to your .env file "
            "(see .env.example), then restart the server."
        )
    return api_key, base_id


def is_configured() -> bool:
    """True if both Airtable credentials are present (handy for a health check)."""
    return bool(os.environ.get("AIRTABLE_API_KEY") and os.environ.get("AIRTABLE_BASE_ID"))


# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _table_url(base_id: str, table: str) -> str:
    # Table names can contain spaces ("Posts and Leads"), so URL-encode.
    return f"{API_BASE}/{base_id}/{quote(table)}"


def _error_message(response: httpx.Response) -> str:
    """
    Pull Airtable's own error text out of a failed response.

    Airtable errors come back as ``{"error": {"type": ..., "message": ...}}`` or
    sometimes just ``{"error": "NOT_FOUND"}``. Fall back to the status line if
    the body isn't a shape we recognise.
    """
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("type")
            if isinstance(message, str) and message.strip():
                return message.strip()
        elif isinstance(error, str) and error.strip():
            return error.strip()

    return f"Airtable returned HTTP {response.status_code}."


def _unknown_field_name(response: httpx.Response) -> str | None:
    """
    If a response is an UNKNOWN_FIELD_NAME error, extract the offending field.

    Airtable's message reads: ``Unknown field name: "Clicks"``. We pull the
    name out of the quotes so the caller can drop just that field and retry.
    """
    try:
        payload = response.json()
    except ValueError:
        return None

    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    if error.get("type") != "UNKNOWN_FIELD_NAME":
        return None

    message = error.get("message") or ""
    match = re.search(r'"([^"]+)"', message)
    return match.group(1) if match else None


def _rejected_value(response: httpx.Response) -> str | None:
    """
    If a response rejected one of our VALUES, extract the offending value.

    Distinct from _unknown_field_name: the column exists, it just can't hold
    what we sent - most often a linked-record column being handed a plain
    name. Airtable words this as ``Value "The Birch" is not a valid record
    id...`` and, unhelpfully, never names the field, so we pull the quoted
    value out and let the caller work backwards to which field held it.
    """
    try:
        payload = response.json()
    except ValueError:
        return None

    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    if error.get("type") != "INVALID_VALUE_FOR_COLUMN":
        return None

    message = error.get("message") or ""
    match = re.search(r'"([^"]+)"', message)
    return match.group(1) if match else None


def _drop_rejected_field(response: httpx.Response, working: dict, table: str) -> bool:
    """
    Remove whichever field a failed write complained about, if we can tell.

    Returns True when a field was dropped (so the caller should retry) and
    False when the failure isn't one we can recover from by shedding a field.
    Handles both shapes of "just this one column is wrong": a name Airtable
    doesn't recognise, and a value the column won't accept.
    """
    unknown = _unknown_field_name(response)
    if unknown:
        # Match the offending name to one of our keys (case-insensitively) so
        # we drop the right one regardless of exact casing in the message.
        key = next((k for k in working if k.lower() == unknown.lower()), unknown)
        if key in working:
            logger.warning(
                "Airtable table %r has no field %r - skipping it for this record.",
                table,
                key,
            )
            working.pop(key)
            return True
        return False

    rejected = _rejected_value(response)
    if rejected is not None:
        # The error quotes the value, not the column, so find the field we sent
        # it in. Values are compared as strings because Airtable echoes them
        # back that way.
        key = next((k for k, v in working.items() if str(v) == rejected), None)
        if key is not None:
            logger.warning(
                "Airtable table %r would not accept %r for field %r (check the "
                "column's type) - skipping it for this record.",
                table,
                rejected,
                key,
            )
            working.pop(key)
            return True

    return False


def _create_record(table: str, fields: dict) -> str:
    """
    Create one record in ``table``, returning its Airtable record id.

    Defensive against a column being renamed, missing, or the wrong type: if
    Airtable rejects a single field - either as an unknown name or as a value
    the column can't hold - we log a warning, drop that field, and retry. If
    every field ends up dropped we raise a clear error rather than silently
    creating an empty row.
    """
    api_key, base_id = _require_credentials()
    url = _table_url(base_id, table)
    headers = _headers(api_key)

    working = {name: value for name, value in fields.items() if value is not None}

    # At most one retry per field (each retry can only drop one), plus a final
    # attempt - so len(fields) + 1 iterations is a safe upper bound.
    for _ in range(len(working) + 1):
        if not working:
            raise AirtableError(
                f"Could not create a record in the {table!r} table: none of the "
                f"expected fields exist there. Check the column names in Airtable "
                f"or set the AIRTABLE_FIELD_* environment overrides."
            )

        try:
            response = httpx.post(
                url,
                headers=headers,
                # typecast lets Airtable coerce strings into single-selects/numbers
                # and auto-create new select options (e.g. a new style tag).
                json={"fields": working, "typecast": True},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except httpx.RequestError as exc:
            raise AirtableError(
                f"Could not reach Airtable to write to {table!r} (network error): {exc}."
            )

        if response.is_success:
            try:
                record_id = response.json().get("id")
            except ValueError:
                record_id = None
            if not record_id:
                raise AirtableError(
                    f"Airtable accepted the write to {table!r} but returned no record id."
                )
            return record_id

        # A single bad field? Drop it and retry rather than failing the row.
        if _drop_rejected_field(response, working, table):
            continue

        # Any other error is not something we can recover from by dropping fields.
        raise AirtableError(_error_message(response))

    # Exhausted the retry budget without success (should be unreachable).
    raise AirtableError(f"Could not create a record in the {table!r} table after retries.")


def _update_record(table: str, record_id: str, fields: dict) -> str:
    """
    Patch one existing record in ``table``, returning its Airtable record id.

    The PATCH counterpart to _create_record, with the same defences: fields
    whose value is None are omitted (so a caller can leave a column alone by
    passing None), and a field Airtable rejects - unknown name or unusable
    value - is logged, dropped, and the write retried rather than failing the
    whole row. A PATCH leaves every column we don't mention untouched, which is
    what keeps the Leads table's Name/Contact intact when the chatbot updates a
    lead.
    """
    api_key, base_id = _require_credentials()
    url = f"{_table_url(base_id, table)}/{quote(record_id)}"
    headers = _headers(api_key)

    working = {name: value for name, value in fields.items() if value is not None}

    # Same retry budget as _create_record: at most one dropped field per pass.
    for _ in range(len(working) + 1):
        if not working:
            raise AirtableError(
                f"Could not update a record in the {table!r} table: none of the "
                f"expected fields exist there. Check the column names in Airtable "
                f"or set the AIRTABLE_FIELD_* environment overrides."
            )

        try:
            response = httpx.patch(
                url,
                headers=headers,
                json={"fields": working, "typecast": True},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except httpx.RequestError as exc:
            raise AirtableError(
                f"Could not reach Airtable to update {table!r} (network error): {exc}."
            )

        if response.is_success:
            return record_id

        if _drop_rejected_field(response, working, table):
            continue

        raise AirtableError(_error_message(response))

    raise AirtableError(f"Could not update a record in the {table!r} table after retries.")


def _list_records(table: str) -> list[dict]:
    """Fetch every record in ``table``, following Airtable's offset pagination."""
    api_key, base_id = _require_credentials()
    url = _table_url(base_id, table)
    headers = _headers(api_key)

    records: list[dict] = []
    offset: str | None = None

    while True:
        params: dict[str, str] = {"pageSize": str(PAGE_SIZE)}
        if offset:
            params["offset"] = offset

        try:
            response = httpx.get(
                url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS
            )
        except httpx.RequestError as exc:
            raise AirtableError(
                f"Could not reach Airtable to read {table!r} (network error): {exc}."
            )

        if not response.is_success:
            raise AirtableError(_error_message(response))

        try:
            payload = response.json()
        except ValueError:
            raise AirtableError(f"Airtable returned an unreadable response for {table!r}.")

        records.extend(payload.get("records", []))
        offset = payload.get("offset")
        if not offset:
            break

    return records


# ---------------------------------------------------------------------------
# Defensive field readers (used by get_style_performance)
# ---------------------------------------------------------------------------


def _read_field(fields: dict, expected: str, keyword: str):
    """
    Read a value from a record's ``fields`` as forgivingly as possible.

    Tries, in order: the exact expected name, a case-insensitive exact match,
    then any field whose name contains ``keyword``. Returns None if nothing
    matches, so a renamed column degrades instead of blowing up.
    """
    if expected in fields:
        return fields[expected]
    lowered = {k.lower(): k for k in fields}
    if expected.lower() in lowered:
        return fields[lowered[expected.lower()]]
    for name, value in fields.items():
        if keyword.lower() in name.lower():
            return value
    return None


def _read_number(fields: dict, expected: str, keyword: str) -> float:
    """Like _read_field but coerces to a float, defaulting a missing/bad value to 0."""
    value = _read_field(fields, expected, keyword)
    if isinstance(value, bool):  # guard: bool is a subclass of int
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _linked_post_ids(fields: dict, known_post_ids: set[str]) -> list[str]:
    """
    Find which Post record id(s) an Engagement row links to.

    Rather than trust a single hard-coded link-field name (which may have been
    renamed), we scan every field value for a list of Airtable record ids that
    intersect the known Post ids. Robust to the link column being renamed.
    """
    linked: list[str] = []
    for value in fields.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item in known_post_ids:
                    linked.append(item)
    return linked


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_post_record(
    template_id: str,
    style_tag: str,
    caption: str,
    facebook_post_id: str,
    tracking_id: str,
    listing_row_index: int | None = None,
) -> str:
    """
    Create a row in the Posts table for a freshly-published post.

    Args:
        template_id:       Which poster template was used (e.g. "A" / "B").
        style_tag:         The caption style/tone (e.g. "Modern", "Warm", "Bold").
        caption:           The caption text that was posted.
        facebook_post_id:  The id Facebook returned for the published post.
        tracking_id:       The short id embedded in the post's tracking link.
        listing_row_index: The room_listings.csv row this post came from (auto-
            ready flow only). None (the default) omits the field entirely, so
            posts from the manual form are unaffected.

    Returns:
        The new Airtable record id (e.g. "recXXXXXXXXXXXXXX").

    Raises:
        AirtableError: for missing credentials, a network error, or Airtable
            rejecting the write. The message is safe to show a user.
    """
    names = _posts_fields()
    fields = {
        names["template_id"]: template_id,
        names["style_tag"]: style_tag,
        names["caption"]: caption,
        names["facebook_post_id"]: facebook_post_id,
        names["tracking_id"]: tracking_id,
        names["listing_row_index"]: listing_row_index,
    }
    return _create_record(_posts_table(), fields)


def log_engagement(
    post_record_id: str,
    clicks: int,
    likes: int,
    comments: int,
) -> str:
    """
    Create a row in the Engagement table, linked to a Post record.

    Args:
        post_record_id: The Airtable record id returned by create_post_record.
        clicks:         Click count (from the tracking link).
        likes:          Facebook like count.
        comments:       Facebook comment count.

    Returns:
        The new Airtable record id.

    Raises:
        AirtableError: for missing credentials, a network error, or Airtable
            rejecting the write.
    """
    names = _engagement_fields()
    fields = {
        # Linked-record fields take an array of record ids.
        names["post_link"]: [post_record_id] if post_record_id else None,
        names["clicks"]: int(clicks),
        names["likes"]: int(likes),
        names["comments"]: int(comments),
    }
    return _create_record(_engagement_table(), fields)


def _read_int(fields: dict, expected: str, keyword: str) -> int | None:
    """Like _read_field but coerces to an int, or None if missing/unparseable."""
    value = _read_field(fields, expected, keyword)
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def get_posts() -> list[dict]:
    """
    Return every row in the Posts table, normalised for the sync flow.

    Each item is a dict::

        {
          "record_id":         "recXXXXXXXXXXXXXX",   # Airtable record id
          "facebook_post_id":  "1234_5678" | None,
          "tracking_id":       "ab12cd34" | None,
          "style_tag":         "Warm" | None,
          "listing_row_index": 17 | None,              # auto-ready posts only
        }

    Field values are read defensively (exact name, then case-insensitive, then
    keyword) so a renamed column degrades to None rather than crashing.

    Raises:
        AirtableError: for missing credentials or an API/network error. Callers
            (e.g. /sync-engagement) surface this as a clean error.
    """
    records = _list_records(_posts_table())
    names = _posts_fields()

    posts: list[dict] = []
    for record in records:
        fields = record.get("fields", {})
        fb_id = _read_field(fields, names["facebook_post_id"], "facebook")
        tracking = _read_field(fields, names["tracking_id"], "tracking")
        style = _read_field(fields, names["style_tag"], "style")
        row_index = _read_int(fields, names["listing_row_index"], "row index")
        posts.append(
            {
                "record_id": record.get("id"),
                "facebook_post_id": str(fb_id).strip() if fb_id else None,
                "tracking_id": str(tracking).strip() if tracking else None,
                "style_tag": str(style).strip() if style else None,
                "listing_row_index": row_index,
            }
        )
    return posts


def upsert_lead_record(
    tracking_id: str,
    interested_listing: str,
    status: str,
    budget_signal: str | None,
    timeline_signal: str | None,
    transcript: str,
) -> str:
    """
    Create or update the Leads row for a chatbot conversation.

    One row per ``tracking_id``: the first message of a conversation creates
    it, every message after that patches the same row, so the Leads table shows
    the current state of each lead rather than one row per chat turn. We find
    the existing row by listing the table and matching client-side, the same
    way get_posts() reads Posts - no formula parameter to get wrong, and the
    Leads table is small.

    "Source" is always written as "Chatbot". "Name" and "Contact" are never
    touched: the assistant is instructed not to ask for them, so whatever a
    human has typed into those columns survives every update.

    NOTE: as the base stands today, "Interested Listing" is a linked-record
    column pointing at the Properties table, whose primary field is a computed
    autonumber - so Airtable can neither match nor create a row there from a
    condo name, and the write of that one field is dropped with a logged
    warning (see _drop_rejected_field). Everything else on the row still
    lands, and the listing stays recoverable from the tracking id. Making the
    condo name stick needs a change in Airtable, not here: either retype
    "Interested Listing" as a plain text field, or give Properties a
    non-computed primary field and populate it from room_listings.csv.

    Args:
        tracking_id:        The id from the lead's tracking link - the key this
            row is matched on. A blank id skips the lookup and creates a row.
        interested_listing: The property they're asking about (the condo name,
            or "Unknown" when the tracking id maps to no listing).
        status:             "lead" or "prospect", from lead_chatbot.qualify_lead.
        budget_signal:      What they've revealed about budget, or None.
        timeline_signal:    What they've revealed about move-in timing, or None.
        transcript:         The whole conversation so far, as plain text.

    Returns:
        The Airtable record id of the created or updated row.

    Raises:
        AirtableError: for missing credentials, a network error, or Airtable
            rejecting the write. Callers treat lead logging as best-effort -
            a chat reply must still reach the user when this fails.
    """
    names = _leads_fields()
    tracking_id = (tracking_id or "").strip()

    fields = {
        names["tracking_id"]: tracking_id or None,
        names["interested_listing"]: interested_listing,
        names["source"]: "Chatbot",
        names["status"]: status,
        names["budget_signal"]: budget_signal,
        names["timeline_signal"]: timeline_signal,
        names["transcript"]: transcript,
    }

    table = _leads_table()

    # Without a tracking id there's nothing to match on, so this can only ever
    # be a new row.
    if tracking_id:
        existing_id = None
        for record in _list_records(table):
            value = _read_field(record.get("fields", {}), names["tracking_id"], "tracking")
            if value is not None and str(value).strip() == tracking_id:
                existing_id = record.get("id")
                break
        if existing_id:
            return _update_record(table, existing_id, fields)

    return _create_record(table, fields)


def get_style_performance() -> dict[str, float]:
    """
    Average engagement per caption style, learned from historical posts.

    Joins the Engagement table onto the Posts table (by linked record id),
    groups by each Post's style_tag, and returns the AVERAGE of
    (clicks + likes + comments) for each style::

        {"Modern": 12.5, "Warm": 8.0, "Bold": 15.2}

    We average (rather than sum) so that repeated engagement snapshots from
    /sync-engagement don't inflate a style's score just because it was synced
    more often.

    BEST-EFFORT: this feeds an optional "lean toward the winning style" hint, so
    it never raises. If Airtable isn't configured, or a read fails, it logs a
    warning and returns ``{}`` - the caller then simply uses no style preference.
    """
    if not is_configured():
        logger.warning(
            "Airtable is not configured (AIRTABLE_API_KEY / AIRTABLE_BASE_ID missing); "
            "style-performance data is unavailable - falling back to no preference."
        )
        return {}

    try:
        posts = _list_records(_posts_table())
        engagements = _list_records(_engagement_table())
    except AirtableError as exc:
        logger.warning("Could not load style-performance data from Airtable: %s", exc)
        return {}

    post_names = _posts_fields()
    eng_names = _engagement_fields()

    # Map each Post record id -> its style_tag.
    style_by_post: dict[str, str] = {}
    for record in posts:
        fields = record.get("fields", {})
        style = _read_field(fields, post_names["style_tag"], "style")
        if isinstance(style, str) and style.strip():
            style_by_post[record["id"]] = style.strip()

    if not style_by_post:
        return {}

    known_ids = set(style_by_post)

    # Collect each engagement value under the style(s) of the post it links to.
    values_by_style: dict[str, list[float]] = {}
    for record in engagements:
        fields = record.get("fields", {})
        clicks = _read_number(fields, eng_names["clicks"], "click")
        likes = _read_number(fields, eng_names["likes"], "like")
        comments = _read_number(fields, eng_names["comments"], "comment")
        total = clicks + likes + comments

        linked = _linked_post_ids(fields, known_ids)
        for post_id in linked:
            style = style_by_post.get(post_id)
            if style:
                values_by_style.setdefault(style, []).append(total)

    return {
        style: sum(values) / len(values)
        for style, values in values_by_style.items()
        if values
    }


# ---------------------------------------------------------------------------
# Read-only reporting reads (the internal dashboard)
# ---------------------------------------------------------------------------
#
# Everything below this line only ever calls _list_records, which is a plain
# HTTP GET. Nothing here creates, updates or deletes an Airtable record: the
# internal dashboard is a read-only consumer of tables the publish flow and the
# lead chatbot own. Values are read through the same defensive helpers the sync
# flow uses, so a renamed or blank column degrades to None instead of raising.


def _leads_read_fields() -> dict[str, str]:
    """
    Every Leads column the dashboard READS, including the three nothing writes.

    Deliberately separate from ``_leads_fields()``: that dict defines what
    ``upsert_lead_record`` writes, and a reporting view must not widen it.
    "Lead ID" is Airtable's own autonumber, and "Name"/"Contact" are only ever
    filled in by a human - the chatbot is instructed never to ask for them - so
    all three are read-only by nature and are usually blank.
    """
    names = dict(_leads_fields())
    names["lead_id"] = _env("AIRTABLE_FIELD_LEAD_ID", "Lead ID")
    names["name"] = _env("AIRTABLE_FIELD_LEAD_NAME", "Name")
    names["contact"] = _env("AIRTABLE_FIELD_LEAD_CONTACT", "Contact")
    return names


def _as_text(value) -> str | None:
    """
    Flatten one Airtable value to trimmed display text, or None if it's empty.

    Airtable hands back plain strings, numbers, and - for linked-record and
    multi-select columns - lists. A list is joined so the caller never has to
    render a Python repr, and an empty/whitespace value collapses to None so
    the dashboard can substitute "Unknown" in exactly one place.
    """
    if value is None:
        return None
    if isinstance(value, list):
        parts = [_as_text(item) for item in value]
        joined = ", ".join(part for part in parts if part)
        return joined or None
    text = str(value).strip()
    return text or None


def _read_text(fields: dict, expected: str, keyword: str) -> str | None:
    """``_read_field`` narrowed to display text (None when missing or blank)."""
    return _as_text(_read_field(fields, expected, keyword))


def get_posts_detailed() -> list[dict]:
    """
    Every Posts row with the columns a reporting view needs, read-only.

    Richer than ``get_posts()`` (which is shaped for /sync-engagement and
    deliberately minimal): this also returns the caption and Airtable's own
    ``createdTime``, which is when the post was logged - i.e. published.

    Each item::

        {
          "record_id": "recXXXXXXXXXXXXXX",
          "template_id": "Template A" | None,
          "style_tag": "Warm" | None,
          "caption": "..." | None,
          "facebook_post_id": "1234_5678" | None,
          "tracking_id": "ab12cd34" | None,
          "listing_row_index": 17 | None,
          "created_time": "2026-07-30T02:11:00.000Z" | None,
        }

    Raises:
        AirtableError: for missing credentials or an API/network error.
    """
    records = _list_records(_posts_table())
    names = _posts_fields()

    posts: list[dict] = []
    for record in records:
        fields = record.get("fields", {})
        posts.append(
            {
                "record_id": record.get("id"),
                "template_id": _read_text(fields, names["template_id"], "template"),
                "style_tag": _read_text(fields, names["style_tag"], "style"),
                "caption": _read_text(fields, names["caption"], "caption"),
                "facebook_post_id": _read_text(
                    fields, names["facebook_post_id"], "facebook"
                ),
                "tracking_id": _read_text(fields, names["tracking_id"], "tracking"),
                "listing_row_index": _read_int(
                    fields, names["listing_row_index"], "row index"
                ),
                "created_time": record.get("createdTime"),
            }
        )
    return posts


def get_engagement_records(known_post_ids: set[str]) -> list[dict]:
    """
    Every Engagement row, with the Post record id(s) it links back to.

    ``known_post_ids`` is the set of Post record ids to match links against -
    pass the ids from ``get_posts_detailed()``. Links are found the same way
    ``get_style_performance`` finds them (scan every list value for a known
    Post id) so a renamed link column still resolves.

    Each item::

        {
          "record_id": "recXXXXXXXXXXXXXX",
          "post_record_ids": ["recYYYYYYYYYYYYYY"],
          "clicks": 4, "likes": 10, "comments": 2,
          "created_time": "2026-07-30T02:11:00.000Z" | None,
        }

    Raises:
        AirtableError: for missing credentials or an API/network error.
    """
    records = _list_records(_engagement_table())
    names = _engagement_fields()

    rows: list[dict] = []
    for record in records:
        fields = record.get("fields", {})
        rows.append(
            {
                "record_id": record.get("id"),
                "post_record_ids": _linked_post_ids(fields, known_post_ids),
                "clicks": int(_read_number(fields, names["clicks"], "click")),
                "likes": int(_read_number(fields, names["likes"], "like")),
                "comments": int(_read_number(fields, names["comments"], "comment")),
                "created_time": record.get("createdTime"),
            }
        )
    return rows


def get_leads() -> list[dict]:
    """
    Every Leads row, read-only, with each column flattened to display text.

    Missing and blank columns come back as None rather than "" or a KeyError -
    the dashboard turns those into "Unknown". Name, Contact and (today)
    Interested Listing are blank on most rows by design: the chatbot never asks
    for the first two, and the third is a linked-record column Airtable rejects
    on write (see ``upsert_lead_record``).

    Each item::

        {
          "record_id": "recXXXXXXXXXXXXXX",
          "lead_id": "12" | None, "name": None, "contact": None,
          "interested_listing": "The Birch" | None,
          "source": "Chatbot" | None, "status": "Viewing Booked" | None,
          "tracking_id": "ab12cd34" | None,
          "budget_signal": None, "timeline_signal": None,
          "transcript": "Lead: ...\\nAssistant: ..." | None,
          "created_time": "2026-07-30T02:11:00.000Z" | None,
        }

    Raises:
        AirtableError: for missing credentials or an API/network error.
    """
    records = _list_records(_leads_table())
    names = _leads_read_fields()

    leads: list[dict] = []
    for record in records:
        fields = record.get("fields", {})
        leads.append(
            {
                "record_id": record.get("id"),
                "lead_id": _read_text(fields, names["lead_id"], "lead id"),
                "name": _read_text(fields, names["name"], "name"),
                "contact": _read_text(fields, names["contact"], "contact"),
                "interested_listing": _read_text(
                    fields, names["interested_listing"], "interested"
                ),
                "source": _read_text(fields, names["source"], "source"),
                "status": _read_text(fields, names["status"], "status"),
                "tracking_id": _read_text(fields, names["tracking_id"], "tracking"),
                "budget_signal": _read_text(fields, names["budget_signal"], "budget"),
                "timeline_signal": _read_text(
                    fields, names["timeline_signal"], "timeline"
                ),
                "transcript": _read_text(fields, names["transcript"], "transcript"),
                "created_time": record.get("createdTime"),
            }
        )
    return leads
