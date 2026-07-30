"""
PropVibe - Internal Dashboard Data
==================================

Assembles the read-only snapshot behind ``GET /internal-dashboard`` - the
internal, secret-gated view of the whole sales flow: what we posted, how it
performed, which caption style is winning, and who came back as a lead.

One public function::

    collect_dashboard_data() -> dict

READ-ONLY, BY CONSTRUCTION
--------------------------
Every Airtable call this module makes goes through ``get_posts_detailed``,
``get_engagement_records``, ``get_leads`` or ``get_style_performance``, all of
which are plain list reads (HTTP GET). Nothing here creates, updates or deletes
a record, and it deliberately does not import ``create_post_record``,
``log_engagement`` or ``upsert_lead_record`` at all - so an accidental write is
an ImportError, not a silent data change.

It also never records a click. The Posts view links to ``/listing?tid=...``
rather than to the ``/t/{tracking_id}`` redirect precisely so that opening the
dashboard can't inflate the click counts the dashboard is reporting on.

BEST-EFFORT PER SECTION
-----------------------
Each section is fetched inside its own try/except and degrades to empty with an
error string attached. A missing Leads table shouldn't blank out the Posts and
engagement view, and none of this should ever 500 - it's a reporting page.
"""

from __future__ import annotations

import logging
import re

from app.airtable_client import (
    AirtableError,
    get_engagement_records,
    get_leads,
    get_posts_detailed,
    get_style_performance,
    group_engagement_by_post,
    is_configured,
)
from app.click_tracker import get_clicks, get_tracking_row_index
from app.listings_source import get_listing

logger = logging.getLogger("propvibe.internal_dashboard")

# What we show in place of a blank Name / Contact / Interested Listing.
UNKNOWN = "Unknown"

# The Status value that marks the conversion we actually care about. Compared
# case-insensitively so "viewing booked" from a different code path still counts.
VIEWING_BOOKED = "Viewing Booked"

# An Airtable record id, e.g. "recA1b2C3d4E5f6G7". A linked-record column that
# couldn't be resolved to a name comes back as a raw id like this, which is
# noise to a human reader - we treat it as "no value" and fall back.
_RECORD_ID_RE = re.compile(r"^rec[A-Za-z0-9]{14}$")


def _clean(value: str | None) -> str | None:
    """Drop a value that's blank or just a bare Airtable record id."""
    if not value:
        return None
    text = value.strip()
    if not text or _RECORD_ID_RE.match(text):
        return None
    return text


def _or_unknown(value: str | None) -> str:
    """A displayable string for a field that may be missing, blank, or an id."""
    return _clean(value) or UNKNOWN


def _listing_name(row_index: int | None, tracking_id: str | None) -> str | None:
    """
    The condo name behind a post or lead, from whichever pointer we have.

    Two routes to the same answer: the Posts row records the
    room_listings.csv row index directly, while a lead only carries the
    tracking id, which app.click_tracker maps to that same row index. Returns
    None when neither resolves - an older post from before we recorded indices,
    or a row that has since left the CSV.
    """
    if row_index is None and tracking_id:
        row_index = get_tracking_row_index(tracking_id)
    if row_index is None:
        return None

    listing = get_listing(row_index)
    return listing.condo_name if listing else None


def _posts_and_engagement() -> tuple[list[dict], list[dict], str | None]:
    """
    The Posts and Engagement views, cross-referenced to each other.

    Returns ``(posts, engagement, error)``. Posts carry their latest engagement
    snapshot rolled up (``clicks``/``likes``/``comments``), because
    /sync-engagement appends a NEW Engagement row every run - summing those
    would count the same clicks over and over, so the most recent row is the
    current truth. Engagement rows carry the listing and style of the post they
    belong to, so each row is readable on its own.
    """
    try:
        posts = get_posts_detailed()
    except AirtableError as exc:
        logger.warning("Internal dashboard could not read Posts: %s", exc)
        return [], [], str(exc)

    known_ids = {post["record_id"] for post in posts if post.get("record_id")}

    engagement_error: str | None = None
    try:
        engagement = get_engagement_records(known_ids)
    except AirtableError as exc:
        logger.warning("Internal dashboard could not read Engagement: %s", exc)
        engagement = []
        engagement_error = str(exc)

    # Shared with /public-stats' whole-system click total, so the two can never
    # disagree about which snapshot counts as a post's current engagement.
    by_post = group_engagement_by_post(engagement)

    post_view: list[dict] = []
    post_by_id: dict[str, dict] = {}
    for post in posts:
        rows = by_post.get(post["record_id"], [])
        latest = rows[-1] if rows else None
        listing_name = _listing_name(post["listing_row_index"], post["tracking_id"])
        entry = {
            "record_id": post["record_id"],
            "listing_name": listing_name or UNKNOWN,
            "style_tag": _or_unknown(post["style_tag"]),
            "template_id": _or_unknown(post["template_id"]),
            "published_at": post["created_time"],
            "caption": post["caption"] or "",
            "tracking_id": post["tracking_id"],
            "facebook_post_id": post["facebook_post_id"],
            # The landing page itself, NOT the /t/ redirect - see module docstring.
            "listing_url": (
                f"/listing?tid={post['tracking_id']}" if post["tracking_id"] else None
            ),
            "facebook_url": (
                f"https://www.facebook.com/{post['facebook_post_id']}"
                if post["facebook_post_id"]
                else None
            ),
            "clicks": latest["clicks"] if latest else 0,
            "likes": latest["likes"] if latest else 0,
            "comments": latest["comments"] if latest else 0,
            "engagement_records": len(rows),
            # Clicks counted locally by app.click_tracker. On Railway's
            # ephemeral disk this resets on redeploy, so it can legitimately
            # read lower than the Airtable snapshot above.
            "clicks_local": get_clicks(post["tracking_id"] or ""),
        }
        post_view.append(entry)
        if post["record_id"]:
            post_by_id[post["record_id"]] = entry

    # Newest post first.
    post_view.sort(key=lambda entry: entry.get("published_at") or "", reverse=True)

    engagement_view: list[dict] = []
    for row in engagement:
        parents = [post_by_id[pid] for pid in row["post_record_ids"] if pid in post_by_id]
        parent = parents[0] if parents else None
        engagement_view.append(
            {
                "record_id": row["record_id"],
                "recorded_at": row["created_time"],
                "clicks": row["clicks"],
                "likes": row["likes"],
                "comments": row["comments"],
                "total": row["clicks"] + row["likes"] + row["comments"],
                "post_listing": parent["listing_name"] if parent else UNKNOWN,
                "post_style_tag": parent["style_tag"] if parent else UNKNOWN,
                "post_facebook_id": parent["facebook_post_id"] if parent else None,
                "orphaned": parent is None,
            }
        )
    engagement_view.sort(key=lambda row: row.get("recorded_at") or "", reverse=True)

    return post_view, engagement_view, engagement_error


def _style_performance() -> dict:
    """
    Average engagement per caption style, plus the current front-runner.

    Calls ``get_style_performance()`` exactly as the caption generator does -
    same function, same numbers, no second implementation to drift from it. The
    winner is ``max`` over those averages, which is the same rule
    ``airtable_client.best_performing_style()`` applies when /create-post picks
    a style hint, so the dashboard reports the style the app is actually
    leaning toward.
    """
    try:
        averages = get_style_performance()
    except Exception as exc:  # noqa: BLE001 - a reporting view must not 500
        logger.warning("Internal dashboard could not read style performance: %s", exc)
        return {"averages": {}, "best": None, "error": str(exc)}

    ranked = sorted(averages.items(), key=lambda item: item[1], reverse=True)
    return {
        "averages": [{"style_tag": style, "average": value} for style, value in ranked],
        "best": ranked[0][0] if ranked else None,
        "error": None,
    }


def _leads() -> dict:
    """
    The Leads view plus its headline counts.

    Blank Name / Contact / Interested Listing become "Unknown" here rather than
    in the template, so every consumer of this payload sees the same thing. The
    transcript is passed through whole - the page is responsible for showing it
    collapsed - along with a short preview so the table itself never has to
    render the full text.
    """
    try:
        rows = get_leads()
    except AirtableError as exc:
        logger.warning("Internal dashboard could not read Leads: %s", exc)
        return {"rows": [], "total": 0, "viewings_booked": 0, "by_status": [], "error": str(exc)}

    view: list[dict] = []
    status_counts: dict[str, int] = {}
    viewings_booked = 0

    for lead in rows:
        status = _clean(lead["status"])
        is_booked = bool(status) and status.casefold() == VIEWING_BOOKED.casefold()
        if is_booked:
            viewings_booked += 1
        label = status or UNKNOWN
        status_counts[label] = status_counts.get(label, 0) + 1

        # Fall back to the listing the tracking id was published for. The
        # Airtable column is a linked-record field the chatbot's write can't
        # populate today, so without this it would read "Unknown" on every row.
        listing = _clean(lead["interested_listing"]) or _listing_name(
            None, lead["tracking_id"]
        )

        transcript = lead["transcript"] or ""
        view.append(
            {
                "record_id": lead["record_id"],
                "lead_id": _or_unknown(lead["lead_id"]),
                "name": _or_unknown(lead["name"]),
                "contact": _or_unknown(lead["contact"]),
                "interested_listing": listing or UNKNOWN,
                "source": _or_unknown(lead["source"]),
                "status": label,
                "is_viewing_booked": is_booked,
                "tracking_id": _or_unknown(lead["tracking_id"]),
                "budget_signal": _or_unknown(lead["budget_signal"]),
                "timeline_signal": _or_unknown(lead["timeline_signal"]),
                "transcript": transcript,
                "has_transcript": bool(transcript),
                "created_at": lead["created_time"],
            }
        )

    # Booked viewings first, then newest - the conversions are the point of
    # this table, so they shouldn't be buried under a page of raw chats.
    view.sort(
        key=lambda row: (row["is_viewing_booked"], row.get("created_at") or ""),
        reverse=True,
    )

    by_status = sorted(
        ({"status": status, "count": count} for status, count in status_counts.items()),
        key=lambda item: item["count"],
        reverse=True,
    )

    return {
        "rows": view,
        "total": len(view),
        "viewings_booked": viewings_booked,
        "by_status": by_status,
        "error": None,
    }


def collect_dashboard_data() -> dict:
    """
    The whole read-only snapshot the internal dashboard renders.

    Returns::

        {
          "airtable_configured": true,
          "posts": [...], "engagement": [...],
          "style_performance": {"averages": [...], "best": "Warm", "error": null},
          "leads": {"rows": [...], "total": 7, "viewings_booked": 2,
                    "by_status": [...], "error": null},
          "errors": ["..."],          # any section that failed to load
        }

    Never raises: a section that can't be read comes back empty with its
    message collected in ``errors``, so the page renders what it does have.
    """
    if not is_configured():
        return {
            "airtable_configured": False,
            "posts": [],
            "engagement": [],
            "style_performance": {"averages": [], "best": None, "error": None},
            "leads": {
                "rows": [],
                "total": 0,
                "viewings_booked": 0,
                "by_status": [],
                "error": None,
            },
            "errors": [
                "Airtable is not configured (AIRTABLE_API_KEY / AIRTABLE_BASE_ID "
                "are missing), so there is nothing to report yet."
            ],
        }

    posts, engagement, engagement_error = _posts_and_engagement()
    style_performance = _style_performance()
    leads = _leads()

    errors = [
        message
        for message in (engagement_error, style_performance["error"], leads["error"])
        if message
    ]

    return {
        "airtable_configured": True,
        "posts": posts,
        "engagement": engagement,
        "style_performance": style_performance,
        "leads": leads,
        "errors": errors,
    }
