"""
PropVibe - Click Tracker
========================

A tiny file-backed counter of tracking-link hits, keyed by ``tracking_id``.

WHY A LOCAL FILE (and not an Airtable write per click)?
  * The ``/t/{tracking_id}`` redirect must be fast and must still count a click
    even if Airtable is momentarily down - a lead landing page should never wait
    on a third-party API.
  * It keeps click counting off Airtable's write path. ``/sync-engagement`` later
    reads these counts and folds them into each post's engagement row, so all
    the "write to Airtable" logic lives in one place.

Storage is a single JSON file ``{tracking_id: count}``. Reads/writes are guarded
by a lock so two concurrent redirects in the same process can't clobber each
other, and the write is done to a temp file then renamed so a crash mid-write
can't corrupt the store.

This is deliberately hackathon-grade: on an ephemeral filesystem (e.g. Railway)
the counts reset on redeploy. That's fine for the demo loop - the durable record
of engagement lives in Airtable.

Which is why this store is a STAGING BUFFER, not a reporting source. Anything
that displays a standing total (the landing-page hero, the internal dashboard,
the daily report) must read the synced Airtable numbers instead - see
``airtable_client.total_clicks_recorded()``. Reading a lifetime total from here
would silently reset it on the next restart.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger("propvibe.clicks")

# Resolved relative to THIS file (not the CWD) so it behaves the same locally and
# on Railway, matching how main.py resolves UPLOAD_ROOT / STATIC_DIR.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CLICKS_FILE = DATA_DIR / "tracking_clicks.json"

# Maps a published post's facebook_post_id -> its tracking_id. Recorded at
# publish time so /sync-engagement can find a post's clicks by its Facebook id
# even when the Airtable Posts table has no "Tracking ID" column to store the
# tracking id in. Clicks arrive keyed by tracking_id (that's all the /t/{id} URL
# carries); this map lets sync bridge facebook_post_id -> tracking_id -> clicks
# without depending on Airtable's schema.
LINKS_FILE = DATA_DIR / "post_links.json"

# Maps a tracking_id -> the room_listings.csv row index the post was built from.
# Recorded at publish time (auto-ready flow only) so the landing page a lead
# clicks through to, and the chatbot that answers their questions there, can
# both tell WHICH property they're actually asking about. The tracking id is the
# only thing the /t/{id} link carries, so this is the bridge from that id back
# to the listing's real details.
TRACKING_LISTINGS_FILE = DATA_DIR / "tracking_listings.json"

_lock = threading.Lock()


def _load() -> dict[str, int]:
    """Load the counts file, tolerating a missing/corrupt/foreign-shaped file."""
    try:
        with CLICKS_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}

    if not isinstance(data, dict):
        return {}

    counts: dict[str, int] = {}
    for key, value in data.items():
        try:
            counts[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return counts


def _save(counts: dict[str, int]) -> None:
    """Persist counts via a temp-file-then-rename so a crash can't corrupt it."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CLICKS_FILE.with_name(CLICKS_FILE.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(counts, handle)
    tmp.replace(CLICKS_FILE)


def record_click(tracking_id: str) -> int:
    """
    Increment and persist the click count for ``tracking_id``.

    Returns the new count. A blank id is ignored (returns 0). A failure to
    persist is logged, not raised - a redirect must never 500 just because the
    counter file couldn't be written.
    """
    tracking_id = (tracking_id or "").strip()
    if not tracking_id:
        return 0

    with _lock:
        counts = _load()
        counts[tracking_id] = counts.get(tracking_id, 0) + 1
        new_count = counts[tracking_id]
        try:
            _save(counts)
        except OSError as exc:
            logger.warning("Could not persist click for %r: %s", tracking_id, exc)
        return new_count


def get_clicks(tracking_id: str) -> int:
    """Return the recorded click count for ``tracking_id`` (0 if never hit)."""
    tracking_id = (tracking_id or "").strip()
    if not tracking_id:
        return 0
    with _lock:
        return _load().get(tracking_id, 0)


def _load_links() -> dict[str, str]:
    """Load the facebook_post_id -> tracking_id map, tolerating a missing file."""
    try:
        with LINKS_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}

    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if v is not None}


def _save_links(links: dict[str, str]) -> None:
    """Persist the post-link map via a temp-file-then-rename."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LINKS_FILE.with_name(LINKS_FILE.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(links, handle)
    tmp.replace(LINKS_FILE)


def link_post(facebook_post_id: str, tracking_id: str) -> None:
    """
    Remember which ``tracking_id`` belongs to a published post.

    Called at publish time. Lets /sync-engagement resolve a post's clicks from
    its facebook_post_id (which IS stored in Airtable) without relying on the
    Airtable Posts table having a "Tracking ID" column. Blank ids are ignored;
    a persistence failure is logged, not raised.
    """
    facebook_post_id = (facebook_post_id or "").strip()
    tracking_id = (tracking_id or "").strip()
    if not facebook_post_id or not tracking_id:
        return

    with _lock:
        links = _load_links()
        links[facebook_post_id] = tracking_id
        try:
            _save_links(links)
        except OSError as exc:
            logger.warning(
                "Could not persist post link %r -> %r: %s",
                facebook_post_id,
                tracking_id,
                exc,
            )


def tracking_id_for_post(facebook_post_id: str) -> str | None:
    """Return the tracking_id recorded for a facebook_post_id, or None."""
    facebook_post_id = (facebook_post_id or "").strip()
    if not facebook_post_id:
        return None
    with _lock:
        return _load_links().get(facebook_post_id)


def _load_tracking_listings() -> dict[str, int]:
    """Load the tracking_id -> row_index map, tolerating a missing/corrupt file."""
    try:
        with TRACKING_LISTINGS_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}

    if not isinstance(data, dict):
        return {}

    listings: dict[str, int] = {}
    for key, value in data.items():
        try:
            listings[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return listings


def _save_tracking_listings(listings: dict[str, int]) -> None:
    """Persist the tracking-listing map via a temp-file-then-rename."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = TRACKING_LISTINGS_FILE.with_name(TRACKING_LISTINGS_FILE.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(listings, handle)
    tmp.replace(TRACKING_LISTINGS_FILE)


def save_tracking_listing(tracking_id: str, row_index: int) -> None:
    """
    Remember which room_listings.csv row a ``tracking_id`` was published for.

    Called at publish time, right after the post link is recorded. Lets
    /listing-info and /chat resolve the listing a lead is looking at from the
    tracking id in their URL. A blank id or an unusable row index is ignored;
    a persistence failure is logged, not raised - publishing must never fail
    because this bookkeeping file couldn't be written.
    """
    tracking_id = (tracking_id or "").strip()
    if not tracking_id:
        return
    try:
        row_index = int(row_index)
    except (TypeError, ValueError):
        return

    with _lock:
        listings = _load_tracking_listings()
        listings[tracking_id] = row_index
        try:
            _save_tracking_listings(listings)
        except OSError as exc:
            logger.warning(
                "Could not persist tracking listing %r -> %d: %s",
                tracking_id,
                row_index,
                exc,
            )


def get_tracking_row_index(tracking_id: str) -> int | None:
    """Return the listing row index recorded for a tracking_id, or None."""
    tracking_id = (tracking_id or "").strip()
    if not tracking_id:
        return None
    with _lock:
        return _load_tracking_listings().get(tracking_id)


# Maps a tracking_id -> the pool photo filename the post was published with.
# A sibling to TRACKING_LISTINGS_FILE rather than a widened row_index shape:
# get_tracking_row_index() has existing callers that treat its return value
# as a plain int, so this keeps that contract untouched. Lets the public
# landing page (GET /listing-info) show the exact photo a lead's post used,
# so the AI lighting/weather toggle there starts from the real photo instead
# of a random one.
TRACKING_PHOTOS_FILE = DATA_DIR / "tracking_photos.json"


def _load_tracking_photos() -> dict[str, str]:
    """Load the tracking_id -> photo_filename map, tolerating a missing/corrupt file."""
    try:
        with TRACKING_PHOTOS_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}

    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if v is not None}


def _save_tracking_photos(photos: dict[str, str]) -> None:
    """Persist the tracking-photo map via a temp-file-then-rename."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = TRACKING_PHOTOS_FILE.with_name(TRACKING_PHOTOS_FILE.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(photos, handle)
    tmp.replace(TRACKING_PHOTOS_FILE)


def save_tracking_photo(tracking_id: str, photo_filename: str) -> None:
    """
    Remember which pool photo a tracking_id was published with.

    Called at publish time, right alongside save_tracking_listing(). Same
    behaviour as its sibling: a blank id or filename is a no-op, and a
    persistence failure is logged, not raised - publishing must never fail
    because this bookkeeping file couldn't be written.
    """
    tracking_id = (tracking_id or "").strip()
    photo_filename = (photo_filename or "").strip()
    if not tracking_id or not photo_filename:
        return

    with _lock:
        photos = _load_tracking_photos()
        photos[tracking_id] = photo_filename
        try:
            _save_tracking_photos(photos)
        except OSError as exc:
            logger.warning(
                "Could not persist tracking photo %r -> %r: %s",
                tracking_id,
                photo_filename,
                exc,
            )


def get_tracking_photo(tracking_id: str) -> str | None:
    """Return the photo_filename recorded for a tracking_id, or None."""
    tracking_id = (tracking_id or "").strip()
    if not tracking_id:
        return None
    with _lock:
        return _load_tracking_photos().get(tracking_id)
