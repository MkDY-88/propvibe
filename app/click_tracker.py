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
