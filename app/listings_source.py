"""
PropVibe - Listings Source
===========================

Reads the auto-ready listing pool from data/room_listings.csv (a room-rental
schema: one row per rentable room, not a whole-unit schema) and the demo photo
pool from data/unit_photos/.

Each Listing is identified by ROW INDEX - its position among data rows, zero-
indexed, header and any blank rows excluded - rather than by condo_name +
room_type. The CSV contains genuine duplicate name+type rows (e.g. "Nexus
Residency, Medium" appears 3 times), so a name-based identifier would cause
publishing one duplicate to silently mark its siblings as already-posted too.

The two public entry points:

    next_unposted_listing(posted_indices, after=None) -> Listing | None
        The first listing not in `posted_indices`, optionally strictly after
        row index `after` (used by the dashboard's Skip button).

    random_pool_photo() -> str
        A random file path from the demo photo pool.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Under assets/, not data/ - data/ is gitignored (it's app.click_tracker's
# runtime state directory), which would silently exclude these from git and
# therefore from a Railway deploy. These files are static seed content, not
# runtime state, so they belong alongside assets/fonts/ instead.
CSV_PATH = REPO_ROOT / "assets" / "listings" / "room_listings.csv"
PHOTO_POOL_DIR = REPO_ROOT / "assets" / "listings" / "unit_photos"

# Values that mean "no parking data", beyond a plain empty string - the CSV
# uses "N/A" for a handful of rows (e.g. Icon City).
_BLANK_PARKING_VALUES = {"", "n/a", "na", "none", "-"}


@dataclass(frozen=True)
class Listing:
    row_index: int  # zero-indexed among data rows; stable identifier
    condo_name: str
    price: str  # pre-formatted for display, e.g. "RM 1,000"
    address: str
    room_type: str
    bed_type: str
    bathroom_type: str
    parking_rental: str | None  # None when blank/"N/A" in the CSV

    @property
    def features_text(self) -> str:
        """Short comma-separated line for the poster's bottom-left text block."""
        parts = [
            self.room_type,
            f"{self.bed_type} bed",
            f"{self.bathroom_type} bathroom",
        ]
        if self.parking_rental:
            parts.append(f"parking {self.parking_rental}/month")
        return ", ".join(part for part in parts if part)


def _clean_parking(raw: str) -> str | None:
    value = raw.strip()
    if value.lower() in _BLANK_PARKING_VALUES:
        return None
    return f"RM {value}"


def load_listings() -> list[Listing]:
    """
    Parse room_listings.csv into an ordered list of Listing rows.

    Re-reads the file on every call - at 178 rows this is a trivial cost, and
    it keeps the CSV as the single source of truth without a caching layer to
    invalidate.
    """
    listings: list[Listing] = []
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        row_index = 0
        for row in reader:
            condo_name = (row.get("Condo Name") or "").strip()
            if not condo_name:
                # Blank trailing line at EOF, or a malformed row - not a listing.
                continue

            price_raw = (row.get("Price (RM)") or "").strip()
            listings.append(
                Listing(
                    row_index=row_index,
                    condo_name=condo_name,
                    price=f"RM {price_raw}" if price_raw else "Price on request",
                    address=(row.get("Address") or "").strip(),
                    room_type=(row.get("Room Type") or "").strip(),
                    bed_type=(row.get("Bed Type") or "").strip(),
                    bathroom_type=(row.get("Bathroom Type") or "").strip(),
                    parking_rental=_clean_parking(row.get("Parking Rental (RM)") or ""),
                )
            )
            row_index += 1

    return listings


def next_unposted_listing(posted_indices: set[int], after: int | None = None) -> Listing | None:
    """
    The first listing not in `posted_indices`, optionally strictly after `after`.

    `after` is what makes the dashboard's Skip button work: the frontend passes
    the currently-shown row's index and gets a different (still un-posted)
    candidate back, with no server-side persistence needed for the skip itself.
    """
    for listing in load_listings():
        if after is not None and listing.row_index <= after:
            continue
        if listing.row_index in posted_indices:
            continue
        return listing
    return None


def random_pool_photo() -> str:
    """A random file path from the demo unit-photo pool."""
    photos = sorted(p for p in PHOTO_POOL_DIR.iterdir() if p.is_file())
    if not photos:
        raise FileNotFoundError(f"No photos found in {PHOTO_POOL_DIR}")
    return str(random.choice(photos))
