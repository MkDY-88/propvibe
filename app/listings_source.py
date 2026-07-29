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

The public entry points:

    next_unposted_listing(posted_indices, after=None) -> Listing | None
        The first listing not in `posted_indices`, optionally strictly after
        row index `after` (used by the dashboard's Skip button).

    get_listing(row_index) -> Listing | None
        The listing at a given row index, for turning a published post's
        recorded row index back into the property's details.

    search_listings(**criteria) -> SearchResult
        Ranked listings matching price/area/room/bed/bathroom/feature criteria,
        for suggesting real alternatives to a lead who asks what else is
        available. The result carries an `is_exact` flag - see SearchResult.

    random_pool_photo() -> str
        A random file path from the demo photo pool.
"""

from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

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
    price_value: int | None  # same price as a plain int for comparison; None if unparseable
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


def _parse_price(raw: str) -> int | None:
    """
    The numeric ringgit amount in a raw CSV price cell, or None.

    The display string is built separately (and unchanged) from the same cell;
    this is only for comparisons like "under RM 900". Blank cells, "N/A" and
    anything else without a number give None, which every caller must treat as
    "no price to compare" rather than as zero.
    """
    digits = re.sub(r"[^\d.]", "", raw or "")
    if not digits:
        return None
    try:
        return int(float(digits))
    except ValueError:
        # e.g. a cell containing several dots - not a number we can trust.
        return None


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
                    price_value=_parse_price(price_raw),
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


def get_listing(row_index: int) -> Listing | None:
    """
    The listing at `row_index`, or None if there's no such row.

    Used by the lead-facing endpoints (/listing-info, /chat) to turn the row
    index recorded against a tracking id back into the property's real details.
    Re-reads the CSV via load_listings() like every other entry point here -
    178 rows is cheap, and it keeps the file the single source of truth.
    """
    for listing in load_listings():
        if listing.row_index == row_index:
            return listing
    return None


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

# How far either side of a stated target price still counts as "around" it.
# A lead saying "around RM 1,000" plainly means RM 950 too, and prices in this
# pool move in RM 50-100 steps, so a percentage band a shade under a couple of
# those steps is the honest reading of the phrase.
DEFAULT_PRICE_TOLERANCE = 0.15

# The wider bands the relaxation ladder falls back to, in order, once nothing
# matches the stated one. Only ever reached on a search that would otherwise
# return nothing, and the result is flagged as relaxed when they are used.
_RELAXED_PRICE_TOLERANCES = (0.30, 0.50)

# What an explicit ceiling ("under RM 800") stretches to at those same two
# steps. Deliberately much tighter than the target-price bands: a ceiling is a
# statement about what someone can afford, so overshooting it by a lot is not a
# near miss, it is a different conversation.
_RELAXED_MAX_PRICE_STRETCH = (1.10, 1.20)

# Sorts a listing we cannot price to the back of any price-based tie-break,
# without excluding it outright.
_UNKNOWN_PRICE_DISTANCE = float("inf")

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Shorthands people type that no address in the CSV spells out. Kept to the two
# that are genuinely ubiquitous here rather than growing into a gazetteer -
# everything else in this pool ("Cheras", "Mont Kiara", "Setapak") already
# appears in the address text verbatim.
_LOCATION_ALIASES = {
    "kl": "kuala lumpur",
    "pj": "petaling jaya",
}

# The Bathroom Type column only ever holds "Private" or "Shared", but leads say
# it a dozen ways. Map the ways they say it onto the two values that exist.
_BATHROOM_SYNONYMS = {
    "private": ("private", "ensuite", "en suite", "own", "attached", "personal"),
    "shared": ("shared", "share", "sharing", "common"),
}

# Tokens shorter than this are skipped when falling back to word-by-word
# matching - "en", "of" and friends match everything and mean nothing.
_MIN_TOKEN_LENGTH = 3


def _norm(text: str | None) -> str:
    """Lowercase `text` with every run of punctuation flattened to one space."""
    return _NON_ALNUM.sub(" ", (text or "").lower()).strip()


def _tokens(text: str) -> list[str]:
    """The usefully-long words of an already-normalised string."""
    return [word for word in text.split() if len(word) >= _MIN_TOKEN_LENGTH]


def _loose_match(keyword: str, *fields: str) -> bool:
    """
    Whether `keyword` describes any of `fields`, forgivingly.

    Two passes, both on normalised text: the whole phrase as a substring
    ("big window" in "Medium With Big Window"), then any single word of it
    ("master bedroom" -> "master", which is what the Room Type column actually
    says). Word-level matching is what makes real phrasing work at all, since
    almost nobody types a column value exactly as it is written.
    """
    needle = _norm(keyword)
    if not needle:
        return False

    haystack = " ".join(_norm(field) for field in fields if field)
    if not haystack:
        return False

    if needle in haystack:
        return True
    return any(word in haystack for word in _tokens(needle))


def _location_match(keyword: str, address: str) -> bool:
    """
    Whether `address` is in the area `keyword` names.

    Substring first, then every word of the keyword having to appear somewhere
    in the address - "Taman Cheras" should match "Taman Bukit Cheras" even
    though the phrase itself is not in there. Unlike _loose_match this is ALL
    words rather than any: an area is the one criterion we never relax, so a
    half-match on it would quietly send someone to the wrong side of the city.
    """
    needle = _norm(keyword)
    if not needle:
        return False

    needle = _LOCATION_ALIASES.get(needle, needle)
    haystack = _norm(address)
    if needle in haystack:
        return True

    words = _tokens(needle)
    return bool(words) and all(word in haystack for word in words)


@dataclass(frozen=True)
class _Criteria:
    """
    One normalised, immutable filter set. Internal to this module.

    Kept separate from search_listings' keyword arguments so the relaxation
    ladder can derive loosened copies of it with dataclasses.replace() without
    re-validating or re-lowercasing anything.
    """

    max_price: int | None = None
    target_price: int | None = None
    price_tolerance: float = DEFAULT_PRICE_TOLERANCE
    location: str = ""
    room_type: str = ""
    bed_type: str = ""
    bathroom_type: str = ""
    features: tuple[str, ...] = ()

    @property
    def has_any(self) -> bool:
        return bool(
            self.max_price is not None
            or self.target_price is not None
            or self.location
            or self.room_type
            or self.bed_type
            or self.bathroom_type
            or self.features
        )

    @property
    def price_reference(self) -> int | None:
        """
        The figure to measure "close to their budget" against, if any.

        A target wins over a ceiling because it is the more specific statement.
        With only a ceiling we still rank towards it rather than towards the
        cheapest row: someone who says "up to RM 900" is telling us what they
        are willing to spend, and the best room they can afford is a better
        first offer than the least room they could settle for.
        """
        return self.target_price if self.target_price is not None else self.max_price


def _price_band(criteria: _Criteria) -> tuple[int | None, int | None]:
    """The (low, high) ringgit window a price must fall in, either end open."""
    low: int | None = None
    high = criteria.max_price

    if criteria.target_price is not None:
        spread = criteria.target_price * criteria.price_tolerance
        low = int(criteria.target_price - spread)
        target_high = int(criteria.target_price + spread)
        high = target_high if high is None else min(high, target_high)

    return low, high


def _checks(listing: Listing, criteria: _Criteria) -> list[bool]:
    """
    One pass/fail per criterion asked for, in no particular order.

    The list length is how many things they asked for and the number of True
    values is how many this listing gives them - which is all ranking needs.
    Criteria that were not asked for are simply absent, so they can neither
    help nor hurt a listing's score.
    """
    results: list[bool] = []

    low, high = _price_band(criteria)
    if low is not None or high is not None:
        # No parseable price means nothing to compare, so this is a miss rather
        # than an optimistic hit - we will not offer an unpriced room to
        # someone who led with a number.
        price = listing.price_value
        results.append(
            price is not None
            and (low is None or price >= low)
            and (high is None or price <= high)
        )

    if criteria.location:
        results.append(_location_match(criteria.location, listing.address))

    if criteria.room_type:
        # Room Type and Bed Type together: to a lead "master" and "queen" are
        # both just the kind of room they want, not two different columns.
        results.append(_loose_match(criteria.room_type, listing.room_type, listing.bed_type))

    if criteria.bed_type:
        results.append(_loose_match(criteria.bed_type, listing.bed_type))

    if criteria.bathroom_type:
        results.append(_bathroom_match(criteria.bathroom_type, listing.bathroom_type))

    for feature in criteria.features:
        # features_text already folds in room type, bed, bathroom and parking,
        # so "parking" only matches rows that actually have a parking rental
        # and "balcony" catches the Balcony room types.
        results.append(_loose_match(feature, listing.features_text, listing.parking_rental or ""))

    return results


def _bathroom_match(keyword: str, bathroom_type: str) -> bool:
    """Whether a bathroom description means the same as the row's own value."""
    wanted = _norm(keyword)
    actual = _norm(bathroom_type)
    if not wanted or not actual:
        return False

    for canonical, synonyms in _BATHROOM_SYNONYMS.items():
        if any(word in wanted for word in synonyms):
            return canonical in actual
    return _loose_match(keyword, bathroom_type)


def _price_distance(listing: Listing, criteria: _Criteria) -> float:
    """How far this listing's price sits from whatever figure they named."""
    reference = criteria.price_reference
    if reference is None:
        return 0.0  # No price in play - every listing ties, order decides.
    if listing.price_value is None:
        return _UNKNOWN_PRICE_DISTANCE
    return abs(listing.price_value - reference)


def _widen_price(current: _Criteria, original: _Criteria, step: int) -> _Criteria:
    """
    `current` with the price window opened to relaxation `step`.

    Both widths are measured off `original`, never off the already-widened
    `current`: the rungs are alternative widths for one window, not successive
    stretches of it. Compounding them turned "under RM 700" into RM 924 at the
    bottom rung (700 x 1.10 x 1.20), which is not a near miss, it is a
    different budget.
    """
    widened = current
    if original.target_price is not None:
        widened = replace(widened, price_tolerance=_RELAXED_PRICE_TOLERANCES[step])
    if original.max_price is not None:
        widened = replace(
            widened, max_price=int(original.max_price * _RELAXED_MAX_PRICE_STRETCH[step])
        )
    return widened


# Least important constraint first: what someone would give up first if a
# letting agent said "nothing has all of that". Applied cumulatively, so step
# three has already dropped features and bathroom. Each rung takes the current
# criteria and the original ones. LOCATION IS NOT IN HERE ON PURPOSE - an area
# with no stock must come back empty so the caller can say "nothing there"
# instead of offering a room across town as a near miss.
_RELAXATION_LADDER: tuple[tuple[str, Callable[[_Criteria, _Criteria], _Criteria]], ...] = (
    ("must-have features", lambda current, original: replace(current, features=())),
    ("bathroom type", lambda current, original: replace(current, bathroom_type="")),
    ("room and bed type", lambda current, original: replace(current, room_type="", bed_type="")),
    ("price range", lambda current, original: _widen_price(current, original, 0)),
    ("price range", lambda current, original: _widen_price(current, original, 1)),
)


def _relaxation_steps(criteria: _Criteria):
    """
    Yield (criteria, what_was_relaxed) progressively loosened, exact first.

    A rung that would not actually change anything - dropping features when
    none were asked for - is skipped rather than yielded as a duplicate search,
    so `what_was_relaxed` only ever names constraints that were really given up.
    """
    yield criteria, ()

    current = criteria
    relaxed: list[str] = []
    for label, loosen in _RELAXATION_LADDER:
        widened = loosen(current, criteria)
        if widened == current:
            continue
        current = widened
        if label not in relaxed:
            relaxed.append(label)
        yield current, tuple(relaxed)


@dataclass(frozen=True)
class SearchResult:
    """
    What search_listings gives back: the rows, and how literally to take them.

    `is_exact` is the whole point of the type. False means the criteria as
    stated matched nothing and these are the closest the pool gets, with
    `relaxed_criteria` naming what had to be given up ("price range",
    "must-have features") - so a caller can say "nothing exactly matches, but"
    rather than presenting a near miss as the thing that was asked for.

    Iterates and lens as the list of listings, so callers that only care about
    the rows can treat it as one.
    """

    listings: tuple[Listing, ...]
    is_exact: bool
    relaxed_criteria: tuple[str, ...] = ()

    def __iter__(self):
        return iter(self.listings)

    def __len__(self) -> int:
        return len(self.listings)

    def __bool__(self) -> bool:
        return bool(self.listings)


def search_listings(
    max_price: int | None = None,
    location_keyword: str | None = None,
    room_type_keyword: str | None = None,
    target_price: int | None = None,
    bed_type_keyword: str | None = None,
    bathroom_type_keyword: str | None = None,
    must_have_features: list[str] | tuple[str, ...] | None = None,
    exclude_row_index: int | None = None,
    limit: int = 4,
) -> SearchResult:
    """
    Up to `limit` listings for a lead's stated preferences, best first.

    Every criterion is optional and they combine with AND:

        max_price             a ceiling - price_value <= max_price.
        target_price          a figure they named loosely ("around RM 1,000").
                              Matches a band either side of it, so somewhat
                              above as well as below, not just under.
        location_keyword      the area, matched against the address.
        room_type_keyword     the kind of room, matched against Room Type OR
                              Bed Type - "master" and "queen" are both just
                              "the sort of room I want" to a lead.
        bed_type_keyword      the bed specifically, when they distinguish.
        bathroom_type_keyword "shared", "private", "ensuite", "own" and so on,
                              mapped onto the two values the column holds.
        must_have_features    keywords like ["parking", "balcony"], each
                              matched against features_text and the parking
                              rental.

    Text criteria match forgivingly (case, punctuation and word-by-word), so a
    phrase a person would actually type finds the rows a column value spells
    differently.

    RANKING: results are ordered by how many of the stated criteria each
    listing satisfies, then by how close its price sits to whatever figure they
    named, then by CSV order. Ties are broken deterministically, so the same
    question twice running gets the same answer.

    RELAXATION: if nothing satisfies all of it, the least important constraint
    is dropped and the search retried - features first, then bathroom type,
    then room and bed type, then the price window widened in two steps. The
    result comes back with is_exact=False and relaxed_criteria naming what was
    given up. THE AREA IS NEVER RELAXED: if they named somewhere we have no
    rooms, the honest answer is that we have nothing there, not a room on the
    other side of the city. Scoring is always against what they ORIGINALLY
    asked for, so even in a relaxed pass the listing that misses least ranks
    first.

    `exclude_row_index` always drops that one row, criteria or not: it is the
    property the lead is already looking at, and offering it back to them as an
    "alternative" would read as a bug. With no criteria at all we return the
    first `limit` listings, so a caller that could not pull anything out of the
    conversation still has something real to show.
    """
    if limit <= 0:
        return SearchResult((), True, ())

    # Blank strings arrive from callers passing a model's "" straight through;
    # treat them as "not asked for" rather than as a substring matching all.
    criteria = _Criteria(
        max_price=max_price,
        target_price=target_price,
        location=(location_keyword or "").strip(),
        room_type=(room_type_keyword or "").strip(),
        bed_type=(bed_type_keyword or "").strip(),
        bathroom_type=(bathroom_type_keyword or "").strip(),
        features=tuple(
            feature.strip()
            for feature in (must_have_features or [])
            if isinstance(feature, str) and feature.strip()
        ),
    )

    pool = [
        listing
        for listing in load_listings()
        if exclude_row_index is None or listing.row_index != exclude_row_index
    ]

    if not criteria.has_any:
        return SearchResult(tuple(pool[:limit]), True, ())

    for attempt, relaxed in _relaxation_steps(criteria):
        matches = [listing for listing in pool if all(_checks(listing, attempt))]
        if not matches:
            continue

        # Scored against `criteria`, never `attempt`: in a relaxed pass the
        # rows that still happen to satisfy the dropped constraint are exactly
        # the ones worth showing first.
        matches.sort(
            key=lambda listing: (
                -sum(_checks(listing, criteria)),
                _price_distance(listing, criteria),
                listing.row_index,
            )
        )
        return SearchResult(tuple(matches[:limit]), not relaxed, relaxed)

    return SearchResult((), True, ())


def random_pool_photo() -> str:
    """A random file path from the demo unit-photo pool."""
    photos = sorted(p for p in PHOTO_POOL_DIR.iterdir() if p.is_file())
    if not photos:
        raise FileNotFoundError(f"No photos found in {PHOTO_POOL_DIR}")
    return str(random.choice(photos))
