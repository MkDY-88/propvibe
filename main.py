"""
PropVibe - FastAPI entrypoint.

Exposes the poster generator over HTTP so the front end (or a webhook, or a
plain curl) can post some property photos plus listing details and get a
ready-to-share 1080x1080 PNG straight back in the response body.

Endpoints:
  GET  /              - landing page (static HTML)
  GET  /dashboard     - minimal approval/preview page (static HTML)
  GET  /privacy       - privacy policy page (Facebook App Live mode requirement)
  GET  /business      - business model page (static HTML)
  POST /generate-poster - returns the poster PNG directly
  POST /create-post   - returns poster (base64) + Claude-written caption as JSON
  POST /publish-post  - publishes a poster + caption to a Facebook Page
  GET  /browse        - public card grid of every posted listing (static HTML)
  GET  /posted-listings - the data behind /browse: posted listings, as JSON
  GET  /listing-info/{tracking_id} - the listing behind a tracking link, as JSON
  POST /chat          - AI leasing assistant for a lead on the listing page
  POST /sync-engagement - refreshes engagement stats (requires X-Sync-Secret header)
  GET  /internal-dashboard - internal read-only sales-flow view (secret-gated)
  GET  /daily-report  - end-of-day numbers + AI-written summary, as JSON
  GET  /report        - the page that renders /daily-report (static HTML)
  GET  /public-stats  - aggregate-only live totals for the landing page (no auth)
"""

import base64
import binascii
import logging
import os
import random
import secrets
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from fastapi import (
    Body,
    Cookie,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import FileResponse, RedirectResponse, Response
from starlette.background import BackgroundTask

# Load variables from a local .env file (e.g. ANTHROPIC_API_KEY,
# FACEBOOK_PAGE_ACCESS_TOKEN) into the process environment at import time.
#
# We point at the .env sitting next to THIS file rather than relying on
# load_dotenv()'s default CWD search: when the server is launched from a parent
# directory (e.g. `uvicorn main:app --app-dir propvibe`, or on Railway) that
# search starts above the app folder and silently finds nothing. Resolving
# relative to __file__ - the same trick UPLOAD_ROOT/STATIC_DIR use below - makes
# config load no matter where the process was started from.
#
# This stays a no-op in production where the platform injects real env vars (the
# file simply won't exist), and it deliberately does NOT raise if a key is
# missing - endpoints that actually need one validate at call time and return a
# clean error instead of crashing the whole app on startup.
load_dotenv(Path(__file__).resolve().parent / ".env")

# Registers a Pillow opener for .HEIC/.HEIF (iPhone photos), which Pillow can't
# decode on its own. Must happen before any Image.open() call sees one - so at
# import time here, not lazily inside a request handler. Best-effort: if the
# package is missing, HEIC uploads just fail their existing "could not be read
# as an image" check in poster_generator instead of crashing startup.
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    logging.getLogger("propvibe.main").warning(
        "pillow-heif is not installed - .HEIC/.HEIF photos will be rejected as unreadable."
    )

# NOTE: this is the `from X import Y` form on purpose - it binds only
# `generate_poster`, so the `app` package name never lands in this module's
# namespace and cannot shadow the `app = FastAPI()` instance below.
from app.poster_generator import (
    GRID_TEMPLATE_MIN_PHOTOS,
    add_disclosure_label,
    check_photos,
    generate_poster,
    photo_as_jpeg_bytes,
)
from app.copy_generator import STYLE_TAGS, CaptionError, generate_caption
from app.lead_chatbot import ChatError, extract_search_intent, qualify_lead
from app.photo_condition import FalEditError, TIME_OF_DAY_VALUES, WEATHER_VALUES, edit_photo_condition
from app.mascot_video import router as mascot_video_router
from app.trend_research import research_trend
from app.listings_source import (
    PHOTO_POOL_DIR,
    Listing,
    get_listing,
    load_listings,
    next_unposted_listing,
    pool_photo_for_listing,
    search_listings,
)
from app.facebook_publisher import (
    FacebookPublishError,
    get_post_engagement,
    publish_post,
)
from app.click_tracker import (
    get_clicks,
    get_tracking_photo,
    get_tracking_row_index,
    link_post,
    record_click,
    save_tracking_listing,
    save_tracking_photo,
    tracking_id_for_post,
)
from app.airtable_client import (
    AirtableError,
    best_performing_style,
    create_post_record,
    get_photo_condition_cache_entries,
    get_posts,
    # Aliased: bare `is_configured` would read ambiguously in this module, which
    # talks to Airtable, Facebook and fal.ai.
    is_configured as airtable_is_configured,
    latest_clicks_by_post,
    log_engagement,
    save_cached_photo_condition,
    total_clicks_recorded,
    upsert_lead_record,
)
from app.internal_dashboard import collect_dashboard_data
from app.daily_report import ReportError, collect_report_data, generate_daily_report

logger = logging.getLogger("propvibe.main")

app = FastAPI()

# Mascot Video Studio (stretch feature, dev-only): registers GET /mascot-studio
# and its own endpoints. Everything that feature needs lives in
# app/mascot_video.py - this line is its only contact with the rest of the app,
# and it is not linked from any nav.
app.include_router(mascot_video_router)

# Scratch space for uploads. Resolved relative to THIS file rather than the
# working directory so it behaves the same locally and on Railway. Gitignored -
# every request gets its own subfolder which is deleted once the response has
# been streamed.
UPLOAD_ROOT = Path(__file__).resolve().parent / "uploads"

# Static assets (the approval/preview dashboard). Resolved relative to THIS file
# so it works the same locally and once deployed.
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Anything larger than this per photo is almost certainly not a listing shot.
MAX_PHOTO_BYTES = 15 * 1024 * 1024  # 15 MB

# Public base URL of THIS server, used to build the tracking links embedded in
# posts. Locally the default is fine; in production (Railway) set BASE_URL to the
# deployed public domain so the link a lead clicks resolves back here. See
# .env.example.
BASE_URL = "http://localhost:8000"


def _base_url() -> str:
    """The server's public base URL (env BASE_URL), without a trailing slash."""
    configured = os.environ.get("BASE_URL")
    return (configured.strip() if configured and configured.strip() else BASE_URL).rstrip("/")


def _generate_tracking_id() -> str:
    """
    A short, URL-safe, hard-to-guess id for a post's tracking link.

    ``token_urlsafe(6)`` yields 8 characters from [A-Za-z0-9_-] - short enough to
    tack onto a CTA, random enough to not collide across a demo's worth of posts.
    """
    return secrets.token_urlsafe(6)


def _tracking_url(tracking_id: str) -> str:
    """The full clickable tracking link for a post, e.g. https://host/t/ab12cd34."""
    return f"{_base_url()}/t/{tracking_id}"


@app.get("/")
def read_root():
    """Serve the PropVibe landing page, same FileResponse pattern as /dashboard."""
    page = STATIC_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Home page not found.")
    return FileResponse(page, media_type="text/html")


@app.get("/dashboard")
def dashboard():
    """Serve the minimal approval/preview page."""
    page = STATIC_DIR / "dashboard.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Dashboard page not found.")
    return FileResponse(page, media_type="text/html")


@app.get("/listing")
def listing():
    """
    The placeholder 'listing details' landing page a tracking link redirects to.

    Stands in for wherever a real lead would land after clicking through from a
    social post.
    """
    page = STATIC_DIR / "listing.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Listing page not found.")
    return FileResponse(page, media_type="text/html")


@app.get("/browse")
def browse():
    """Serve the public Browse page - a card grid of every posted listing."""
    page = STATIC_DIR / "browse.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Browse page not found.")
    return FileResponse(page, media_type="text/html")


@app.get("/posted-listings")
def posted_listings_endpoint():
    """
    Every listing that has actually been published, for /browse's card grid.

    "Posted" means an Airtable Posts row carrying BOTH a tracking id and a
    listing row index - the same durable record /auto-create-post reads to
    decide what is still un-posted, so the two views can never disagree. Rows
    without either are skipped: no row index means no listing details to show
    (the manual-upload flow), and no tracking id means no landing page to
    link the card to.

    Returns::

        {"listings": [
          {"tracking_id": "ab12cd34", "condo_name": "Sunway Serene",
           "room_type": "Master", "price": "RM 1,000", "address": "..."},
          ...
        ]}

    Each card links straight to /listing?tid=<tracking_id> - the real landing
    page (photo, details, chat) a lead reaches from Facebook. Deliberately NOT
    via /t/<id>: that route records a click, and browsing must not inflate
    the engagement numbers.

    A listing posted more than once appears once, under its most recent
    post's tracking id. Zero posted listings is an empty list (the page's
    honest "nothing posted yet" state); an unreadable Airtable is a 502,
    kept distinct so an outage never masquerades as an empty catalogue.
    """
    try:
        posts = get_posts()
    except AirtableError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not read the posted listings: {exc}",
        )

    # get_posts() returns rows in Airtable's creation order, so walking it
    # reversed puts the newest posts first AND makes the first tracking id
    # seen for a row index the most recent one - `seen` then drops the older
    # duplicates of a re-posted listing.
    listings_by_index = {listing.row_index: listing for listing in load_listings()}
    seen: set[int] = set()
    cards: list[dict] = []
    for post in reversed(posts):
        tracking_id = post.get("tracking_id")
        row_index = post.get("listing_row_index")
        if not tracking_id or row_index is None or row_index in seen:
            continue
        listing = listings_by_index.get(row_index)
        if listing is None:
            # A post whose row has since left the CSV - nothing to render.
            continue
        seen.add(row_index)
        cards.append(
            {
                "tracking_id": tracking_id,
                "condo_name": listing.condo_name,
                "room_type": listing.room_type,
                "price": listing.price,
                "address": listing.address,
            }
        )

    return {"listings": cards}


@app.get("/privacy")
def privacy():
    """Serve the privacy policy page (required for Facebook App Live mode)."""
    page = STATIC_DIR / "privacy.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Privacy policy page not found.")
    return FileResponse(page, media_type="text/html")


@app.get("/business")
def business():
    """Serve the business model page."""
    page = STATIC_DIR / "business.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Business model page not found.")
    return FileResponse(page, media_type="text/html")


@app.get("/t/{tracking_id}")
def track_click(tracking_id: str):
    """
    Record a click on a post's tracking link, then redirect to the listing page.

    Every published post's CTA carries a ``/t/{tracking_id}`` link. When a lead
    clicks it we bump a local counter (see app.click_tracker) keyed by that id -
    fast, and independent of Airtable being reachable - then 302 them on to the
    placeholder listing page. /sync-engagement later folds these click counts
    into each post's Airtable engagement row.

    The tracking id rides along to the landing page as ``?tid=`` so the page
    (and the chatbot on it) can look up WHICH listing this lead clicked
    through from - see /listing-info and /chat.
    """
    record_click(tracking_id)
    # quote() so a hand-crafted id with '#' or '&' in it can't mangle the query
    # string; a real token_urlsafe id passes through unchanged.
    return RedirectResponse(url=f"/listing?tid={quote(tracking_id)}", status_code=302)


def _row_index_for_tracking(tracking_id: str) -> int | None:
    """
    The room_listings.csv row index a tracking id was published for, or None.

    The local map (app.click_tracker) is checked first - free, and covering
    every post this process has published. On a miss we fall back to
    Airtable's Posts table, which records the same tracking_id ->
    listing_row_index pair durably at publish time: the local map lives on an
    ephemeral disk and empties on every restart or redeploy, and without the
    fallback every post published before the last restart greets its leads
    with generic copy instead of the real listing - a degradation /browse
    makes prominent, since its cards link straight to these pages. A
    successful Airtable resolution is written back into the local map so the
    next lookup for the same id stays local.

    Best-effort on the Airtable leg: an unreadable table degrades to None
    (generic copy), never an error - a lead's landing page must not 500 over
    bookkeeping.
    """
    row_index = get_tracking_row_index(tracking_id)
    if row_index is not None:
        return row_index

    if not (tracking_id or "").strip():
        return None

    try:
        posts = get_posts()
    except AirtableError as exc:
        logger.warning(
            "Could not resolve tracking id %r from Airtable: %s", tracking_id, exc
        )
        return None

    for post in posts:
        if (
            post.get("tracking_id") == tracking_id
            and post.get("listing_row_index") is not None
        ):
            save_tracking_listing(tracking_id, post["listing_row_index"])
            return post["listing_row_index"]
    return None


def _listing_context(tracking_id: str) -> dict | None:
    """
    The listing a tracking link was published for, as a plain dict, or None.

    Two hops: tracking_id -> row index (the local publish-time record, with a
    durable Airtable fallback - see _row_index_for_tracking) -> Listing
    (re-read from room_listings.csv). Returns None whenever either hop comes
    up empty - an unrecognised id, a post from before we recorded listing
    indices, or a row that has since left the CSV. Shared by /listing-info
    and /chat so the two can never disagree about which property a lead is
    looking at.
    """
    row_index = _row_index_for_tracking(tracking_id)
    if row_index is None:
        return None

    listing = get_listing(row_index)
    if listing is None:
        return None

    return _listing_to_context(listing)


def _listing_to_context(listing: Listing) -> dict:
    """
    A Listing as the plain dict the chatbot and the landing page consume.

    One definition of that shape, used for the property a lead clicked through
    for AND for any alternatives we search up in /chat, so the assistant can
    never be handed two differently-shaped listings.
    """
    return {
        "condo_name": listing.condo_name,
        "price": listing.price,
        "address": listing.address,
        "features_text": listing.features_text,
    }


@app.get("/listing-info/{tracking_id}")
def listing_info(tracking_id: str):
    """
    The listing details behind a tracking link, for the landing page to render.

    Returns JSON::

        {"found": true, "condo_name": "...", "price": "RM 1,000",
         "address": "...", "features_text": "Master, Queen bed, ..."}

    A tracking id we have no listing for is NOT a 404 - it returns
    ``{"found": false}`` so the page can quietly fall back to generic copy.
    Only a genuinely unrecognised link should look like an error, and from a
    lead's point of view an old post isn't one.
    """
    context = _listing_context(tracking_id)
    if context is None:
        return {"found": False}
    return {"found": True, **context, "photo_filename": _photo_for_tracking(tracking_id)}


def _photo_for_tracking(tracking_id: str) -> str | None:
    """
    The pool photo to show on a tracking link's landing page, or None.

    The filename recorded at publish time when we still have it. When that
    local record has been lost (ephemeral disk - same story as
    _row_index_for_tracking), the deterministic pool assignment for the
    listing's row instead: pool_photo_for_listing() is exactly how
    /auto-create-post picked the photo at publish time, so this reproduces
    the photo the post really used rather than substituting a random one.
    """
    recorded = get_tracking_photo(tracking_id)
    if recorded:
        return recorded

    row_index = _row_index_for_tracking(tracking_id)
    if row_index is None:
        return None
    try:
        return Path(pool_photo_for_listing(row_index)).name
    except FileNotFoundError:
        return None


def _parse_count(raw: str | None, field: str) -> int:
    """
    Turn a form field into a non-negative int, or raise a clean 400.

    Form values always arrive as strings, so "3" is fine but "", "three" and
    "3.5" all need to come back as a readable error rather than a stack trace.
    """
    if raw is None or not raw.strip():
        raise HTTPException(status_code=400, detail=f"'{field}' is required.")
    try:
        value = int(raw.strip())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"'{field}' must be a whole number, got {raw!r}.",
        )
    if value < 0:
        raise HTTPException(status_code=400, detail=f"'{field}' cannot be negative.")
    return value


def _require_text(raw: str | None, field: str) -> str:
    """Non-empty string, or a clean 400."""
    if raw is None or not raw.strip():
        raise HTTPException(status_code=400, detail=f"'{field}' is required.")
    return raw.strip()


def _usable_photos(saved_paths: list[str]) -> list[str]:
    """
    Drop the uploads we can't decode, or raise a clean 400 if none are usable.

    The poster generator falls back to a flat grey block for a photo it can't
    open. That keeps one bad upload from killing the job, but it is not a
    result worth shipping: the grey rectangle lands in the finished poster, and
    when every photo fails the "poster" is a plain grey square with a price on
    it - returned as a cheerful 200. So we check up front, build from whatever
    survives, and refuse outright when nothing does.

    Partial failures are dropped silently (logged, not surfaced): the caller
    still gets a good poster from their good photos, which is the useful
    outcome. The size and emptiness checks in _save_uploads run earlier and
    already reject their own cases with a clean 400.
    """
    usable, problems = check_photos(saved_paths)

    if not usable:
        raise HTTPException(
            status_code=400,
            detail="None of the uploaded photos could be used. " + " ".join(problems),
        )

    if problems:
        logger.warning(
            "Building poster from %d of %d photos; skipped: %s",
            len(usable),
            len(saved_paths),
            " ".join(problems),
        )

    return usable


@app.post("/generate-poster")
# Deliberately `def`, not `async def`. Everything below blocks - reading the
# uploads, Pillow rasterising the poster - and blocking work inside an async
# endpoint occupies the event loop, which serialises every other request behind
# it. Declared sync, FastAPI runs it in its threadpool and concurrent requests
# actually overlap. See the note on /create-post.
def generate_poster_endpoint(
    # Every field is declared optional and validated by hand below. If we let
    # FastAPI enforce them, a missing field would come back as a 422 with a
    # nested Pydantic error body - the brief asks for a plain 400 and a
    # human-readable message.
    photos: list[UploadFile] | None = File(None),
    price: str | None = Form(None),
    location: str | None = Form(None),
    bedrooms: str | None = Form(None),
    bathrooms: str | None = Form(None),
):
    """
    Build a property poster from uploaded photos and return it as a PNG.

    Send `multipart/form-data` with one or more `photos` parts plus `price`,
    `location`, `bedrooms` and `bathrooms`. The response is the raw image, so
    hitting this URL in a browser renders the poster inline.
    """
    price = _require_text(price, "price")
    location = _require_text(location, "location")
    bedroom_count = _parse_count(bedrooms, "bedrooms")
    bathroom_count = _parse_count(bathrooms, "bathrooms")

    # An HTML form that submits an empty file input still sends a `photos` part,
    # it just has no filename - so filter those out before counting.
    uploads = [item for item in (photos or []) if item is not None and item.filename]
    if not uploads:
        raise HTTPException(
            status_code=400,
            detail="At least one photo is required. Attach it as a 'photos' form field.",
        )

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    job_dir = Path(tempfile.mkdtemp(prefix="job_", dir=UPLOAD_ROOT))

    # From here on any failure must still clean up the job folder, otherwise a
    # bad request would leak a directory on every call.
    try:
        saved_paths: list[str] = []
        for index, upload in enumerate(uploads, start=1):
            # Never trust the client-supplied filename for a path - keep only
            # the extension and generate our own name.
            suffix = Path(upload.filename).suffix[:10]
            destination = job_dir / f"photo_{index:02d}{suffix}"

            with destination.open("wb") as handle:
                shutil.copyfileobj(upload.file, handle, length=1024 * 1024)

            if destination.stat().st_size == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Photo {index} ({upload.filename!r}) is empty.",
                )
            if destination.stat().st_size > MAX_PHOTO_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Photo {index} ({upload.filename!r}) is larger than "
                        f"{MAX_PHOTO_BYTES // (1024 * 1024)} MB."
                    ),
                )

            saved_paths.append(str(destination))

        usable_paths = _usable_photos(saved_paths)

        output_path = job_dir / f"poster_{uuid.uuid4().hex[:8]}.png"

        try:
            generate_poster(
                photos=usable_paths,
                price=price,
                location=location,
                bedrooms=bedroom_count,
                bathrooms=bathroom_count,
                output_path=str(output_path),
            )
        except ValueError as exc:
            # generate_poster() raises this for an empty photo list. We already
            # guard that above, so this is belt-and-braces.
            raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    # The file has to outlive this function so Starlette can stream it, so the
    # cleanup runs as a background task once the response is fully sent.
    return FileResponse(
        path=output_path,
        media_type="image/png",
        # `inline` (rather than FileResponse's `filename=`, which forces an
        # attachment) is what makes the poster render in a browser tab.
        headers={"Content-Disposition": 'inline; filename="poster.png"'},
        background=BackgroundTask(shutil.rmtree, job_dir, ignore_errors=True),
    )


def _save_uploads(uploads: list[UploadFile], job_dir: Path) -> list[str]:
    """
    Persist each upload into `job_dir`, validating size, and return the paths.

    Mirrors the per-photo handling in /generate-poster: we never trust the
    client filename (keep only the extension, generate our own name), reject
    empty or oversized files with a clean 400, and hand back the saved paths in
    order. The caller owns `job_dir` and is responsible for cleaning it up.
    """
    saved_paths: list[str] = []
    for index, upload in enumerate(uploads, start=1):
        suffix = Path(upload.filename).suffix[:10]
        destination = job_dir / f"photo_{index:02d}{suffix}"

        with destination.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle, length=1024 * 1024)

        size = destination.stat().st_size
        if size == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Photo {index} ({upload.filename!r}) is empty.",
            )
        if size > MAX_PHOTO_BYTES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Photo {index} ({upload.filename!r}) is larger than "
                    f"{MAX_PHOTO_BYTES // (1024 * 1024)} MB."
                ),
            )

        saved_paths.append(str(destination))

    return saved_paths


def _template_id_for(photo_count: int) -> str:
    """Which poster template generate_poster() will pick for this many photos."""
    return "Template A" if photo_count < GRID_TEMPLATE_MIN_PHOTOS else "Template B"


@app.post("/create-post")
# Deliberately `def`, not `async def`. This endpoint contains no `await` at all:
# saving uploads, Pillow, the trend web search and the caption call are every
# one of them blocking, and the Anthropic SDK client used here is the sync one.
# Run as `async def` that work sat on uvicorn's single event loop, so five
# simultaneous requests were served strictly one after another - measured at
# 35.3s wall for five requests against a 7.0s single-request baseline, i.e. no
# overlap whatsoever. As a plain `def`, FastAPI dispatches it to its threadpool
# and the same five overlap properly.
def create_post_endpoint(
    # Same manual-validation approach as /generate-poster so a missing field is
    # a plain 400 with a readable message rather than a nested 422 body.
    photos: list[UploadFile] | None = File(None),
    price: str | None = Form(None),
    location: str | None = Form(None),
    bedrooms: str | None = Form(None),
    bathrooms: str | None = Form(None),
):
    """
    Build the poster AND the social caption in one call.

    Same multipart inputs as /generate-poster. Before building either one, we
    resolve a single style_tag for the whole request - whichever style has been
    performing best in Airtable, or a random pick from STYLE_TAGS when there's
    no data yet - and pass that identical value into both generate_poster()
    (which picks a colour palette) and generate_caption() (which picks a tone),
    so the two can never disagree. Returns JSON:

        {
          "caption":       "<engaging post text>",
          "hashtags":      ["luxuryliving", ...],   # no '#' prefix
          "cta":           "DM us to book a viewing",
          "style_tag":     "Warm",                  # style the caption was written in
          "template_id":   "Template A",            # poster template used
          "trend_used":    true,                    # did live trend research feed in?
          "trend_context": "<current market note>", # null when it didn't
          "poster_base64": "<base64 PNG bytes>"     # no data: prefix
        }

    style_tag and template_id are echoed back so the front end can hand them to
    /publish-post, which records them in Airtable for the learning loop.

    Before writing the caption we also web-search the location for a current
    market note (app.trend_research) and pass it to generate_caption() as an
    optional hint. That step is best-effort: if it fails or finds nothing,
    trend_context is null and the caption is generated exactly as before.

    The poster is inlined as base64 so the front end can render it immediately
    without a separate file-download route.
    """
    price = _require_text(price, "price")
    location = _require_text(location, "location")
    bedroom_count = _parse_count(bedrooms, "bedrooms")
    bathroom_count = _parse_count(bathrooms, "bathrooms")

    uploads = [item for item in (photos or []) if item is not None and item.filename]
    if not uploads:
        raise HTTPException(
            status_code=400,
            detail="At least one photo is required. Attach it as a 'photos' form field.",
        )

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    job_dir = Path(tempfile.mkdtemp(prefix="job_", dir=UPLOAD_ROOT))

    # Unlike /generate-poster (which streams a file and cleans up afterwards via
    # a background task), we read the poster into memory here and return JSON, so
    # the job folder can be removed synchronously in `finally`.
    try:
        saved_paths = _save_uploads(uploads, job_dir)
        usable_paths = _usable_photos(saved_paths)

        # Resolve ONE style for the whole request, before either generator runs:
        # whichever style has performed best so far, or a random pick from
        # STYLE_TAGS when there's no engagement data yet. Doing this here (not
        # inside generate_poster()/generate_caption()) guarantees the poster's
        # palette and the caption's tone always match on a given post.
        style_tag = best_performing_style() or random.choice(STYLE_TAGS)

        output_path = job_dir / f"poster_{uuid.uuid4().hex[:8]}.png"
        try:
            generate_poster(
                photos=usable_paths,
                price=price,
                location=location,
                bedrooms=bedroom_count,
                bathrooms=bathroom_count,
                output_path=str(output_path),
                style_tag=style_tag,
            )
        except ValueError as exc:
            # Belt-and-braces: generate_poster() raises this for an empty photo
            # list, which we already guard above.
            raise HTTPException(status_code=400, detail=str(exc))

        poster_base64 = base64.b64encode(output_path.read_bytes()).decode("ascii")

        # Ask Claude to web-search what's currently happening in this area, so
        # the caption can reference something real rather than evergreen filler.
        # research_trend() never raises and returns None on any failure (no API
        # key, timeout, nothing found), so this is a straight assignment - a
        # None here just means the caption is generated exactly as it was
        # before. It's also internally capped at a few seconds and cached per
        # location per day, so it can't stall the response for long or re-search
        # on every post for the same area.
        trend_context = research_trend(location)

        # The caption call hits the network and may fail for reasons the user
        # can act on (missing key, rate limit, ...). Surface the clean message.
        try:
            copy = generate_caption(
                price,
                location,
                bedroom_count,
                bathroom_count,
                preferred_style=style_tag,
                trend_context=trend_context,
            )
        except CaptionError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

    return {
        "caption": copy["caption"],
        "hashtags": copy["hashtags"],
        "cta": copy["cta"],
        "style_tag": copy["style_tag"],
        # Counted from the photos the poster was actually built from, not from
        # how many arrived: a request that uploads 4 photos of which 2 are
        # unreadable gets Template A, and Airtable has to record the template
        # that was really used or the learning loop attributes engagement to
        # the wrong one.
        "template_id": _template_id_for(len(usable_paths)),
        # The live trend note the caption was written with, or null when the
        # search found nothing / wasn't available. Echoed back so the front end
        # (and a demo audience) can see the research actually ran and what it
        # found, rather than taking "the AI decided" on faith.
        "trend_used": trend_context is not None,
        "trend_context": trend_context,
        "poster_base64": poster_base64,
    }


@app.get("/search-listings")
def search_listings_endpoint(q: str = Query("")):
    """
    Up to 8 listings whose condo name or address matches the free-text `q`,
    for the dashboard's search box. Reuses the lead chatbot's search_listings()
    (its name_or_address_keyword criterion) rather than a second matcher, so
    "casa ti" and "birch ipoh" match the same forgiving way the chatbot does.

    Returns just enough to render a pick-list; the click itself goes through
    /auto-create-post?row_index=... which returns the full listing details.
    """
    query = q.strip()
    if not query:
        # An empty box means "not searching", not "show me everything".
        return {"results": []}

    matches = search_listings(name_or_address_keyword=query, limit=8)
    return {
        "results": [
            {
                "row_index": listing.row_index,
                "condo_name": listing.condo_name,
                # The distinguisher: the CSV holds several units per building
                # ("Sunway Serene" x4), so name+price+address alone renders as
                # near-identical rows. The dashboard shows this as
                # "Condo Name — Room Type", the same header format the
                # approve/skip card already uses.
                "room_type": listing.room_type,
                "price": listing.price,
                "address": listing.address,
            }
            for listing in matches
        ]
    }


@app.post("/auto-create-post")
# `def`, not `async def` - same reasoning as /create-post: Pillow, the trend
# search and the caption call are all blocking, so this needs FastAPI's
# threadpool to let concurrent requests actually overlap.
def auto_create_post_endpoint(
    # Set by the dashboard's Skip button to the row_index of the listing
    # currently on screen, so the next candidate is a *different* listing
    # rather than the same one again. No server-side state for skips - the
    # value is only ever passed through on this one request.
    after: int | None = Query(None),
    # Set by the dashboard's search box to load one exact listing instead of
    # the next-in-sequence candidate. Takes precedence over `after`. An
    # already-posted row is allowed through (the agent may genuinely want to
    # re-post it) but flagged in the response so the frontend can warn.
    row_index: int | None = Query(None),
):
    """
    Auto-ready the next un-posted room_listings.csv row: pick a listing,
    assign a demo photo, and build the same poster + caption /create-post
    does - with no photo upload or form fields required.

    With `row_index`, load exactly that row instead (the dashboard's search
    box picking a specific listing); everything downstream - deterministic
    photo, poster, caption - is identical to the sequential path.

    "Un-posted" is read from Airtable (see app.airtable_client.get_posts's
    listing_row_index), not a local file, because Railway's filesystem is
    ephemeral - a local tracker would reset on every redeploy and risk
    re-posting an already-published listing straight back to Facebook.

    NOTE: this requires a "Listing Row Index" Number column on the Airtable
    Posts table (or the AIRTABLE_FIELD_LISTING_ROW_INDEX override pointing at
    an existing column). Without it, create_post_record() silently drops the
    field (same defensive behaviour as every other field in airtable_client),
    and "un-posted" tracking has no way to durably progress past whatever
    starting rows.

    Returns the same JSON shape as /create-post, plus the listing's own
    details (`row_index`, `condo_name`, `price`, `address`, `features`) so the
    dashboard can render a single approve/skip card without a manual form.
    """
    try:
        posted = get_posts()
    except AirtableError as exc:
        # Direct loads only need Airtable for the already_posted flag, but a
        # wrong "not posted yet" is worse than an error: the whole point of
        # the flag is stopping an unnoticed duplicate going to Facebook.
        raise HTTPException(
            status_code=502,
            detail=f"Could not read Airtable's posted listings: {exc}",
        )

    posted_indices = {
        post["listing_row_index"]
        for post in posted
        if post.get("listing_row_index") is not None
    }

    if row_index is not None:
        listing = get_listing(row_index)
        if listing is None:
            raise HTTPException(
                status_code=404,
                detail=f"No listing exists at row_index {row_index}.",
            )
    else:
        listing = next_unposted_listing(posted_indices, after=after)
        if listing is None:
            raise HTTPException(
                status_code=404,
                detail="No un-posted listings remain in room_listings.csv.",
            )

    try:
        photo_path = pool_photo_for_listing(listing.row_index)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Same reused check as the manual flow: confirms the photo actually
    # decodes (e.g. a .HEIC file when pillow-heif isn't installed) rather than
    # silently shipping a grey placeholder rectangle.
    usable_paths = _usable_photos([photo_path])

    style_tag = best_performing_style() or random.choice(STYLE_TAGS)

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    job_dir = Path(tempfile.mkdtemp(prefix="auto_", dir=UPLOAD_ROOT))
    try:
        output_path = job_dir / f"poster_{uuid.uuid4().hex[:8]}.png"
        try:
            generate_poster(
                photos=usable_paths,
                price=listing.price,
                location=listing.address,
                bedrooms=0,
                bathrooms=0,
                output_path=str(output_path),
                style_tag=style_tag,
                details=listing.features_text,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        poster_base64 = base64.b64encode(output_path.read_bytes()).decode("ascii")

        trend_context = research_trend(listing.address)

        try:
            copy = generate_caption(
                listing.price,
                listing.address,
                0,
                0,
                preferred_style=style_tag,
                trend_context=trend_context,
                listing_details=listing.features_text,
            )
        except CaptionError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

    return {
        "row_index": listing.row_index,
        "condo_name": listing.condo_name,
        "price": listing.price,
        "address": listing.address,
        "features": listing.features_text,
        # Always False on the sequential path (next_unposted_listing skips
        # posted rows by construction); only a search-loaded listing can be
        # True. The frontend warns but doesn't block - re-posting may be
        # exactly what the agent wants, it just mustn't happen unnoticed.
        "already_posted": listing.row_index in posted_indices,
        "caption": copy["caption"],
        "hashtags": copy["hashtags"],
        "cta": copy["cta"],
        "style_tag": copy["style_tag"],
        "template_id": _template_id_for(len(usable_paths)),
        "trend_used": trend_context is not None,
        "trend_context": trend_context,
        "poster_base64": poster_base64,
        # The demo photo actually used, so an edit to price/location/features
        # can ask /regenerate-poster to redraw the same photo with new text
        # instead of picking a different one at random.
        "photo_filename": Path(photo_path).name,
    }


@app.post("/regenerate-poster")
# `def`, not `async def` - same reasoning as /auto-create-post: generate_poster()
# is blocking Pillow work, so this needs FastAPI's threadpool.
def regenerate_poster_endpoint(payload: dict = Body(...)):
    """
    Redraw the auto-flow's poster text overlay with edited price/location/
    features, reusing the exact demo photo /auto-create-post already picked.

    Send JSON:

        {
          "photo_filename": "unit_012.jpg",  # from /auto-create-post's response
          "price": "RM 1,200",
          "location": "Mont Kiara, Kuala Lumpur",
          "features": "Master, Queen bed, Private bathroom",
          "style_tag": "Warm"                # optional - from /auto-create-post
        }

    This is the dashboard Edit step's Save action for price/location/features:
    those three are Pillow-drawn pixels, not separate text fields, so changing
    them means calling generate_poster() again rather than editing a string.
    Caption edits don't come through here at all - they're a plain string swap
    the dashboard makes directly, no regeneration needed.

    `photo_filename` must be a bare filename that exists in the demo photo pool
    (assets/listings/unit_photos/) - it never touches arbitrary paths on disk.
    Returns `{"poster_base64": "<base64 PNG bytes>"}`.
    """
    photo_filename = payload.get("photo_filename")
    if not isinstance(photo_filename, str) or not photo_filename.strip():
        raise HTTPException(status_code=400, detail="'photo_filename' is required.")

    # Bare filename only - no path separators, so this can't escape
    # PHOTO_POOL_DIR regardless of what a client sends.
    if photo_filename != Path(photo_filename).name:
        raise HTTPException(status_code=400, detail="'photo_filename' is invalid.")

    photo_path = PHOTO_POOL_DIR / photo_filename
    try:
        photo_path.resolve().relative_to(PHOTO_POOL_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="'photo_filename' is invalid.")

    if not photo_path.is_file():
        raise HTTPException(status_code=404, detail="That demo photo no longer exists.")

    price = _require_text(payload.get("price"), "price")
    location = _require_text(payload.get("location"), "location")

    features = payload.get("features")
    if features is not None and not isinstance(features, str):
        raise HTTPException(status_code=400, detail="'features' must be a string.")

    style_tag = payload.get("style_tag")
    if style_tag is not None and not isinstance(style_tag, str):
        raise HTTPException(status_code=400, detail="'style_tag' must be a string.")

    usable_paths = _usable_photos([str(photo_path)])

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    job_dir = Path(tempfile.mkdtemp(prefix="regen_", dir=UPLOAD_ROOT))
    try:
        output_path = job_dir / f"poster_{uuid.uuid4().hex[:8]}.png"
        try:
            generate_poster(
                photos=usable_paths,
                price=price,
                location=location,
                bedrooms=0,
                bathrooms=0,
                output_path=str(output_path),
                style_tag=style_tag,
                details=features or "",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        poster_base64 = base64.b64encode(output_path.read_bytes()).decode("ascii")
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

    return {"poster_base64": poster_base64}


def _resolve_pool_photo(raw: object) -> Path:
    """
    Bare-filename + no-path-traversal + must-exist validation for a demo pool
    photo. Identical rules to /regenerate-poster's inline checks above -
    deliberately duplicated rather than refactored into that endpoint, so
    the already-working /regenerate-poster keeps a zero-line diff. Shared by
    /toggle-photo-condition, GET /pool-photo/{filename}, and /publish-post's
    optional photo_filename field.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise HTTPException(status_code=400, detail="'photo_filename' is required.")

    photo_filename = raw.strip()
    if photo_filename != Path(photo_filename).name:
        raise HTTPException(status_code=400, detail="'photo_filename' is invalid.")

    photo_path = PHOTO_POOL_DIR / photo_filename
    try:
        photo_path.resolve().relative_to(PHOTO_POOL_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="'photo_filename' is invalid.")

    if not photo_path.is_file():
        raise HTTPException(status_code=404, detail="That demo photo no longer exists.")

    return photo_path


@app.get("/pool-photo/{photo_filename}")
def pool_photo_endpoint(photo_filename: str):
    """
    Serve an ORIGINAL, untouched pool photo as a browser-renderable JPEG.

    No AI cost - this never calls fal.ai. Exists so the landing page's photo
    widget has something to show before any lighting/weather combo is
    toggled. Re-encodes via photo_as_jpeg_bytes so a .HEIC source photo (4 of
    the 10 in the pool) still renders in a plain <img> tag.
    """
    photo_path = _resolve_pool_photo(photo_filename)
    jpeg_bytes = photo_as_jpeg_bytes(str(photo_path))
    if jpeg_bytes is None:
        raise HTTPException(status_code=500, detail="That photo could not be read as an image.")
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.get("/listing-photo-widget.js")
def listing_photo_widget_js():
    """
    Serve the isolated landing-page photo-toggle widget script.

    A dedicated route rather than a generic StaticFiles mount, matching the
    existing pattern here: /dashboard, /listing and /privacy are each their
    own FileResponse route reading from STATIC_DIR - there is no
    app.mount("/static", ...) anywhere in this app, so a plain
    `<script src="/static/...">` would 404.
    """
    page = STATIC_DIR / "listing-photo-widget.js"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Widget script not found.")
    return FileResponse(page, media_type="application/javascript")


# How many (time_of_day, weather) combinations exist per photo. Matches
# len(TIME_OF_DAY_VALUES) * len(WEATHER_VALUES) - kept as its own constant so
# the cap check below reads as a business rule, not an incidental product of
# two tuple lengths.
MAX_CONDITION_COMBOS_PER_PHOTO = 9


@app.post("/toggle-photo-condition")
# `def`, not `async def` - same reasoning as every other endpoint here that
# calls a blocking third-party API (Airtable, fal.ai) and does blocking
# Pillow work, with no `await` anywhere in the body.
def toggle_photo_condition_endpoint(payload: dict = Body(...)):
    """
    Re-light a demo photo for a given time-of-day + weather combo, then
    redraw the poster's price/location/features overlay on top of it.

    Send JSON:

        {
          "photo_filename": "unit_012.jpg",
          "time_of_day": "morning" | "evening" | "night",
          "weather": "sunny" | "cloudy" | "rainy",
          "price": "RM 1,200",
          "location": "Mont Kiara, Kuala Lumpur",
          "features": "Master, Queen bed, Private bathroom",  # optional
          "style_tag": "Warm"                                 # optional
        }

    CACHE FIRST, ALWAYS: every combo is looked up in the Airtable "Photo
    Condition Cache" table before fal.ai is ever called. fal.ai is only
    called on a genuine cache miss, always starting from the ORIGINAL pool
    photo (never a previously-edited one), and the result is written back to
    the cache immediately, before this function returns - a combo is
    generated at most once, ever. This is what keeps the fal.ai budget
    bounded at photo_pool_size * 9 generations worst case, not per-request.

    The AI edit only ever touches the photo itself. The fixed-text price/
    location/features overlay is always drawn by the existing
    generate_poster() -> _draw_text_block() path, exactly as every other
    poster-producing endpoint does it, and is redrawn fresh on every call
    (cache hit or miss) - so an edited price still shows correctly on a
    photo that was cached earlier with a different price. A visible
    disclosure label is burned into the final image afterwards.

    Returns `{"poster_base64": "<base64 PNG bytes>", "cached": true|false}`.
    """
    photo_path = _resolve_pool_photo(payload.get("photo_filename"))
    photo_filename = photo_path.name

    time_of_day = payload.get("time_of_day")
    if time_of_day not in TIME_OF_DAY_VALUES:
        raise HTTPException(
            status_code=400,
            detail=f"'time_of_day' must be one of {list(TIME_OF_DAY_VALUES)}.",
        )

    weather = payload.get("weather")
    if weather not in WEATHER_VALUES:
        raise HTTPException(
            status_code=400,
            detail=f"'weather' must be one of {list(WEATHER_VALUES)}.",
        )

    price = _require_text(payload.get("price"), "price")
    location = _require_text(payload.get("location"), "location")

    features = payload.get("features")
    if features is not None and not isinstance(features, str):
        raise HTTPException(status_code=400, detail="'features' must be a string.")

    style_tag = payload.get("style_tag")
    if style_tag is not None and not isinstance(style_tag, str):
        raise HTTPException(status_code=400, detail="'style_tag' must be a string.")

    # Cache lookup ALWAYS comes first. If Airtable can't even be read here,
    # we must not fall through and call fal.ai blind - that would defeat the
    # entire point of the cache, which is what keeps spend bounded.
    try:
        entries = get_photo_condition_cache_entries(photo_filename)
    except AirtableError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not check the photo cache before generating: {exc}",
        )

    match = next(
        (
            entry
            for entry in entries
            if entry["time_of_day"] == time_of_day
            and entry["weather"] == weather
            and entry["attachment_url"]
        ),
        None,
    )

    if match is not None:
        try:
            image_response = httpx.get(match["attachment_url"], timeout=30)
            image_response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Could not download the cached photo from Airtable: {exc}",
            )
        image_bytes = image_response.content
        cached = True
    else:
        # Defensive only: with time_of_day/weather constrained to exactly 9
        # valid combinations, normal traffic can never reach a genuine 9th
        # miss (the 9th combo's lookup above would already have been a hit).
        # Counts every existing row, including an attachment-less orphan
        # (see get_photo_condition_cache_entries), so the worst-case-spend
        # guarantee holds even after a rare crash-mid-write.
        if len(entries) >= MAX_CONDITION_COMBOS_PER_PHOTO:
            raise HTTPException(
                status_code=409,
                detail="All 9 lighting/weather combinations already exist for this photo.",
            )

        _usable_photos([str(photo_path)])

        try:
            image_bytes = edit_photo_condition(str(photo_path), time_of_day, weather)
        except FalEditError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

        try:
            save_cached_photo_condition(photo_filename, time_of_day, weather, image_bytes)
        except AirtableError as exc:
            # The edit already happened - real money was spent. The caller
            # should still get what they paid for; a failed cache write just
            # means this exact combo risks being re-generated (and re-billed)
            # on a future request, which is why this is logged loudly rather
            # than swallowed quietly.
            logger.error(
                "Generated a photo-condition edit for %r (%s/%s) but could not "
                "cache it in Airtable - this combo may be re-billed on a future "
                "request: %s",
                photo_filename,
                time_of_day,
                weather,
                exc,
            )
        cached = False

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    job_dir = Path(tempfile.mkdtemp(prefix="toggle_", dir=UPLOAD_ROOT))
    try:
        edited_photo_path = job_dir / "edited_source.jpg"
        edited_photo_path.write_bytes(image_bytes)

        output_path = job_dir / f"poster_{uuid.uuid4().hex[:8]}.png"
        try:
            generate_poster(
                photos=[str(edited_photo_path)],
                price=price,
                location=location,
                bedrooms=0,
                bathrooms=0,
                output_path=str(output_path),
                style_tag=style_tag,
                details=features or "",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        add_disclosure_label(str(output_path))

        poster_base64 = base64.b64encode(output_path.read_bytes()).decode("ascii")
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

    return {"poster_base64": poster_base64, "cached": cached}


def _build_fb_message(caption: str | None, hashtags: list | None, cta: str | None) -> str:
    """
    Assemble the final Facebook post text: caption, hashtags, then CTA.

    Blocks are separated by a blank line. Each hashtag is prefixed with '#'
    (any stray leading '#' the caller already added is stripped first so we
    never end up with '##tag'). Empty blocks are dropped so the result never
    starts or ends with dangling blank lines.
    """
    parts: list[str] = []

    if isinstance(caption, str) and caption.strip():
        parts.append(caption.strip())

    tags: list[str] = []
    for tag in hashtags or []:
        cleaned = str(tag).lstrip("#").strip()
        if cleaned:
            tags.append("#" + cleaned)
    if tags:
        parts.append(" ".join(tags))

    if isinstance(cta, str) and cta.strip():
        parts.append(cta.strip())

    # A blank line between blocks == two newlines.
    return "\n\n".join(parts)


def _decode_poster(poster_base64: str) -> bytes:
    """
    Turn the base64 poster string from the client back into PNG bytes.

    Tolerates an accidental `data:image/png;base64,` prefix, and raises a clean
    400 (rather than a stack trace) if the value is missing or not valid base64.
    """
    if not isinstance(poster_base64, str) or not poster_base64.strip():
        raise HTTPException(status_code=400, detail="'poster_base64' is required.")

    encoded = poster_base64.strip()
    # Strip a data URL prefix if one slipped in (the API returns raw base64, but
    # be forgiving in case a caller re-uses an <img> src verbatim).
    if encoded.startswith("data:"):
        comma = encoded.find(",")
        if comma != -1:
            encoded = encoded[comma + 1 :]

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="'poster_base64' is not valid base64.")

    if not image_bytes:
        raise HTTPException(
            status_code=400, detail="'poster_base64' decoded to an empty image."
        )
    return image_bytes


@app.post("/publish-post")
# `def` for the same reason as /create-post: the Facebook upload and the
# Airtable write are both blocking calls and there is no `await` here, so as an
# async endpoint a slow Graph API response would stall every other request on
# the server. Matches /sync-engagement, which was already sync.
def publish_post_endpoint(payload: dict = Body(...)):
    """
    Publish an already-generated poster + caption to the Facebook Page.

    Send JSON:

        {
          "poster_base64": "<base64 PNG bytes>",   # as returned by /create-post
          "caption":       "<post text>",
          "hashtags":      ["luxuryliving", ...],  # no '#' prefix needed
          "cta":           "DM us to book a viewing",
          "style_tag":     "Warm",                 # optional - from /create-post
          "template_id":   "A",                    # optional - from /create-post
          "listing_row_index": 17,                  # optional - from /auto-create-post
          "photo_filename": "unit_012.jpg"          # optional - the pool photo actually shown
        }

    Before publishing we mint a short tracking id and append its link
    (``<BASE_URL>/t/<id>``) to the CTA, so clicks route through the tracking
    endpoint. The caption, hashtags (each '#'-prefixed) and the link-carrying CTA
    are combined - separated by blank lines - into the final post text.

    `photo_filename`, when present, is recorded locally against the new
    tracking id (see app.click_tracker.save_tracking_photo) so the public
    landing page (GET /listing-info) can show and toggle the SAME pool photo
    this post was actually published with, rather than none at all.

    After a successful publish we record the post in Airtable (template, style,
    caption, Facebook post id, tracking id). That logging is best-effort: if it
    fails, the post has still gone live, so we report success with a note rather
    than failing the request. Returns JSON:

        {
          "post_id": "...", "post_url": "https://www.facebook.com/...",
          "tracking_id": "ab12cd34", "tracking_url": "https://host/t/ab12cd34",
          "airtable_logged": true, "airtable_record_id": "rec..."
        }

    On a publishing failure the endpoint returns a clean 502 carrying Facebook's
    own error message.
    """
    hashtags = payload.get("hashtags")
    if hashtags is not None and not isinstance(hashtags, list):
        raise HTTPException(status_code=400, detail="'hashtags' must be a list.")

    caption = payload.get("caption")
    # Optional metadata threaded through from /create-post so we can learn from it
    # later. Absent/None is fine - create_post_record simply skips empty fields.
    style_tag = payload.get("style_tag")
    template_id = payload.get("template_id")

    # Threaded through from /auto-create-post (absent for the manual-form
    # flow). This is what makes "next un-posted listing" durable in Airtable
    # across redeploys - see app.listings_source and get_posts().
    raw_row_index = payload.get("listing_row_index")
    listing_row_index: int | None = None
    if raw_row_index is not None:
        try:
            listing_row_index = int(raw_row_index)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400, detail="'listing_row_index' must be an integer."
            )

    # Optional - the pool photo this poster was actually built from. Validated
    # with the same bare-filename/no-path-traversal rules as every other
    # endpoint that touches PHOTO_POOL_DIR; re-validating here (rather than
    # trusting the value the front end already had) costs nothing and means
    # a stale/deleted filename fails loudly instead of silently recording
    # bad data for the landing page to later 404 on.
    raw_photo_filename = payload.get("photo_filename")
    photo_filename: str | None = None
    if raw_photo_filename is not None:
        photo_filename = _resolve_pool_photo(raw_photo_filename).name

    image_bytes = _decode_poster(payload.get("poster_base64"))

    # Mint the tracking link and append it to the CTA. The link rides on the CTA
    # only; the caption we store in Airtable stays clean (no link appended).
    tracking_id = _generate_tracking_id()
    tracking_url = _tracking_url(tracking_id)
    raw_cta = payload.get("cta")
    if isinstance(raw_cta, str) and raw_cta.strip():
        cta_with_link = f"{raw_cta.strip()} {tracking_url}"
    else:
        cta_with_link = tracking_url

    message = _build_fb_message(caption, hashtags, cta_with_link)

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    job_dir = Path(tempfile.mkdtemp(prefix="publish_", dir=UPLOAD_ROOT))

    # Decode to a real file on disk (the publisher does a multipart file upload),
    # then remove the job folder synchronously once we have the API response.
    try:
        image_path = job_dir / "poster.png"
        image_path.write_bytes(image_bytes)

        try:
            result = publish_post(str(image_path), message)
        except FacebookPublishError as exc:
            # Same shape as /create-post's caption failure: a clean 502 with the
            # underlying (Facebook) message rather than a stack trace.
            raise HTTPException(status_code=502, detail=str(exc))
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

    # Record the facebook_post_id -> tracking_id link locally, right after the
    # publish succeeds. This is what lets /sync-engagement attribute clicks to
    # this post by its Facebook id, independent of whether Airtable stores the
    # tracking id (the Posts table may have no "Tracking ID" column). Do this
    # before the Airtable write so the link survives even if that write fails.
    link_post(result["post_id"], tracking_id)

    # Also remember which listing this tracking id belongs to (auto-ready flow
    # only - the manual form has no row index). This is what lets /listing-info
    # and /chat tell a lead who clicks the link WHICH property they're looking
    # at, instead of falling back to generic copy.
    if listing_row_index is not None:
        save_tracking_listing(tracking_id, listing_row_index)

    # Same idea, for the photo: lets the public landing page show/toggle the
    # exact photo this post used instead of nothing at all.
    if photo_filename is not None:
        save_tracking_photo(tracking_id, photo_filename)

    response = {
        "post_id": result["post_id"],
        "post_url": result["post_url"],
        "tracking_id": tracking_id,
        "tracking_url": tracking_url,
    }

    # The Facebook post already succeeded - that's the important part. Recording
    # it in Airtable is best-effort: on failure we log it and flag it in the
    # response, but still return success so the user isn't told the post failed
    # when it didn't.
    try:
        record_id = create_post_record(
            template_id=template_id,
            style_tag=style_tag,
            caption=caption if isinstance(caption, str) else "",
            facebook_post_id=result["post_id"],
            tracking_id=tracking_id,
            listing_row_index=listing_row_index,
        )
        response["airtable_logged"] = True
        response["airtable_record_id"] = record_id
    except AirtableError as exc:
        logger.warning(
            "Airtable logging failed after publishing post %s: %s",
            result["post_id"],
            exc,
        )
        response["airtable_logged"] = False
        response["note"] = f"Post published to Facebook, but Airtable logging failed: {exc}"

    return response


def _build_transcript(history: list, message: str, reply: str) -> str:
    """
    Flatten the conversation into the plain text we store in Airtable.

    One labelled line per turn ("Lead:" / "Assistant:"), oldest first, ending
    with the exchange we just had. History arrives straight off a JSON body, so
    anything that isn't a {role, content} pair of strings is skipped rather
    than trusted - same defensiveness as lead_chatbot._clean_history.
    """
    lines: list[str] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant"):
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        speaker = "Lead" if role == "user" else "Assistant"
        lines.append(f"{speaker}: {content.strip()}")

    lines.append(f"Lead: {message}")
    lines.append(f"Assistant: {reply}")
    return "\n".join(lines)


# How many alternatives to put in front of the assistant. It is told to name at
# most a couple; a few spares let it pick the ones that actually fit what they
# asked, without turning the reply into a list.
MAX_ALTERNATIVES = 4


def _alternatives_for(tracking_id: str, history: list, message: str) -> dict | None:
    """
    Real listings to offer this lead, or None if they aren't asking for any.

    Two steps: a cheap Claude call reads the whole conversation for "got
    anything else?" plus every preference they have accumulated - budget, area,
    room, bed, bathroom, must-have features - then search_listings ranks the
    CSV against all of it. The property they are already looking at is
    excluded, so it can never come back as its own alternative.

    Returns None when they aren't asking - including when the intent call
    failed and degraded to its safe default - which leaves the chat turn
    exactly as it behaved before. Otherwise a dict::

        {"listings": [context, ...], "exact": bool, "relaxed_criteria": [...]}

    An empty "listings" is different from None and deliberate: it means they
    DID ask and nothing matched, and the assistant says so rather than
    inventing something. "exact" carries search_listings' own verdict through
    to the assistant untouched - False means these are near misses and it has
    to say so.
    """
    intent = extract_search_intent(history, message)
    if not intent["wants_alternatives"]:
        return None

    result = search_listings(
        max_price=intent["max_price"],
        target_price=intent["target_price"],
        location_keyword=intent["location_keyword"],
        room_type_keyword=intent["room_type_keyword"],
        bed_type_keyword=intent["bed_type_keyword"],
        bathroom_type_keyword=intent["bathroom_type_keyword"],
        must_have_features=intent["must_have_features"],
        # Same durable resolution as _listing_context, so the property they
        # are already looking at is excluded even after a restart wiped the
        # local map.
        exclude_row_index=_row_index_for_tracking(tracking_id),
        limit=MAX_ALTERNATIVES,
    )
    return {
        "listings": [_listing_to_context(listing) for listing in result],
        "exact": result.is_exact,
        "relaxed_criteria": list(result.relaxed_criteria),
    }


@app.post("/chat")
# `def`, not `async def` - same reasoning as /create-post: the Claude call and
# the Airtable read/write are blocking and there is no `await` here, so as an
# async endpoint one slow chat turn would stall every other request.
def chat_endpoint(payload: dict = Body(...)):
    """
    Answer a lead's chat message on the listing page and record the lead.

    Send JSON:

        {
          "tracking_id": "ab12cd34",       # from the ?tid= on the landing page
          "message":     "Is parking included?",
          "history":     [{"role": "user", "content": "..."}, ...],  # optional
          "followup_sent": false           # optional - see followup_email below
        }

    We resolve which listing the tracking id was published for (same lookup as
    /listing-info) and hand that context to the assistant so it can talk about
    the actual property; an unknown id just means it stays general.

    If the conversation reads like they want to see what ELSE is available (or
    are following up on something we already offered), we also search
    room_listings.csv for real alternatives - ranked against every preference
    they have given across the whole chat, minus the property they're already
    on - and pass those in too, so the assistant offers listings that genuinely
    exist instead of improvising. When nothing matches all of it the search
    loosens the softest criteria and says so, and that "these are near misses"
    verdict is passed to the assistant rather than dropped. See
    _alternatives_for.

    Alongside the reply, the assistant reports a qualification `status` and
    what it has picked up about the lead's budget and move-in timeline. The
    signals are written to the Airtable Leads table - one row per tracking id,
    updated as the conversation goes - but deliberately NOT returned here: the
    lead should never see that they're being scored. Airtable logging is
    best-effort; if it fails we log a warning and still answer.

    Returns JSON::

        {"reply": "<what to say back>", "status": "lead" | "prospect",
         "followup_email": "<draft email>" | null}

    `followup_email` is the assistant's one concrete next step: on the single
    turn a lead first qualifies as a prospect it is a short draft email written
    in their voice, to the agent, from what they actually said - the listing,
    their budget, their timing - which the page offers them to copy and send.
    It is null on every other turn, so it appears once rather than trailing
    every later message. Nothing is sent from here: this is text for the lead's
    own mail client, not an email the server delivers.

    That "once" is what the request's optional `followup_sent` is for. This
    endpoint holds no conversation state - the client owns the history - so it
    cannot know a draft already went out three messages ago; the page says so
    by sending back `followup_sent: true` from then on. Omitting it is safe
    but softer: the assistant is also told to write only one, and mostly does.

    A failure from the assistant itself comes back as a clean 502 carrying the
    underlying message, the same shape as /create-post's caption failures.
    """
    raw_message = payload.get("message")
    message = _require_text(raw_message if isinstance(raw_message, str) else None, "message")

    tracking_id = payload.get("tracking_id")
    tracking_id = tracking_id.strip() if isinstance(tracking_id, str) else ""

    history = payload.get("history")
    if history is None:
        history = []
    if not isinstance(history, list):
        raise HTTPException(status_code=400, detail="'history' must be a list.")

    # Anything other than a real `true` means "no draft yet" - a client that
    # omits it, or sends something odd, gets the default behaviour rather than
    # a 400. It only ever suppresses a draft, so a wrong value costs the lead
    # a suggestion, never anything worse.
    followup_sent = payload.get("followup_sent") is True

    listing_context = _listing_context(tracking_id)
    alternatives = _alternatives_for(tracking_id, history, message) or {}

    try:
        result = qualify_lead(
            history,
            message,
            listing_context,
            other_listings=alternatives.get("listings"),
            # Default True so a turn with no search at all is described
            # accurately: nothing was loosened because nothing was searched.
            alternatives_exact=alternatives.get("exact", True),
            relaxed_criteria=alternatives.get("relaxed_criteria"),
            followup_sent=followup_sent,
        )
    except ChatError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    # Recording the lead is bookkeeping - the reply is what the person on the
    # other end is waiting for, so an Airtable problem must never cost them it.
    try:
        upsert_lead_record(
            tracking_id=tracking_id,
            interested_listing=(
                listing_context["condo_name"] if listing_context else "Unknown"
            ),
            status=result["status"],
            budget_signal=result["budget_signal"],
            timeline_signal=result["timeline_signal"],
            transcript=_build_transcript(history, message, result["reply"]),
        )
    except AirtableError as exc:
        logger.warning(
            "Airtable lead logging failed for tracking_id %r: %s",
            tracking_id or None,
            exc,
        )

    return {
        "reply": result["reply"],
        "status": result["status"],
        "followup_email": result["followup_email"],
    }


@app.post("/sync-engagement")
def sync_engagement_endpoint(x_sync_secret: str | None = Header(default=None)):
    """
    Pull fresh engagement for every logged post and record it in Airtable.

    Intended to be called periodically (e.g. by Make.com on a schedule) or by
    hand for testing. For each Post row in Airtable that has a facebook_post_id:

      1. Read the click count recorded locally for that post's tracking id.
      2. Ask the Facebook Graph API for the post's current like and comment
         counts (likes.summary(true) / comments.summary(true)).
      3. Write a new Engagement row (clicks + likes + comments), linked to the
         post, via airtable_client.log_engagement.

    CLICKS ARE A RATCHET. The local counter lives on an ephemeral disk and
    empties on restart, so step 1 can legitimately read LOWER than what Airtable
    already has. Writing that blindly buried real history, because every reader
    treats the newest Engagement row as current truth. A post is therefore never
    written below its latest recorded click count - see the ratchet comment in
    the loop below. Likes and comments are not ratcheted: those come from
    Facebook, which is authoritative and where a decrease is real (an unlike, a
    deleted comment).

    Failures are isolated per-post so a single bad post can't abort the run:
      * A Facebook read failure does NOT fail the post - clicks are independent
        of Facebook, so we still log the clicks with likes/comments as 0 and mark
        the detail with note "facebook_read_failed". The post counts as synced
        (partial data).
      * An Airtable write failure does fail the post (Airtable is the durable
        store, so nothing was recorded).
    Zero posts is not an error - it returns an empty summary.

    Returns JSON::

        {
          "posts_checked": 3, "synced": 2, "skipped": 1, "failed": 0,
          "details": [ {"facebook_post_id": "...", "clicks": 4, "likes": 10,
                        "comments": 2, "logged": true},
                       {"facebook_post_id": "...", "clicks": 26, "likes": 3,
                        "comments": 0, "logged": true,
                        "clicks_local": 0, "clicks_preserved": true},
                       {"facebook_post_id": "...", "clicks": 3, "likes": 0,
                        "comments": 0, "logged": true,
                        "note": "facebook_read_failed"}, ... ]
        }

    ``clicks_preserved`` marks a post whose local counter had reset: ``clicks``
    is the carried-forward recorded value and ``clicks_local`` is what the
    counter actually said. A top-level ``clicks_baseline_unavailable`` means the
    recorded counts couldn't be read at all, so no regression could be prevented
    on that run.

    Requires the ``X-Sync-Secret`` header to match the ``SYNC_ENGAGEMENT_SECRET``
    env var - this endpoint has no other auth and would otherwise let anyone
    with the URL trigger Facebook API calls and duplicate Airtable rows.
    """
    expected_secret = os.environ.get("SYNC_ENGAGEMENT_SECRET")
    if not expected_secret:
        # Fail closed: an unconfigured secret must not mean "no auth required".
        raise HTTPException(status_code=503, detail="SYNC_ENGAGEMENT_SECRET is not configured")
    if not x_sync_secret or not secrets.compare_digest(x_sync_secret, expected_secret):
        raise HTTPException(status_code=401, detail="Missing or invalid X-Sync-Secret header")

    try:
        posts = get_posts()
    except AirtableError as exc:
        # Can't read Airtable at all (missing creds / API error) - a clean 502.
        raise HTTPException(status_code=502, detail=str(exc))

    summary: dict = {
        "posts_checked": len(posts),
        "synced": 0,
        "skipped": 0,
        "failed": 0,
        "details": [],
    }

    if not posts:
        summary["message"] = "No posts found in Airtable yet - nothing to sync."
        return summary

    # The click count already recorded for each post, read ONCE for the whole
    # run rather than per post. This is the ratchet's baseline - see where
    # `clicks` is decided below for why a sync must never write beneath it.
    #
    # Best-effort: if this read fails we can't tell what was already recorded,
    # so we fall back to the raw local counts (the pre-ratchet behaviour) rather
    # than refusing to sync at all. That window is flagged on the response so a
    # regression that slips through it isn't silent.
    try:
        recorded_clicks = latest_clicks_by_post(
            {post["record_id"] for post in posts if post.get("record_id")}
        )
    except AirtableError as exc:
        recorded_clicks = {}
        summary["clicks_baseline_unavailable"] = str(exc)
        logger.warning(
            "sync: could not read recorded click counts, click regressions "
            "cannot be prevented this run: %s",
            exc,
        )

    for post in posts:
        record_id = post.get("record_id")
        fb_id = post.get("facebook_post_id")

        # A post with no Facebook id can't have its engagement fetched - skip it.
        if not fb_id:
            summary["skipped"] += 1
            summary["details"].append(
                {"record_id": record_id, "skipped": True, "reason": "no facebook_post_id"}
            )
            continue

        # Resolve the tracking id for this post. Prefer the value stored on the
        # Airtable record, but fall back to the local facebook_post_id ->
        # tracking_id map recorded at publish time - the Posts table may have no
        # "Tracking ID" column, in which case create_post_record couldn't store
        # it and post["tracking_id"] comes back None. Without this fallback,
        # clicks would silently read as 0 for every post.
        tracking_id = post.get("tracking_id") or tracking_id_for_post(str(fb_id)) or ""

        # Clicks come from our own tracking store, so we always have them -
        # they're completely independent of Facebook's API. The Facebook
        # like/comment read is best-effort: if it fails we still log the post
        # with the clicks we have and likes/comments as 0, rather than losing the
        # whole post to a Facebook API hiccup. The post still counts as synced,
        # just with partial (clicks-only) data flagged in the detail.
        local_clicks = get_clicks(tracking_id)

        # RATCHET: clicks only ever go up.
        #
        # app.click_tracker is a local file on an ephemeral disk - a restart or
        # redeploy empties it. Writing its count blindly meant a post that had
        # genuinely earned 26 clicks got a fresh Engagement row saying 0 the next
        # time sync ran after a restart, and since every reader takes the LATEST
        # row as current truth, that permanently buried the real number.
        #
        # A counter forgetting is not the same as clicks un-happening, so a
        # lower local count is treated as "no new information" and the recorded
        # value is carried forward instead. Genuine growth is unaffected: a
        # higher local count is written exactly as before. A post with no
        # Engagement rows yet has no baseline and writes its local count as-is
        # (`.get` default of 0 can't exceed a non-negative count anyway).
        recorded = recorded_clicks.get(record_id, 0)
        clicks = max(local_clicks, recorded)
        clicks_preserved = clicks != local_clicks

        logger.info(
            "sync: post fb=%s tracking_id=%r -> clicks=%d (local=%d, recorded=%d)%s",
            fb_id,
            tracking_id or None,
            clicks,
            local_clicks,
            recorded,
            " [local counter reset - carrying recorded value forward]"
            if clicks_preserved
            else "",
        )
        facebook_read_failed = False

        try:
            engagement = get_post_engagement(str(fb_id))
            likes = engagement["likes"]
            comments = engagement["comments"]
        except FacebookPublishError as exc:
            likes = 0
            comments = 0
            facebook_read_failed = True
            logger.warning("Facebook engagement read failed for %s: %s", fb_id, exc)

        try:
            log_engagement(record_id, clicks, likes, comments)
        except AirtableError as exc:
            # Airtable is the durable store - if we can't write at all, this post
            # genuinely failed to sync.
            summary["failed"] += 1
            summary["details"].append(
                {
                    "facebook_post_id": fb_id,
                    "clicks": clicks,
                    "likes": likes,
                    "comments": comments,
                    "error": f"Airtable write failed: {exc}",
                }
            )
            continue

        summary["synced"] += 1
        detail = {
            "facebook_post_id": fb_id,
            "clicks": clicks,
            "likes": likes,
            "comments": comments,
            "logged": True,
        }
        if clicks_preserved:
            # Make the ratchet visible: without these the response would just
            # show an unchanged click count with no hint that the local counter
            # had reset and the recorded value was held instead.
            detail["clicks_local"] = local_clicks
            detail["clicks_preserved"] = True
        if facebook_read_failed:
            # Partial sync: clicks were logged, Facebook counts defaulted to 0.
            detail["note"] = "facebook_read_failed"
        summary["details"].append(detail)

    return summary


# ---------------------------------------------------------------------------
# Internal dashboard - read-only, shared-secret gated
# ---------------------------------------------------------------------------
#
# Same auth shape as /sync-engagement (a shared secret from the environment,
# compared in constant time, failing closed when unset), widened just enough to
# be usable from a browser: the secret may arrive as the X-Internal-Secret
# header (curl, tests), as ?secret= (the login form), or as the cookie the
# first successful load sets so refreshes work without the query string.
#
# READ-ONLY: both handlers only call collect_dashboard_data(), which reads
# Airtable and never writes - see app/internal_dashboard.py.

# Name of the cookie the login flow sets so a human isn't retyping the secret
# on every page load. Scoped to the dashboard's own path so it is never sent to
# the public routes.
INTERNAL_DASHBOARD_COOKIE = "internal_dashboard_secret"

# How long that cookie survives. Short by design - it holds the shared secret.
INTERNAL_DASHBOARD_COOKIE_MAX_AGE = 12 * 60 * 60  # 12 hours


def _internal_secret_ok(*candidates: str | None) -> bool:
    """
    True if any of ``candidates`` is the configured internal dashboard secret.

    The secret lives in the INTERNAL_DASHBOARD_SECRET env var (set on Railway;
    never in code). Compared with ``compare_digest`` so a wrong guess can't be
    narrowed down by timing.

    Raises:
        HTTPException(503): if the env var isn't set. Fails closed - an
            unconfigured secret must never read as "no auth required", exactly
            as in /sync-engagement.
    """
    expected = os.environ.get("INTERNAL_DASHBOARD_SECRET")
    if not expected:
        raise HTTPException(
            status_code=503, detail="INTERNAL_DASHBOARD_SECRET is not configured"
        )

    for candidate in candidates:
        # .encode() because compare_digest rejects str containing non-ASCII.
        if candidate and secrets.compare_digest(candidate.encode(), expected.encode()):
            return True
    return False


def _set_internal_cookie(response, secret: str) -> None:
    """Remember a valid secret for this browser, scoped to the dashboard path."""
    response.set_cookie(
        INTERNAL_DASHBOARD_COOKIE,
        secret,
        max_age=INTERNAL_DASHBOARD_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        path="/internal-dashboard",
        # Only mark it Secure when this server is actually served over TLS,
        # otherwise the cookie would be silently dropped in local http testing.
        secure=_base_url().startswith("https://"),
    )


@app.get("/internal-dashboard")
def internal_dashboard_page(
    x_internal_secret: str | None = Header(default=None),
    secret: str | None = Query(default=None),
    internal_dashboard_secret: str | None = Cookie(default=None),
):
    """
    The internal sales-flow dashboard page (HTML shell only - no data).

    Wrong or missing secret returns 401 with a small login form and nothing
    else: the page carries no records of its own, so a rejected request leaks
    neither post, engagement nor lead data. The form re-requests this URL with
    ``?secret=``; on success we set the dashboard cookie and the page strips the
    query string back out of the address bar.
    """
    if not _internal_secret_ok(x_internal_secret, secret, internal_dashboard_secret):
        login = STATIC_DIR / "internal_dashboard_login.html"
        if not login.exists():
            raise HTTPException(status_code=401, detail="Invalid internal dashboard secret")
        return FileResponse(login, media_type="text/html", status_code=401)

    page = STATIC_DIR / "internal_dashboard.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Internal dashboard page not found.")

    response = FileResponse(page, media_type="text/html")
    # Whichever way the caller proved themselves, hand the browser a cookie so
    # the data fetch and any refresh work without the secret in the URL.
    _set_internal_cookie(
        response, x_internal_secret or secret or internal_dashboard_secret or ""
    )
    return response


@app.get("/internal-dashboard/data")
def internal_dashboard_data(
    x_internal_secret: str | None = Header(default=None),
    secret: str | None = Query(default=None),
    internal_dashboard_secret: str | None = Cookie(default=None),
):
    """
    The dashboard's read-only snapshot as JSON: posts, engagement, styles, leads.

    Same gate as the page, but a rejected request gets a plain 401 JSON error
    (no records) rather than a login form - this is only ever called by fetch().
    """
    if not _internal_secret_ok(x_internal_secret, secret, internal_dashboard_secret):
        raise HTTPException(
            status_code=401, detail="Missing or invalid X-Internal-Secret"
        )

    return collect_dashboard_data()


@app.get("/report")
def daily_report_page():
    """Serve the end-of-day report page (it fetches /daily-report for its data)."""
    page = STATIC_DIR / "daily_report.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Daily report page not found.")
    return FileResponse(page, media_type="text/html")


@app.get("/daily-report")
# Deliberately `def`, not `async def`, for the same reason as /create-post: the
# Airtable reads (sync httpx) and the Claude call (sync Anthropic client) are
# all blocking, so a plain def lets FastAPI run this on its threadpool instead
# of stalling the event loop for the several seconds the summary takes.
def daily_report():
    """
    The end-of-day self-learning report: assembled numbers + an AI summary.

    Returns::

        {
          "data": {...},            # collect_report_data() output - see there
          "summary": {              # Claude-written, structured for skimming
            "headline": "...",      # one-sentence TL;DR (or a fallback line)
            "highlights": ["...", ...]   # 2-4 single-sentence bullets
          },
          "summary_error": null,    # the ReportError message when it fell back
        }

    This endpoint never 500s over its dependencies: Airtable problems degrade
    inside collect_report_data (empty sections + an errors list), and a Claude
    failure degrades to a clear "summary unavailable" headline (with no
    highlights) so the raw numbers still come back.
    """
    data = collect_report_data()

    try:
        summary = generate_daily_report(data)
        summary_error = None
    except ReportError as exc:
        logger.warning("Daily report summary unavailable: %s", exc)
        summary = {
            "headline": (
                "The AI-written summary is unavailable right now - the numbers "
                "here are still live. Refresh in a moment to try again."
            ),
            "highlights": [],
        }
        summary_error = str(exc)

    return {"data": data, "summary": summary, "summary_error": summary_error}


# ---------------------------------------------------------------------------
# Public stats - the live numbers the landing page hero counts up
# ---------------------------------------------------------------------------
#
# UNAUTHENTICATED BY DESIGN, AND AGGREGATE-ONLY BY DESIGN.
#
# The landing page is public, so anything it fetches has to be public too. That
# makes the shape of this response a deliberate decision rather than a
# convenience: it returns three whole-system totals and nothing else. No
# per-post rows, no per-listing breakdown, no lead or engagement records, no
# tracking ids. It is emphatically NOT a public mirror of
# /internal-dashboard/data - that stays secret-gated, and nothing here reads,
# imports or needs INTERNAL_DASHBOARD_SECRET.
#
# It is also deliberately NOT /daily-report: that endpoint calls Claude to write
# prose, which costs money and takes seconds. A landing page hit must not do
# either, so this assembles the same underlying numbers and stops there.

# How long an assembled snapshot is reused before we read Airtable again. The
# landing page is the most-hit route in the app and these numbers move slowly
# (a post at a time), so a short cache keeps a burst of visitors from turning
# into a burst of Airtable reads. Small enough that the page still reads as live.
PUBLIC_STATS_TTL_SECONDS = 60

# (monotonic_deadline, payload) for the cached snapshot, or None before the
# first successful assembly. Module-level rather than functools.lru_cache
# because we need time-based expiry, not value-based memoisation.
_public_stats_cache: tuple[float, dict] | None = None


def _assemble_public_stats() -> dict:
    """
    The three public totals, read fresh.

    Every number here comes from Airtable, which is the point: these are the
    figures a visitor sees first, so they have to be the ones that survive a
    restart or a redeploy rather than whatever this particular process happens
    to have accumulated since it booted.

    Never raises: each read is guarded independently so one unavailable source
    degrades that single number to its zero/None rather than failing the whole
    response. ``airtable_configured`` lets the page tell "genuinely zero so far"
    apart from "the backend isn't wired up", so it can stay honest either way
    instead of advertising a confident 0.
    """
    configured = airtable_is_configured()

    posts_published = 0
    if configured:
        try:
            posts_published = len(get_posts())
        except AirtableError as exc:
            logger.warning("Public stats could not read Posts: %s", exc)

    # Airtable, NOT app.click_tracker. The local counts file is an ephemeral
    # staging buffer that /sync-engagement drains into Airtable - it resets on
    # every restart and redeploy. Reading it here would put a reset-to-zero
    # clicks figure next to a posts count that survived the restart, which is
    # the one combination that makes the hero look broken rather than quiet.
    # Same rollup the internal dashboard's clicks column uses, via the shared
    # group_engagement_by_post() rule, so the two pages can't disagree.
    clicks_tracked = 0
    if configured:
        try:
            clicks_tracked = total_clicks_recorded()
        except AirtableError as exc:
            logger.warning("Public stats could not read Engagement: %s", exc)

    # Same single source of truth as /create-post's caption hint and the daily
    # report, so the landing page can never claim a different winner than the
    # system is actually using. Returns None when there's no engagement yet.
    winning_style = best_performing_style()

    return {
        "posts_published": posts_published,
        "clicks_tracked": clicks_tracked,
        "winning_style": winning_style,
        "airtable_configured": configured,
    }


@app.get("/public-stats")
# Deliberately `def`, not `async def`, for the same reason as /create-post: the
# Airtable reads underneath are blocking sync httpx, so a plain def runs this on
# FastAPI's threadpool instead of stalling the event loop.
def public_stats():
    """
    Whole-system totals for the public landing page hero. No auth, no secrets.

    Returns::

        {
          "posts_published": 6,       # rows in the Airtable Posts table
          "clicks_tracked": 39,       # each post's latest synced click count, summed
          "winning_style": "Warm",    # or null when no engagement is synced yet
          "airtable_configured": true
        }

    ``clicks_tracked`` is the synced Airtable figure, not the local counter, so
    it can lag by up to one /sync-engagement run - a durable number that updates
    on sync beats a live one that resets to zero on restart.

    Aggregates only - see the module comment above for why that boundary is
    drawn where it is. Served from a short in-process cache
    (``PUBLIC_STATS_TTL_SECONDS``) and never 500s: an unreadable source degrades
    to 0 / null so the hero always has something honest to render.
    """
    global _public_stats_cache

    now = time.monotonic()
    if _public_stats_cache is not None and now < _public_stats_cache[0]:
        return _public_stats_cache[1]

    stats = _assemble_public_stats()
    _public_stats_cache = (now + PUBLIC_STATS_TTL_SECONDS, stats)
    return stats
