"""
PropVibe - FastAPI entrypoint.

Exposes the poster generator over HTTP so the front end (or a webhook, or a
plain curl) can post some property photos plus listing details and get a
ready-to-share 1080x1080 PNG straight back in the response body.

Endpoints:
  GET  /              - health check
  GET  /dashboard     - minimal approval/preview page (static HTML)
  GET  /privacy       - privacy policy page (Facebook App Live mode requirement)
  POST /generate-poster - returns the poster PNG directly
  POST /create-post   - returns poster (base64) + Claude-written caption as JSON
  POST /publish-post  - publishes a poster + caption to a Facebook Page
  GET  /listing-info/{tracking_id} - the listing behind a tracking link, as JSON
  POST /chat          - AI leasing assistant for a lead on the listing page
  POST /sync-engagement - refreshes engagement stats (requires X-Sync-Secret header)
  GET  /internal-dashboard - internal read-only sales-flow view (secret-gated)
"""

import base64
import binascii
import logging
import os
import random
import secrets
import shutil
import tempfile
import uuid
from pathlib import Path
from urllib.parse import quote

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
from fastapi.responses import FileResponse, RedirectResponse
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
from app.poster_generator import GRID_TEMPLATE_MIN_PHOTOS, check_photos, generate_poster
from app.copy_generator import STYLE_TAGS, CaptionError, generate_caption
from app.lead_chatbot import ChatError, extract_search_intent, qualify_lead
from app.trend_research import research_trend
from app.listings_source import (
    PHOTO_POOL_DIR,
    Listing,
    get_listing,
    next_unposted_listing,
    random_pool_photo,
    search_listings,
)
from app.facebook_publisher import (
    FacebookPublishError,
    get_post_engagement,
    publish_post,
)
from app.click_tracker import (
    get_clicks,
    get_tracking_row_index,
    link_post,
    record_click,
    save_tracking_listing,
    tracking_id_for_post,
)
from app.airtable_client import (
    AirtableError,
    create_post_record,
    get_posts,
    get_style_performance,
    log_engagement,
    upsert_lead_record,
)
from app.internal_dashboard import collect_dashboard_data

logger = logging.getLogger("propvibe.main")

app = FastAPI()

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
    return {"status": "PropVibe backend is live"}


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


@app.get("/privacy")
def privacy():
    """Serve the privacy policy page (required for Facebook App Live mode)."""
    page = STATIC_DIR / "privacy.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Privacy policy page not found.")
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


def _listing_context(tracking_id: str) -> dict | None:
    """
    The listing a tracking link was published for, as a plain dict, or None.

    Two hops: tracking_id -> row index (recorded locally at publish time by
    app.click_tracker) -> Listing (re-read from room_listings.csv). Returns
    None whenever either hop comes up empty - an unrecognised id, a post from
    before we recorded listing indices, or a row that has since left the CSV.
    Shared by /listing-info and /chat so the two can never disagree about
    which property a lead is looking at.
    """
    row_index = get_tracking_row_index(tracking_id)
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
    return {"found": True, **context}


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


def _best_performing_style() -> str | None:
    """
    The current best-performing caption style, or None if we can't tell yet.

    Reads average engagement per style from Airtable and returns the top one.
    This is only a hint for caption generation, so it must never break /create-
    post: any problem (Airtable unconfigured, API error, no data yet) yields None
    and generate_caption falls back to a random style. get_style_performance
    already degrades to {} on failure; the broad guard here is belt-and-braces.
    """
    try:
        performance = get_style_performance()
    except Exception as exc:  # noqa: BLE001 - a hint must never crash the request
        logger.warning("Could not read style performance for the caption hint: %s", exc)
        return None

    if not performance:
        return None
    return max(performance, key=performance.get)


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
        style_tag = _best_performing_style() or random.choice(STYLE_TAGS)

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
):
    """
    Auto-ready the next un-posted room_listings.csv row: pick a listing,
    assign a demo photo, and build the same poster + caption /create-post
    does - with no photo upload or form fields required.

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
        raise HTTPException(
            status_code=502,
            detail=f"Could not read Airtable to pick the next listing: {exc}",
        )

    posted_indices = {
        post["listing_row_index"]
        for post in posted
        if post.get("listing_row_index") is not None
    }

    listing = next_unposted_listing(posted_indices, after=after)
    if listing is None:
        raise HTTPException(
            status_code=404,
            detail="No un-posted listings remain in room_listings.csv.",
        )

    try:
        photo_path = random_pool_photo()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Same reused check as the manual flow: confirms the photo actually
    # decodes (e.g. a .HEIC file when pillow-heif isn't installed) rather than
    # silently shipping a grey placeholder rectangle.
    usable_paths = _usable_photos([photo_path])

    style_tag = _best_performing_style() or random.choice(STYLE_TAGS)

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
          "listing_row_index": 17                   # optional - from /auto-create-post
        }

    Before publishing we mint a short tracking id and append its link
    (``<BASE_URL>/t/<id>``) to the CTA, so clicks route through the tracking
    endpoint. The caption, hashtags (each '#'-prefixed) and the link-carrying CTA
    are combined - separated by blank lines - into the final post text.

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


def _alternatives_for(tracking_id: str, history: list, message: str) -> list[dict] | None:
    """
    Real listings to offer this lead, or None if they aren't asking for any.

    Two steps: a cheap Claude call reads the message for "got anything else?"
    plus any budget/area/room-type they named, then we search the CSV for it.
    The property they are already looking at is excluded, so it can never come
    back as its own alternative.

    Returns None when they aren't asking - including when the intent call
    failed and degraded to its safe default - which leaves the chat turn
    exactly as it behaved before. An empty list is different and deliberate: it
    means they DID ask and nothing matched, and the assistant says so rather
    than inventing something.
    """
    intent = extract_search_intent(history, message)
    if not intent["wants_alternatives"]:
        return None

    matches = search_listings(
        max_price=intent["max_price"],
        location_keyword=intent["location_keyword"],
        room_type_keyword=intent["room_type_keyword"],
        exclude_row_index=get_tracking_row_index(tracking_id),
        limit=MAX_ALTERNATIVES,
    )
    return [_listing_to_context(listing) for listing in matches]


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
          "history":     [{"role": "user", "content": "..."}, ...]  # optional
        }

    We resolve which listing the tracking id was published for (same lookup as
    /listing-info) and hand that context to the assistant so it can talk about
    the actual property; an unknown id just means it stays general.

    If the message reads like they're asking what ELSE is available, we also
    search room_listings.csv for real alternatives - on whatever budget, area
    or room type they mentioned, minus the property they're already on - and
    pass those in too, so the assistant offers listings that genuinely exist
    instead of improvising. See _alternatives_for.

    Alongside the reply, the assistant reports a qualification `status` and
    what it has picked up about the lead's budget and move-in timeline. The
    signals are written to the Airtable Leads table - one row per tracking id,
    updated as the conversation goes - but deliberately NOT returned here: the
    lead should never see that they're being scored. Airtable logging is
    best-effort; if it fails we log a warning and still answer.

    Returns JSON::

        {"reply": "<what to say back>", "status": "lead" | "prospect"}

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

    listing_context = _listing_context(tracking_id)
    other_listings = _alternatives_for(tracking_id, history, message)

    try:
        result = qualify_lead(history, message, listing_context, other_listings)
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

    return {"reply": result["reply"], "status": result["status"]}


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
                       {"facebook_post_id": "...", "clicks": 3, "likes": 0,
                        "comments": 0, "logged": true,
                        "note": "facebook_read_failed"}, ... ]
        }

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
        clicks = get_clicks(tracking_id)
        logger.info(
            "sync: post fb=%s tracking_id=%r -> clicks=%d",
            fb_id,
            tracking_id or None,
            clicks,
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
