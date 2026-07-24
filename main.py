"""
PropVibe - FastAPI entrypoint.

Exposes the poster generator over HTTP so the front end (or a webhook, or a
plain curl) can post some property photos plus listing details and get a
ready-to-share 1080x1080 PNG straight back in the response body.

Endpoints:
  GET  /              - health check
  GET  /dashboard     - minimal approval/preview page (static HTML)
  POST /generate-poster - returns the poster PNG directly
  POST /create-post   - returns poster (base64) + Claude-written caption as JSON
  POST /publish-post  - publishes a poster + caption to a Facebook Page
"""

import base64
import binascii
import logging
import os
import secrets
import shutil
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
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

# NOTE: this is the `from X import Y` form on purpose - it binds only
# `generate_poster`, so the `app` package name never lands in this module's
# namespace and cannot shadow the `app = FastAPI()` instance below.
from app.poster_generator import generate_poster
from app.copy_generator import CaptionError, generate_caption
from app.facebook_publisher import FacebookPublishError, publish_post
from app.click_tracker import get_clicks, record_click
from app.airtable_client import AirtableError, create_post_record

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


@app.get("/t/{tracking_id}")
def track_click(tracking_id: str):
    """
    Record a click on a post's tracking link, then redirect to the listing page.

    Every published post's CTA carries a ``/t/{tracking_id}`` link. When a lead
    clicks it we bump a local counter (see app.click_tracker) keyed by that id -
    fast, and independent of Airtable being reachable - then 302 them on to the
    placeholder listing page. /sync-engagement later folds these click counts
    into each post's Airtable engagement row.
    """
    record_click(tracking_id)
    return RedirectResponse(url="/listing", status_code=302)


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


@app.post("/generate-poster")
async def generate_poster_endpoint(
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

        output_path = job_dir / f"poster_{uuid.uuid4().hex[:8]}.png"

        try:
            generate_poster(
                photos=saved_paths,
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


@app.post("/create-post")
async def create_post_endpoint(
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

    Same multipart inputs as /generate-poster. Returns JSON:

        {
          "caption":       "<engaging post text>",
          "hashtags":      ["luxuryliving", ...],   # no '#' prefix
          "cta":           "DM us to book a viewing",
          "poster_base64": "<base64 PNG bytes>"     # no data: prefix
        }

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

        output_path = job_dir / f"poster_{uuid.uuid4().hex[:8]}.png"
        try:
            generate_poster(
                photos=saved_paths,
                price=price,
                location=location,
                bedrooms=bedroom_count,
                bathrooms=bathroom_count,
                output_path=str(output_path),
            )
        except ValueError as exc:
            # Belt-and-braces: generate_poster() raises this for an empty photo
            # list, which we already guard above.
            raise HTTPException(status_code=400, detail=str(exc))

        poster_base64 = base64.b64encode(output_path.read_bytes()).decode("ascii")

        # The caption call hits the network and may fail for reasons the user
        # can act on (missing key, rate limit, ...). Surface the clean message.
        try:
            copy = generate_caption(price, location, bedroom_count, bathroom_count)
        except CaptionError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

    return {
        "caption": copy["caption"],
        "hashtags": copy["hashtags"],
        "cta": copy["cta"],
        "poster_base64": poster_base64,
    }


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
async def publish_post_endpoint(payload: dict = Body(...)):
    """
    Publish an already-generated poster + caption to the Facebook Page.

    Send JSON:

        {
          "poster_base64": "<base64 PNG bytes>",   # as returned by /create-post
          "caption":       "<post text>",
          "hashtags":      ["luxuryliving", ...],  # no '#' prefix needed
          "cta":           "DM us to book a viewing",
          "style_tag":     "Warm",                 # optional - from /create-post
          "template_id":   "A"                     # optional - from /create-post
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
