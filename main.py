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
"""

import base64
import shutil
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

# Load variables from a local .env file (e.g. ANTHROPIC_API_KEY) into the
# process environment at import time. This is a no-op in production where the
# platform injects real env vars, and it deliberately does NOT raise if the key
# is missing - endpoints that actually need it validate at call time and return
# a clean error instead of crashing the whole app on startup.
load_dotenv()

# NOTE: this is the `from X import Y` form on purpose - it binds only
# `generate_poster`, so the `app` package name never lands in this module's
# namespace and cannot shadow the `app = FastAPI()` instance below.
from app.poster_generator import generate_poster
from app.copy_generator import CaptionError, generate_caption

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
