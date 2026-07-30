"""
PropVibe - Mascot Video Studio (stretch feature, dev-only)
===========================================================

A standalone, self-contained feature: drop the BeLive mascot into a real
listing photo and animate it with fal.ai's Kling 3.0 Pro image-to-video
model, driven by a free-text action ("waves and points at the sign").

Deliberately isolated. Everything this feature needs - the router, the page
route, the fal.ai call, the cache and the spend cap - lives in this one
module. main.py imports the router and registers it; nothing else in the
existing app changes. There is no nav link to /mascot-studio anywhere: it is
reachable only by typing the URL, which is why it carries no auth (unlike
/sync-engagement's X-Sync-Secret) - it is not linked, not advertised, and
not part of the demo flow.

WHAT IS REUSED (not reinvented): the listing -> photo mapping and the
photo-to-servable-JPEG conversion both come from the existing modules that
already own them - app.listings_source.pool_photo_for_listing /
PHOTO_POOL_DIR, and app.poster_generator.photo_as_jpeg_bytes. That last one
matters: 4 of the 10 demo pool photos are .HEIC, which neither a browser nor
fal.ai can read, and it is what already re-encodes them to JPEG for
/pool-photo/{filename}.

KNOWN LIMITATIONS, accepted on purpose (this is a pre-presentation stretch
feature, not production code):

  * The video cache and the daily counter are plain in-memory dicts/ints.
    Both reset on every server restart and every Railway redeploy. That
    means a redeploy can re-charge a previously generated (photo, action)
    pair and can reset the "X / 20" cap back to zero. No persistence is
    built for either - see CACHE / CAP notes below.
  * Railway can run more than one worker/instance. Each would keep its own
    cache and its own counter, so the cap is per-process, not global.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import threading
from pathlib import Path
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import FileResponse, Response
from PIL import Image

from app.listings_source import PHOTO_POOL_DIR, load_listings, pool_photo_for_listing
from app.poster_generator import photo_as_jpeg_bytes

logger = logging.getLogger("propvibe.mascot_video")

router = APIRouter()

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = REPO_ROOT / "templates" / "mascot_studio.html"

# The mascot lives under static/ (not data/, which is gitignored) so it is
# committed and therefore present in the Railway deploy. Served by
# mascot_asset() below, giving it a stable public URL.
#
# Two files, because Kling's `elements` validation requires a frontal image
# AND at least one additional reference image (see _submit_generation):
# belive_mascot.png is the neutral standing pose used as the frontal/main
# view, belive_mascot_ref.png is a second pose of the same character.
MASCOT_DIR = REPO_ROOT / "static" / "mascot"
MASCOT_PATH = MASCOT_DIR / "belive_mascot.png"
MASCOT_REF_PATHS = [MASCOT_DIR / "belive_mascot_ref.png"]
MASCOT_URL_PATH = "/static/mascot/belive_mascot.png"

# fal.ai queue API for Kling 3.0 Pro image-to-video. The QUEUE endpoint
# (queue.fal.run), not the synchronous one used by app.photo_condition:
# a video generation takes minutes, not the tens of seconds an image edit
# takes, so it has to be submit-then-poll or the HTTP request would time out.
FAL_SUBMIT_URL = "https://queue.fal.run/fal-ai/kling-video/v3/pro/image-to-video"

# CAREFUL: a job's status/result URLs are NOT the submit URL + /requests/<id>.
# fal.ai scopes queued requests to the APP (fal-ai/kling-video), not to the
# specific versioned endpoint path, so
#   .../v3/pro/image-to-video/requests/<id>/status  -> HTTP 405
#   .../kling-video/requests/<id>/status            -> HTTP 200
# Rather than hardcode that, we use the `status_url` / `response_url` that
# fal.ai hands back in its own submit response, which is authoritative.
# This constant is kept only to document the convention.
FAL_REQUESTS_URL = "https://queue.fal.run/fal-ai/kling-video/requests"

# Submitting and polling are both quick calls - only the generation itself is
# slow, and that happens on fal's side between polls.
REQUEST_TIMEOUT_SECONDS = 60

# Shortest value the endpoint's `duration` enum allows. Verified against
# fal.ai's published schema for this model: duration is a STRING enum of
# "3".."15" (default "5"), so "3" is the floor - 1s and 2s are not
# selectable, which is as close to the 1-3s target as the model permits.
DURATION_SECONDS = "3"

# Hard ceiling on NEW generations per process lifetime. Only successful new
# generations count - see _Budget.
MAX_VIDEOS = 20


class MascotVideoError(Exception):
    """A clean, user-safe failure from a fal.ai mascot-video call."""


# ---------------------------------------------------------------------------
# In-memory cache and spend cap
# ---------------------------------------------------------------------------
#
# KNOWN LIMITATION (accepted): both of the below are process memory only.
# A restart or redeploy wipes the cache (so a pair may be re-billed) and
# resets the counter to 0 (so the cap can be exceeded across a redeploy).
# Persisting them would mean a new Airtable table, which is an explicit
# non-goal for this feature.

# (photo_filename, action) -> finished fal.ai video URL.
_VIDEO_CACHE: dict[tuple[str, str], str] = {}

# fal.ai request_id -> {"key": (photo_filename, action), "status_url", "result_url"}.
# A job is removed from here the moment it is finalised (success or failure),
# which is also what makes the counter increment exactly once per job even
# though the browser polls the status endpoint repeatedly.
_JOBS: dict[str, dict] = {}

# Guards all three of the above plus the counters. FastAPI serves sync
# endpoints on a thread pool, so two concurrent submits really can race.
_LOCK = threading.Lock()

# Successful new generations completed so far.
_generated = 0

# Jobs submitted to fal.ai that have not yet succeeded or failed. Counted
# against the cap alongside _generated so that N concurrent in-flight jobs
# can never overshoot MAX_VIDEOS while they are all still pending.
_in_flight = 0


def _budget_snapshot() -> dict:
    """The counter values the page displays, read under the lock."""
    with _LOCK:
        return {
            "generated": _generated,
            "in_flight": _in_flight,
            "cap": MAX_VIDEOS,
            "cap_reached": _generated + _in_flight >= MAX_VIDEOS,
        }


# ---------------------------------------------------------------------------
# Photo resolution
# ---------------------------------------------------------------------------


def _resolve_pool_photo(raw: object) -> Path:
    """
    Bare-filename + no-path-traversal + must-exist validation for a pool photo.

    Same rules as main.py's `_resolve_pool_photo`, restated here rather than
    imported: main.py imports THIS module (to register the router), so
    importing back from main.py would be a circular import. main.py's own
    version is already a deliberate restatement of /regenerate-poster's
    inline checks for the same reason of keeping working code untouched, so
    this follows the precedent already set in the codebase. The shared source
    of truth that actually matters - PHOTO_POOL_DIR - is imported, not copied.
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


def _photo_choices() -> list[dict]:
    """
    The dropdown's options: one per DISTINCT pool photo, labelled with a real
    listing that uses it.

    Keyed on the photo, not on the listing, because the cache key is the
    photo: pool_photo_for_listing() maps 177 CSV listings onto only 10 pool
    photos (row_index % pool_size), so a listing-keyed dropdown would offer
    177 entries where 167 of them are duplicate work that would silently
    resolve to a cache hit. Ten honest options instead.
    """
    labels: dict[str, str] = {}
    for listing in load_listings():
        try:
            photo_filename = Path(pool_photo_for_listing(listing.row_index)).name
        except FileNotFoundError:
            break
        # First listing wins, so the label is stable between page loads.
        labels.setdefault(
            photo_filename,
            f"{listing.condo_name} - {listing.room_type} ({listing.price})",
        )

    choices = []
    for photo_path in sorted(p for p in PHOTO_POOL_DIR.iterdir() if p.is_file()):
        choices.append(
            {
                "photo_filename": photo_path.name,
                "label": labels.get(photo_path.name, photo_path.stem),
                # The existing no-AI-cost preview route, reused as-is.
                "preview_url": f"/pool-photo/{photo_path.name}",
            }
        )
    return choices


# ---------------------------------------------------------------------------
# fal.ai call
# ---------------------------------------------------------------------------


def _require_api_key() -> str:
    """
    Read FAL_API_KEY at call time, not at import.

    Same pattern as app.photo_condition (and app.copy_generator): the rest of
    the app must boot fine with this unset, and only a request that actually
    wants to spend money should fail.
    """
    api_key = os.environ.get("FAL_API_KEY")
    if not api_key:
        raise MascotVideoError(
            "FAL_API_KEY is not set. Add it to your .env file "
            "(see .env.example), then restart the server."
        )
    return api_key


def _data_uri(raw: bytes, mime: str) -> str:
    """Base64-encode bytes into a data: URI, which fal.ai accepts in its image URL fields."""
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _listing_photo_data_uri(photo_path: Path) -> str:
    """
    The chosen listing photo as a JPEG data URI.

    Goes through the existing photo_as_jpeg_bytes(), which is what already
    turns a .HEIC pool photo into something servable - Kling only accepts
    jpg/jpeg/png, so a raw .HEIC would be rejected outright.

    A data URI rather than this server's own public /pool-photo/ URL because
    fal.ai fetches image URLs from its own infrastructure: BASE_URL defaults
    to http://localhost:8000 and is not set in .env, so a URL built from it
    would be unreachable from fal and the generation would fail with a
    confusing remote error. Inlining the bytes works identically in local dev
    and on Railway. This mirrors what app.photo_condition already does for
    its nano-banana edit calls.
    """
    jpeg_bytes = photo_as_jpeg_bytes(str(photo_path))
    if jpeg_bytes is None:
        raise MascotVideoError("That listing photo could not be read as an image.")
    return _data_uri(jpeg_bytes, "image/jpeg")


def _mascot_data_uri(mascot_path: Path) -> str:
    """
    One mascot asset as a PNG data URI, flattened onto white.

    The committed assets are RGBA with a transparent background (which is what
    makes them look right on the page). Transparency is flattened onto white
    before it goes to Kling: an alpha channel in a character reference image
    is a common source of edge artefacts / grey halos in the generated video.
    """
    if not mascot_path.is_file():
        raise MascotVideoError(
            f"The mascot asset is missing at {mascot_path.relative_to(REPO_ROOT)}."
        )

    with Image.open(mascot_path) as mascot:
        mascot.load()
        if mascot.mode in ("RGBA", "LA", "P"):
            rgba = mascot.convert("RGBA")
            flattened = Image.new("RGB", rgba.size, "white")
            flattened.paste(rgba, mask=rgba.split()[-1])
        else:
            flattened = mascot.convert("RGB")

    buffer = io.BytesIO()
    flattened.save(buffer, format="PNG")
    return _data_uri(buffer.getvalue(), "image/png")


def _build_prompt(action: str) -> str:
    """
    The single motion instruction sent to Kling.

    "@Element1" is Kling 3.0's own syntax for referring to an entry in the
    `elements` array from inside the prompt - the mascot is supplied as an
    element, so this is how the model knows which character the action
    belongs to. Without the @Element1 references the reference image is
    largely ignored.
    """
    return (
        "A cartoon mascot character @Element1 - a friendly smart door-lock "
        "character with a small orange cape - is present in this real property "
        f"photo. @Element1 {action}. Composite @Element1 naturally into the "
        "scene at a believable human-like scale, standing on the floor/ground, "
        "lit to match the room, with a soft contact shadow. Keep the property "
        "itself, the camera angle and the framing completely static - only "
        "@Element1 moves. No on-screen text or captions."
    )


def _submit_generation(photo_path: Path, action: str) -> dict:
    """
    Submit ONE Kling image-to-video job to fal.ai's queue.

    Returns {"request_id", "status_url", "result_url"} - fal.ai's own status
    and result URLs are used verbatim rather than constructed, see
    FAL_REQUESTS_URL.

    Field names are taken from fal.ai's published schema for
    fal-ai/kling-video/v3/pro/image-to-video:

        start_image_url  (required) - the base scene, i.e. the listing photo
        elements[]       - characters/objects to keep consistent, addressed in
                           the prompt as @Element1. GOTCHA: each entry must
                           supply BOTH `frontal_image_url` AND at least one
                           `reference_image_urls` entry (or a `video_url`) -
                           frontal alone is accepted by the queue but then
                           fails validation at inference time with "Either
                           frontal_image_url and reference_image_urls or
                           video_url must be provided". Hence two mascot
                           assets. 1-3 reference images are supported.
        duration         - STRING enum "3".."15"; "3" is the shortest allowed
        generate_audio   - defaults to TRUE, so it must be sent as False

    Raises:
        MascotVideoError: missing API key, unreadable asset, network failure,
            or a non-2xx / unparseable response. Safe to show a user.
    """
    api_key = _require_api_key()
    payload = {
        "prompt": _build_prompt(action),
        "start_image_url": _listing_photo_data_uri(photo_path),
        "elements": [
            {
                "frontal_image_url": _mascot_data_uri(MASCOT_PATH),
                "reference_image_urls": [_mascot_data_uri(p) for p in MASCOT_REF_PATHS],
            }
        ],
        "duration": DURATION_SECONDS,
        "generate_audio": False,
    }

    try:
        response = httpx.post(
            FAL_SUBMIT_URL,
            headers={"Authorization": f"Key {api_key}"},
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except httpx.RequestError as exc:
        raise MascotVideoError(f"Could not reach fal.ai to start the video (network error): {exc}.")

    if not response.is_success:
        raise MascotVideoError(_error_message(response))

    try:
        body = response.json()
    except ValueError:
        raise MascotVideoError("fal.ai returned an unreadable response when starting the video.")

    request_id = body.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise MascotVideoError("fal.ai accepted the job but did not return a request id.")

    status_url = body.get("status_url")
    result_url = body.get("response_url")
    if not isinstance(status_url, str) or not isinstance(result_url, str):
        # Fall back to the documented convention if fal ever stops echoing them.
        status_url = f"{FAL_REQUESTS_URL}/{request_id}/status"
        result_url = f"{FAL_REQUESTS_URL}/{request_id}"

    return {"request_id": request_id, "status_url": status_url, "result_url": result_url}


def _fetch_status(status_url: str) -> str:
    """
    Poll one queued job. Returns fal.ai's raw status string, e.g.
    "IN_QUEUE", "IN_PROGRESS" or "COMPLETED".
    """
    api_key = _require_api_key()
    try:
        response = httpx.get(
            status_url,
            headers={"Authorization": f"Key {api_key}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except httpx.RequestError as exc:
        raise MascotVideoError(f"Could not reach fal.ai to check the video's status: {exc}.")

    if not response.is_success:
        raise MascotVideoError(_error_message(response))

    try:
        body = response.json()
    except ValueError:
        raise MascotVideoError("fal.ai returned an unreadable status response.")

    status = body.get("status")
    if not isinstance(status, str) or not status:
        raise MascotVideoError("fal.ai's status response did not include a status.")
    return status


def _fetch_result_url(result_url: str) -> str:
    """
    Collect the finished video's URL for a COMPLETED job.

    Note that "COMPLETED" only means fal.ai finished processing the request -
    a request rejected by the model's own input validation also reports
    COMPLETED, and surfaces its error here rather than at submit time. That is
    why this returns a MascotVideoError (via _error_message) instead of
    assuming a video is present.
    """
    api_key = _require_api_key()
    try:
        response = httpx.get(
            result_url,
            headers={"Authorization": f"Key {api_key}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except httpx.RequestError as exc:
        raise MascotVideoError(f"Could not reach fal.ai to collect the finished video: {exc}.")

    if not response.is_success:
        raise MascotVideoError(_error_message(response))

    try:
        body = response.json()
    except ValueError:
        raise MascotVideoError("fal.ai returned an unreadable result response.")

    video = body.get("video")
    video_url = video.get("url") if isinstance(video, dict) else None
    if not isinstance(video_url, str) or not video_url:
        raise MascotVideoError("fal.ai's result did not include a video URL.")
    return video_url


def _error_message(response: httpx.Response) -> str:
    """
    Pull fal.ai's own error text out of a failed response.

    Same shape-handling as app.photo_condition._error_message - fal.ai is a
    FastAPI service, so a rejected payload comes back as a `detail` list.
    """
    try:
        body = response.json()
    except ValueError:
        body = None

    if isinstance(body, dict):
        detail = body.get("detail") or body.get("error") or body.get("message")
        if isinstance(detail, str) and detail.strip():
            return f"fal.ai rejected the video request: {detail.strip()}"
        if isinstance(detail, list) and detail:
            first = detail[0]
            if isinstance(first, dict) and isinstance(first.get("msg"), str):
                return f"fal.ai rejected the video request: {first['msg']}"

    return f"fal.ai returned HTTP {response.status_code}."


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/mascot-studio")
def mascot_studio_page():
    """
    Serve the Mascot Video Studio page.

    A dedicated FileResponse route rather than a StaticFiles mount, matching
    how /dashboard, /listing and /privacy are already served - this app has
    no app.mount("/static", ...) anywhere.
    """
    if not TEMPLATE_PATH.exists():
        raise HTTPException(status_code=404, detail="Mascot studio page not found.")
    return FileResponse(TEMPLATE_PATH, media_type="text/html")


@router.get("/static/mascot/{filename}")
def mascot_asset(filename: str):
    """
    Serve a mascot PNG at a stable public URL.

    Both assets are reachable: belive_mascot.png (the frontal pose the page
    shows next to the title) and belive_mascot_ref.png (the extra reference
    pose). Matched against an allow-list of the two known filenames rather
    than joined onto a path, so this cannot be walked out of MASCOT_DIR.

    The fal.ai call does NOT fetch these URLs - it receives the same bytes
    inlined as data URIs instead (see _mascot_data_uri), because fal's
    servers cannot reach a localhost BASE_URL during local development.
    """
    allowed = {path.name for path in [MASCOT_PATH, *MASCOT_REF_PATHS]}
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="Mascot asset not found.")

    asset = MASCOT_DIR / filename
    if not asset.is_file():
        raise HTTPException(status_code=404, detail="Mascot asset not found.")
    return FileResponse(asset, media_type="image/png")


@router.get("/mascot-studio/photos")
def mascot_studio_photos():
    """The dropdown's options plus the current counter state, for page load."""
    return {
        "photos": _photo_choices(),
        "mascot_url": MASCOT_URL_PATH,
        "budget": _budget_snapshot(),
    }


def _require_action(raw: object) -> str:
    """
    Validate the free-text mascot action.

    Only surrounding whitespace is stripped before it becomes part of the
    cache key - the key is otherwise an EXACT match on the text typed, so
    "waves" and "Waves" are two different (and separately billed) entries.
    A trailing space alone should not cost money, hence the strip.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise HTTPException(status_code=400, detail="Describe what the mascot should do.")
    action = raw.strip()
    if len(action) > 500:
        raise HTTPException(status_code=400, detail="That action description is too long (max 500 characters).")
    return action


def _download_url(photo_filename: str, action: str) -> str:
    """The page's download link for a finished (photo, action) pair."""
    query = urlencode({"photo_filename": photo_filename, "action": action})
    return f"/mascot-studio/download?{query}"


@router.post("/mascot-studio/generate")
# `def`, not `async def` - the fal.ai submit and the Pillow/base64 work are
# both blocking and there is no `await` in the body, matching every other
# third-party-calling endpoint in this app.
def mascot_studio_generate(payload: dict = Body(...)):
    """
    Start (or serve from cache) one mascot video.

    Send JSON: {"photo_filename": "IMG_5433.JPG", "action": "waves and points at the sign"}

    CACHE FIRST, ALWAYS. An exact (photo_filename, action) match returns the
    cached video immediately: no fal.ai call, no cost, no counter increment.
    Only a genuine miss checks the cap and submits a job.

    Returns either
        {"status": "done", "cached": true, "video_url": ..., "download_url": ..., "budget": {...}}
    or
        {"status": "queued", "request_id": ..., "budget": {...}}
    """
    global _in_flight

    photo_path = _resolve_pool_photo(payload.get("photo_filename"))
    photo_filename = photo_path.name
    action = _require_action(payload.get("action"))
    key = (photo_filename, action)

    with _LOCK:
        cached_url = _VIDEO_CACHE.get(key)
        if cached_url is None:
            # Reserve a slot before releasing the lock, so two simultaneous
            # submits can't both pass a cap check that only one of them fits
            # under. Released again in the `except` below if the submit fails.
            if _generated + _in_flight >= MAX_VIDEOS:
                raise HTTPException(
                    status_code=429,
                    detail=f"Daily cap reached - {MAX_VIDEOS} videos have already been generated.",
                )
            _in_flight += 1

    if cached_url is not None:
        return {
            "status": "done",
            "cached": True,
            "video_url": cached_url,
            "download_url": _download_url(photo_filename, action),
            "budget": _budget_snapshot(),
        }

    try:
        job = _submit_generation(photo_path, action)
    except MascotVideoError as exc:
        with _LOCK:
            _in_flight -= 1
        raise HTTPException(status_code=502, detail=str(exc))

    with _LOCK:
        _JOBS[job["request_id"]] = {
            "key": key,
            "status_url": job["status_url"],
            "result_url": job["result_url"],
        }

    return {
        "status": "queued",
        "cached": False,
        "request_id": job["request_id"],
        "budget": _budget_snapshot(),
    }


@router.get("/mascot-studio/status/{request_id}")
def mascot_studio_status(request_id: str):
    """
    Poll a submitted job. The page calls this every few seconds.

    Returns {"status": "generating"} until fal.ai reports COMPLETED, then
    {"status": "done", "video_url": ..., "download_url": ...}.

    Finalisation (counter increment + cache write) happens here, exactly
    once: the job's entry is popped from _JOBS under the lock, so repeated
    polls of an already-finished job cannot double-count it. The counter is
    incremented on COMPLETION rather than on submit, so a job that fails
    inside fal's queue never counts against the 20.
    """
    global _generated, _in_flight

    with _LOCK:
        job = _JOBS.get(request_id)

    if job is None:
        # Either an id this process never issued, or one it already finalised
        # (jobs are popped from _JOBS on completion so they cannot be counted
        # twice). The page stops polling as soon as it sees "done", so it
        # never lands here on a normal run.
        raise HTTPException(status_code=404, detail="Unknown or already-finished job.")

    key = job["key"]

    try:
        status = _fetch_status(job["status_url"])
    except MascotVideoError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    if status != "COMPLETED":
        # IN_QUEUE / IN_PROGRESS - still costing time, not yet counted.
        return {"status": "generating", "fal_status": status, "budget": _budget_snapshot()}

    try:
        video_url = _fetch_result_url(job["result_url"])
    except MascotVideoError as exc:
        # The generation itself may well have succeeded and been billed, but
        # we have no URL to show. Release the reserved slot without counting
        # it, and log loudly - this pair is not cached, so a retry re-bills.
        with _LOCK:
            if _JOBS.pop(request_id, None) is not None:
                _in_flight -= 1
        logger.error(
            "Kling job %s for %r completed but its result could not be collected - "
            "this pair is not cached and a retry will re-bill: %s",
            request_id,
            key,
            exc,
        )
        raise HTTPException(status_code=502, detail=str(exc))

    with _LOCK:
        if _JOBS.pop(request_id, None) is not None:
            _in_flight -= 1
            _generated += 1
            _VIDEO_CACHE[key] = video_url

    photo_filename, action = key
    return {
        "status": "done",
        "cached": False,
        "video_url": video_url,
        "download_url": _download_url(photo_filename, action),
        "budget": _budget_snapshot(),
    }


@router.get("/mascot-studio/download")
def mascot_studio_download(
    photo_filename: str = Query(...),
    action: str = Query(...),
):
    """
    Stream a finished video back with a filename, so the page's link actually
    downloads instead of navigating.

    Takes the (photo_filename, action) cache key rather than a video URL:
    accepting a URL here would make this endpoint a fetch-anything proxy.
    Only URLs this server itself put in the cache can be reached.
    """
    photo_filename = _resolve_pool_photo(photo_filename).name
    action = _require_action(action)

    with _LOCK:
        video_url = _VIDEO_CACHE.get((photo_filename, action))

    if video_url is None:
        raise HTTPException(status_code=404, detail="No generated video for that photo and action.")

    try:
        upstream = httpx.get(video_url, timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=True)
        upstream.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Could not download the video from fal.ai: {exc}")

    # A short hash of the pair keeps the filename unique per (photo, action)
    # without dragging the user's free text into a filename.
    suffix = hashlib.sha256(f"{photo_filename}|{action}".encode()).hexdigest()[:8]
    name = f"mascot_{Path(photo_filename).stem}_{suffix}.mp4"
    return Response(
        content=upstream.content,
        media_type="video/mp4",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
