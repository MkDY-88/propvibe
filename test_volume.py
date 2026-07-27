"""
PropVibe - volume / stress test for the full /create-post pipeline.

Where test_poster.py exercises `generate_poster()` on its own, this drives the
*whole* pipeline over HTTP - upload handling, validation, poster generation,
trend research and caption generation - against a deliberately nasty spread of
listings, and reports what survived.

Run it against a real local server:

    uvicorn main:app --host 127.0.0.1 --port 8000     # terminal 1
    python test_volume.py                             # terminal 2

Only /create-post is hit. /publish-post is deliberately NOT exercised - that
posts to a real Facebook Page and this harness would spam it.

    python test_volume.py                # the full matrix, then the concurrency check
    python test_volume.py --matrix-only
    python test_volume.py --concurrency-only
    python test_volume.py --base-url http://127.0.0.1:9000

Every case declares what it SHOULD do:

    expect="ok"    a 200 with a usable poster + caption
    expect="4xx"   a clean client error - the input is genuinely invalid and
                   rejecting it is correct behaviour, not a bug

Anything else - a 500, a hang, a 200 carrying a broken poster - is a finding.
Returned posters are written to test_output/volume_posters/ so they can be
eyeballed, and the full result table is dumped to test_output/volume_results.json.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

from app.poster_generator import CANVAS_SIZE, FONT_BOLD_PATH, PLACEHOLDER_GRAY, _load_font

# Windows consoles default to cp1252, which cannot print the Chinese/Malay test
# locations this harness deliberately uses. Force UTF-8 so the report itself
# doesn't blow up while testing unicode handling.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_BASE_URL = "http://127.0.0.1:8000"

OUTPUT_DIR = Path(__file__).resolve().parent / "test_output"
PHOTO_DIR = OUTPUT_DIR / "volume_photos"
POSTER_DIR = OUTPUT_DIR / "volume_posters"
RESULTS_PATH = OUTPUT_DIR / "volume_results.json"

# Generous per-request ceiling. The point is to catch a hang, not to fail slow
# but working requests: a cold /create-post is a web search (capped at 15s) plus
# a caption call, so ~25s is a realistic worst case and anything past 90s is
# stuck rather than slow.
REQUEST_TIMEOUT = 90.0

# Upload guard in main.py, mirrored here so the oversized fixture is built just
# past it rather than guessing.
MAX_PHOTO_BYTES = 15 * 1024 * 1024


# ---------------------------------------------------------------------------
# FIXTURES - every photo this harness uploads is generated on the fly
# ---------------------------------------------------------------------------

# Plain coloured rectangles with a big number stamped on them, deliberately in
# assorted sizes and aspect ratios so a squashed crop would be obvious.
GOOD_SPECS = [
    ("good_1.jpg", (1600, 900), (46, 92, 138)),  # wide   - blue
    ("good_2.jpg", (900, 1600), (138, 84, 46)),  # tall   - brown
    ("good_3.jpg", (1200, 1200), (58, 110, 74)),  # square - green
    ("good_4.jpg", (1400, 1050), (120, 52, 92)),  # 4:3    - plum
    ("good_5.jpg", (1000, 1000), (140, 118, 40)),  # square - olive
    ("good_6.jpg", (1600, 1000), (60, 60, 90)),  # wide   - slate
    ("good_7.jpg", (1100, 1400), (100, 70, 70)),  # tall   - rose
]


def _numbered_photo(size: tuple[int, int], colour: tuple[int, int, int], label: str) -> Image.Image:
    """A flat coloured rectangle with `label` centred on it."""
    img = Image.new("RGB", size, colour)
    draw = ImageDraw.Draw(img)
    # Capped - on the multi-thousand-pixel fixtures a proportional size would be
    # a 1500pt glyph, which is slow to rasterise for no extra signal.
    font_size = max(48, min(400, min(size) // 4))
    draw.text(
        (size[0] // 2, size[1] // 2),
        label,
        font=_load_font(FONT_BOLD_PATH, font_size),
        fill=(255, 255, 255),
        anchor="mm",
    )
    return img


def build_fixtures() -> dict[str, Path]:
    """
    Create every test photo on disk and return them keyed by short name.

    Regenerated on each run so the suite has no binary fixtures to commit and
    can't drift out of sync with what the tests assume.
    """
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for index, (name, size, colour) in enumerate(GOOD_SPECS, start=1):
        path = PHOTO_DIR / name
        _numbered_photo(size, colour, str(index)).save(path, format="JPEG", quality=90)
        paths[f"good_{index}"] = path

    # A valid JPEG whose body has been scrambled - the header parses, the scan
    # data is garbage.
    buffer = io.BytesIO()
    _numbered_photo((1600, 900), (200, 40, 40), "X").save(buffer, format="JPEG", quality=90)
    valid_jpeg = buffer.getvalue()

    scrambled = bytearray(valid_jpeg)
    rng = random.Random(1)
    for offset in range(200, len(scrambled), 7):
        scrambled[offset] = rng.randrange(256)
    corrupt = PHOTO_DIR / "corrupt.jpg"
    corrupt.write_bytes(bytes(scrambled))
    paths["corrupt"] = corrupt

    # A JPEG cut off a third of the way through - the classic half-finished
    # upload.
    truncated = PHOTO_DIR / "truncated.jpg"
    truncated.write_bytes(valid_jpeg[: len(valid_jpeg) // 3])
    paths["truncated"] = truncated

    # A text file wearing a .jpg extension.
    not_image = PHOTO_DIR / "not_an_image.jpg"
    not_image.write_bytes(b"Dear server, I am a text file in a trenchcoat.\n" * 40)
    paths["not_image"] = not_image

    # Zero bytes on disk.
    empty = PHOTO_DIR / "empty.jpg"
    empty.write_bytes(b"")
    paths["empty"] = empty

    # Big but legitimate: 54 megapixels, comfortably under Pillow's
    # decompression-bomb ceiling. This one MUST still crop and resize correctly.
    huge_ok = PHOTO_DIR / "huge_54mp.jpg"
    if not huge_ok.exists():
        _numbered_photo((9000, 6000), (30, 120, 120), "BIG").save(
            huge_ok, format="JPEG", quality=85
        )
    paths["huge_ok"] = huge_ok

    # 225 megapixels of flat colour: under a megabyte on disk, way past Pillow's
    # bomb ceiling once decoded. Small file, enormous decode - exactly the shape
    # of input that sneaks past a byte-size check.
    bomb = PHOTO_DIR / "huge_225mp.png"
    if not bomb.exists():
        Image.new("RGB", (15000, 15000), (20, 80, 140)).save(bomb, format="PNG")
    paths["bomb"] = bomb

    # Past the 15 MB upload guard. Random noise barely compresses, so a PNG of
    # it lands just over the limit.
    oversized = PHOTO_DIR / "oversized.png"
    if not oversized.exists() or oversized.stat().st_size <= MAX_PHOTO_BYTES:
        side = 2600  # 2600 * 2600 * 3 bytes ~ 19 MB of incompressible noise
        Image.frombytes("RGB", (side, side), os.urandom(side * side * 3)).save(
            oversized, format="PNG", compress_level=1
        )
    paths["oversized"] = oversized

    return paths


# ---------------------------------------------------------------------------
# THE MATRIX
# ---------------------------------------------------------------------------

NORMAL_EN = "Mont Kiara, Kuala Lumpur"
MALAY = "Taman Sri Hartamas, Bukit Kiara, Selangor"
CHINESE = "吉隆坡蒙特基亚拉, 马来西亚"
LONG_LOCATION = (
    "Jalan Persiaran Bukit Bintang Damansara Heights Taman Tun Dr Ismail " * 45
)  # ~3000 characters


@dataclass
class Case:
    """One row of the matrix: what we send, and what should come back."""

    name: str
    group: str
    photos: list[str]
    price: str
    location: str
    bedrooms: str
    bathrooms: str
    expect: str  # "ok" or "4xx"
    why: str = ""
    # How many of `photos` are actually decodable. Only set it when that differs
    # from len(photos) - the template the server picks follows the photos it can
    # really use, not the number that were uploaded.
    usable: int | None = None

    @property
    def usable_count(self) -> int:
        return len(self.photos) if self.usable is None else self.usable


def build_matrix() -> list[Case]:
    good = [f"good_{n}" for n in range(1, 8)]

    cases: list[Case] = [
        # --- photo counts -------------------------------------------------
        Case("photos_0", "photo count", [], "RM 450,000", NORMAL_EN, "3", "2", "4xx",
             "no photos is genuinely invalid input"),
        Case("photos_1", "photo count", good[:1], "RM 450,000", NORMAL_EN, "3", "2", "ok"),
        Case("photos_2", "photo count", good[:2], "RM 450,000", NORMAL_EN, "3", "2", "ok"),
        Case("photos_3", "photo count", good[:3], "RM 780,000", NORMAL_EN, "3", "2", "ok"),
        Case("photos_4", "photo count", good[:4], "RM 2,100,000", NORMAL_EN, "5", "4", "ok"),
        Case("photos_7", "photo count", good[:7], "RM 620,000", NORMAL_EN, "2", "2", "ok",
             "well past the 4-cell grid - expect a '+3 more' badge"),

        # --- prices -------------------------------------------------------
        Case("price_normal", "price", good[:2], "RM 450,000", "Puchong, Selangor", "3", "2", "ok"),
        Case("price_huge", "price", good[:2], "RM 15,800,000", "Puchong, Selangor", "6", "7", "ok"),
        Case("price_unformatted", "price", good[:2], "450000", "Puchong, Selangor", "3", "2", "ok"),
        Case("price_shorthand", "price", good[:2], "RM450k", "Puchong, Selangor", "3", "2", "ok"),
        Case("price_empty", "price", good[:2], "   ", "Puchong, Selangor", "3", "2", "4xx",
             "blank price is genuinely invalid input"),

        # --- locations ----------------------------------------------------
        Case("loc_english", "location", good[:2], "RM 890,000", NORMAL_EN, "3", "2", "ok"),
        Case("loc_malay", "location", good[:2], "RM 890,000", MALAY, "3", "2", "ok"),
        Case("loc_chinese", "location", good[:2], "RM 890,000", CHINESE, "3", "2", "ok",
             "unicode must survive both the poster text and the caption"),
        Case("loc_chinese_grid", "location", good[:4], "RM 3,400,000", CHINESE, "4", "3", "ok",
             "same unicode check on the grid template"),
        Case("loc_very_long", "location", good[:2], "RM 890,000", LONG_LOCATION, "3", "2", "ok",
             "~3000 chars - must truncate, not stall"),
        Case("loc_empty", "location", good[:2], "RM 890,000", "", "3", "2", "4xx",
             "blank location is genuinely invalid input"),

        # --- bed / bath ---------------------------------------------------
        Case("beds_normal", "bed/bath", good[:2], "RM 700,000", "Cyberjaya, Selangor", "3", "2", "ok"),
        Case("beds_zero", "bed/bath", good[:2], "RM 700,000", "Cyberjaya, Selangor", "0", "0", "ok",
             "a studio is a real listing, not an error"),
        Case("beds_large", "bed/bath", good[:2], "RM 700,000", "Cyberjaya, Selangor", "20", "20", "ok"),
        Case("beds_word", "bed/bath", good[:2], "RM 700,000", "Cyberjaya, Selangor", "three", "2", "4xx",
             "non-numeric is genuinely invalid input"),
        Case("beds_decimal", "bed/bath", good[:2], "RM 700,000", "Cyberjaya, Selangor", "3.5", "2", "4xx"),
        Case("beds_negative", "bed/bath", good[:2], "RM 700,000", "Cyberjaya, Selangor", "-2", "2", "4xx"),
        Case("beds_missing", "bed/bath", good[:2], "RM 700,000", "Cyberjaya, Selangor", "", "2", "4xx"),

        # --- malformed / hostile photos -----------------------------------
        Case("photo_corrupt_only", "bad photo", ["corrupt"], "RM 550,000", "Subang Jaya, Selangor", "3", "2", "4xx",
             "the only photo is unreadable - a grey rectangle is not a poster"),
        Case("photo_truncated_only", "bad photo", ["truncated"], "RM 550,000", "Subang Jaya, Selangor", "3", "2", "4xx"),
        Case("photo_not_an_image", "bad photo", ["not_image"], "RM 550,000", "Subang Jaya, Selangor", "3", "2", "4xx",
             "a .txt in a .jpg costume"),
        Case("photo_empty_file", "bad photo", ["empty"], "RM 550,000", "Subang Jaya, Selangor", "3", "2", "4xx",
             "zero bytes - already guarded"),
        Case("photo_mixed_good_bad", "bad photo", ["good_1", "corrupt", "good_3", "not_image"],
             "RM 550,000", "Subang Jaya, Selangor", "4", "3", "ok",
             "2 of 4 readable - poster builds from those 2, so Template A",
             usable=2),
        Case("photo_huge_54mp", "bad photo", ["huge_ok"], "RM 4,900,000", "Bangsar, Kuala Lumpur", "5", "4", "ok",
             "big but legal - must crop correctly without timing out"),
        Case("photo_bomb_225mp", "bad photo", ["bomb"], "RM 4,900,000", "Bangsar, Kuala Lumpur", "5", "4", "4xx",
             "under 1 MB on disk, 225 MP decoded - must be refused cleanly"),
        Case("photo_oversized_bytes", "bad photo", ["oversized"], "RM 4,900,000", "Bangsar, Kuala Lumpur", "5", "4", "4xx",
             "past the 15 MB upload guard"),

        # --- everything horrible at once ----------------------------------
        Case("combo_kitchen_sink", "combo", ["good_1", "corrupt", "good_2", "good_3", "good_4", "good_5", "good_6"],
             "RM450k", CHINESE, "0", "20", "ok",
             "unicode + odd price + zero beds + a bad photo + badge overflow",
             usable=6),
    ]
    return cases


# ---------------------------------------------------------------------------
# RUNNING A CASE
# ---------------------------------------------------------------------------


@dataclass
class Result:
    name: str
    group: str
    expect: str
    status: int | None = None
    seconds: float = 0.0
    outcome: str = ""  # PASS / FAIL / CRASH / TIMEOUT / UPSTREAM
    detail: str = ""
    findings: list[str] = field(default_factory=list)


def _files_for(case: Case, fixtures: dict[str, Path]) -> list[tuple[str, tuple[str, bytes, str]]]:
    """Build the multipart `photos` parts for a case."""
    parts = []
    for key in case.photos:
        path = fixtures[key]
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        parts.append(("photos", (path.name, path.read_bytes(), mime)))
    return parts


# Left edge of the region we sample when checking the text block actually got
# drawn - just inside the poster's MARGIN_X.
TEXT_SAMPLE_LEFT = 40


def _inspect_poster(png_bytes: bytes) -> list[str]:
    """
    Look at a returned poster and report anything visibly wrong.

    Cheap structural checks only - this can't judge taste, but it does catch the
    failure modes that actually happen: wrong canvas size, an all-placeholder
    poster where every photo failed to load, and a text block that never got
    drawn.
    """
    problems: list[str] = []
    try:
        poster = Image.open(io.BytesIO(png_bytes))
        poster.load()
    except Exception as exc:  # noqa: BLE001 - any decode failure is the finding
        return [f"returned poster is not a readable image: {type(exc).__name__}: {exc}"]

    if poster.size != (CANVAS_SIZE, CANVAS_SIZE):
        problems.append(f"poster is {poster.size}, expected {(CANVAS_SIZE, CANVAS_SIZE)}")

    poster = poster.convert("RGB")

    # Sample the photo area (top 600px, which is photo on both templates) and
    # see how much of it is the "this photo failed to load" grey.
    photo_area = poster.crop((0, 0, CANVAS_SIZE, 600))
    pixels = list(photo_area.resize((60, 34)).getdata())
    grey = sum(1 for p in pixels if all(abs(p[i] - PLACEHOLDER_GRAY[i]) <= 6 for i in range(3)))
    grey_share = grey / len(pixels)
    if grey_share > 0.95:
        problems.append("photo area is entirely the failed-to-load placeholder grey")
    elif grey_share > 0.2:
        problems.append(f"{grey_share:.0%} of the photo area is failed-to-load placeholder grey")

    # The text block lives bottom-left. If nothing was drawn there it is a solid
    # colour; real text gives a spread of values.
    text_area = poster.crop((TEXT_SAMPLE_LEFT, CANVAS_SIZE - 260, 700, CANVAS_SIZE - 40))
    distinct = len(set(text_area.convert("L").getdata()))
    if distinct < 8:
        problems.append("text block area looks blank - price/location may not have rendered")

    return problems


def _inspect_caption(payload: dict, case: Case) -> list[str]:
    """Check the caption side of the response for the ways it can come back broken."""
    problems: list[str] = []

    caption = payload.get("caption")
    if not isinstance(caption, str) or not caption.strip():
        problems.append("caption is empty")
    elif "�" in caption:
        problems.append("caption contains U+FFFD replacement characters (mangled unicode)")

    hashtags = payload.get("hashtags")
    if not isinstance(hashtags, list) or not hashtags:
        problems.append("hashtags missing or empty")
    elif any(not isinstance(t, str) or not t.strip() for t in hashtags):
        problems.append("hashtags contains an empty entry")

    cta = payload.get("cta")
    if not isinstance(cta, str) or not cta.strip():
        problems.append("cta is empty")

    if not payload.get("poster_base64"):
        problems.append("poster_base64 missing")

    expected_template = "Template A" if case.usable_count < 3 else "Template B"
    if payload.get("template_id") != expected_template:
        problems.append(
            f"template_id is {payload.get('template_id')!r}, expected {expected_template!r} "
            f"for {case.usable_count} usable photo(s)"
        )

    return problems


def run_case(client: httpx.Client, case: Case, fixtures: dict[str, Path]) -> Result:
    result = Result(name=case.name, group=case.group, expect=case.expect)

    start = time.perf_counter()
    try:
        response = client.post(
            "/create-post",
            files=_files_for(case, fixtures),
            data={
                "price": case.price,
                "location": case.location,
                "bedrooms": case.bedrooms,
                "bathrooms": case.bathrooms,
            },
        )
    except httpx.TimeoutException:
        result.seconds = time.perf_counter() - start
        result.outcome = "TIMEOUT"
        result.detail = f"no response within {REQUEST_TIMEOUT:.0f}s"
        result.findings.append("request hung instead of answering")
        return result
    except httpx.HTTPError as exc:
        result.seconds = time.perf_counter() - start
        result.outcome = "CRASH"
        result.detail = f"{type(exc).__name__}: {exc}"
        return result

    result.seconds = time.perf_counter() - start
    result.status = response.status_code

    # 5xx is never an acceptable answer to a malformed request: either the input
    # is valid (200) or it is the caller's fault (4xx).
    if response.status_code >= 500:
        # A 502 is the endpoint's documented "upstream failed" path (Anthropic
        # down / rate limited). Real, but not a defect in this code, so it's
        # called out separately rather than counted as a crash.
        body = _detail_of(response)
        if response.status_code == 502:
            result.outcome = "UPSTREAM"
            result.detail = body
            return result
        result.outcome = "CRASH"
        result.detail = f"HTTP {response.status_code}: {body}"
        result.findings.append(f"server error {response.status_code} - should be a clean 4xx")
        return result

    if 400 <= response.status_code < 500:
        result.detail = _detail_of(response)
        if case.expect == "4xx":
            result.outcome = "PASS"
        else:
            result.outcome = "FAIL"
            result.findings.append(f"rejected valid input: {result.detail}")
        return result

    # --- 200 --------------------------------------------------------------
    try:
        payload = response.json()
    except ValueError:
        result.outcome = "CRASH"
        result.detail = "200 response was not JSON"
        result.findings.append("200 with a non-JSON body")
        return result

    if case.expect == "4xx":
        result.outcome = "FAIL"
        result.detail = "accepted input that should have been rejected"
        result.findings.append(result.detail)

    problems = _inspect_caption(payload, case)
    poster_b64 = payload.get("poster_base64")
    if isinstance(poster_b64, str) and poster_b64:
        png = base64.b64decode(poster_b64)
        POSTER_DIR.mkdir(parents=True, exist_ok=True)
        (POSTER_DIR / f"{case.name}.png").write_bytes(png)
        problems.extend(_inspect_poster(png))

    if problems:
        result.findings.extend(problems)
        if result.outcome != "FAIL":
            result.outcome = "FAIL"
        result.detail = "; ".join(problems)
    elif not result.outcome:
        result.outcome = "PASS"
        result.detail = (
            f"{payload.get('template_id')} / style={payload.get('style_tag')} / "
            f"trend={'yes' if payload.get('trend_used') else 'no'}"
        )

    return result


def _detail_of(response: httpx.Response) -> str:
    """The `detail` field of an error body, or a trimmed raw body."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200].replace("\n", " ")
    if isinstance(body, dict) and "detail" in body:
        return str(body["detail"])[:300]
    return str(body)[:200]


# ---------------------------------------------------------------------------
# PHASE 3 - concurrency
# ---------------------------------------------------------------------------

# Five listings whose photos are each a distinct flat colour, so we can decode
# every returned poster and prove it was built from ITS OWN request's uploads.
CONCURRENT_JOBS = [
    ("job_a", (200, 40, 40), 1, "RM 450,000", "Mont Kiara, Kuala Lumpur", "3", "2"),
    ("job_b", (40, 160, 60), 2, "RM 1,250,000", "Bangsar South, Kuala Lumpur", "4", "3"),
    ("job_c", (40, 70, 200), 3, "RM 780,000", "Puchong, Selangor", "3", "2"),
    ("job_d", (220, 170, 30), 4, "RM 2,100,000", "Damansara Heights, Kuala Lumpur", "5", "4"),
    ("job_e", (150, 40, 180), 7, "RM 620,000", "Cyberjaya, Selangor", "2", "2"),
]


def _solid_photo_bytes(colour: tuple[int, int, int], index: int) -> bytes:
    buffer = io.BytesIO()
    _numbered_photo((1400, 1050), colour, str(index)).save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def _dominant_colour(png_bytes: bytes) -> tuple[int, int, int]:
    """The most common colour in the poster's photo area."""
    poster = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    sample = poster.crop((0, 0, CANVAS_SIZE, 500)).resize((40, 20))
    counts: dict[tuple[int, int, int], int] = {}
    for pixel in sample.getdata():
        counts[pixel] = counts.get(pixel, 0) + 1
    return max(counts, key=counts.get)


async def _fire_one(client: httpx.AsyncClient, job) -> dict:
    name, colour, count, price, location, beds, baths = job
    files = [
        ("photos", (f"{name}_{i}.jpg", _solid_photo_bytes(colour, i), "image/jpeg"))
        for i in range(1, count + 1)
    ]
    start = time.perf_counter()
    response = await client.post(
        "/create-post",
        files=files,
        data={"price": price, "location": location, "bedrooms": beds, "bathrooms": baths},
    )
    elapsed = time.perf_counter() - start
    return {"name": name, "colour": colour, "count": count, "status": response.status_code,
            "seconds": elapsed, "body": response}


async def _run_concurrency(base_url: str) -> list[dict]:
    async with httpx.AsyncClient(base_url=base_url, timeout=REQUEST_TIMEOUT) as client:
        return await asyncio.gather(*(_fire_one(client, job) for job in CONCURRENT_JOBS))


def concurrency_check(base_url: str) -> list[str]:
    """
    Fire all five listings at once and prove the responses didn't cross wires.

    Two things are checked. First that the poster we got back was built from the
    photos WE uploaded - each job's photos are a unique flat colour, so decoding
    the poster and reading its dominant colour is a direct test of that. Second
    that the echoed template_id matches our own photo count. Either one coming
    back as another job's value would mean shared state.

    Also compares the uploads/ scratch folder before and after: one request
    cleaning up while another is mid-flight would show up as a leaked or missing
    job directory.
    """
    uploads = Path(__file__).resolve().parent / "uploads"
    before = set(uploads.glob("job_*")) if uploads.exists() else set()

    print(f"\nFiring {len(CONCURRENT_JOBS)} /create-post requests concurrently...")
    wall_start = time.perf_counter()
    responses = asyncio.run(_run_concurrency(base_url))
    wall = time.perf_counter() - wall_start

    after = set(uploads.glob("job_*")) if uploads.exists() else set()

    problems: list[str] = []
    slowest = 0.0

    for item in responses:
        name, colour, count = item["name"], item["colour"], item["count"]
        response, status = item["body"], item["status"]
        slowest = max(slowest, item["seconds"])

        if status != 200:
            problems.append(f"{name}: HTTP {status} - {_detail_of(response)}")
            print(f"  {name}  HTTP {status:<4} {item['seconds']:6.2f}s  FAILED")
            continue

        payload = response.json()
        png = base64.b64decode(payload["poster_base64"])
        POSTER_DIR.mkdir(parents=True, exist_ok=True)
        (POSTER_DIR / f"concurrent_{name}.png").write_bytes(png)

        dominant = _dominant_colour(png)
        drift = max(abs(dominant[i] - colour[i]) for i in range(3))
        expected_template = "Template A" if count < 3 else "Template B"
        actual_template = payload.get("template_id")

        colour_ok = drift <= 12
        template_ok = actual_template == expected_template

        if not colour_ok:
            problems.append(
                f"{name}: poster's dominant colour {dominant} is not this job's "
                f"upload colour {colour} - responses may be crossed"
            )
        if not template_ok:
            problems.append(
                f"{name}: template_id {actual_template!r}, expected {expected_template!r} "
                f"for {count} photos"
            )

        flag = "OK " if colour_ok and template_ok else "MISMATCH"
        print(
            f"  {name}  HTTP {status}  {item['seconds']:6.2f}s  {count} photo(s)  "
            f"{actual_template}  colour={dominant} {flag}"
        )

    leaked = after - before
    if leaked:
        problems.append(f"{len(leaked)} job folder(s) left behind in uploads/: {sorted(leaked)}")

    print(f"\n  wall clock for all {len(CONCURRENT_JOBS)}: {wall:.2f}s (slowest single: {slowest:.2f}s)")
    if wall > 0 and slowest > 0:
        print(f"  overlap factor: {sum(r['seconds'] for r in responses) / wall:.2f}x "
              f"(1.0 = fully serialised, {len(CONCURRENT_JOBS)}.0 = fully parallel)")
    print(f"  uploads/ job folders leaked: {len(leaked)}")

    return problems


# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------


def print_table(results: list[Result]) -> None:
    print()
    print(f"{'CASE':<24} {'GROUP':<12} {'EXPECT':<7} {'STATUS':<7} {'TIME':>7}  RESULT")
    print("-" * 110)
    for r in results:
        status = str(r.status) if r.status is not None else "-"
        print(
            f"{r.name:<24} {r.group:<12} {r.expect:<7} {status:<7} {r.seconds:>6.2f}s  "
            f"{r.outcome:<8} {r.detail[:60]}"
        )


def print_summary(results: list[Result], concurrency_problems: list[str] | None) -> int:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.outcome] = counts.get(r.outcome, 0) + 1

    print("\n" + "=" * 110)
    print(f"{len(results)} cases run: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    bad = [r for r in results if r.outcome in ("FAIL", "CRASH", "TIMEOUT")]
    if bad:
        print(f"\n{len(bad)} case(s) need attention:\n")
        for r in bad:
            print(f"  [{r.outcome}] {r.name} (HTTP {r.status})")
            for finding in r.findings or [r.detail]:
                print(f"      - {finding}")

    upstream = [r for r in results if r.outcome == "UPSTREAM"]
    if upstream:
        print(f"\n{len(upstream)} case(s) hit an upstream (Anthropic) failure - not a code defect:")
        for r in upstream:
            print(f"  {r.name}: {r.detail[:120]}")

    if concurrency_problems is not None:
        if concurrency_problems:
            print(f"\nConcurrency: {len(concurrency_problems)} problem(s)")
            for problem in concurrency_problems:
                print(f"  - {problem}")
        else:
            print("\nConcurrency: all 5 concurrent requests stayed isolated - each poster was "
                  "built from its own upload and no job folders leaked.")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "name": r.name, "group": r.group, "expect": r.expect,
                        "status": r.status, "seconds": round(r.seconds, 3),
                        "outcome": r.outcome, "detail": r.detail, "findings": r.findings,
                    }
                    for r in results
                ],
                "concurrency_problems": concurrency_problems,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nFull results: {RESULTS_PATH}")
    print(f"Posters:      {POSTER_DIR}")

    return 1 if bad or concurrency_problems else 0


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--matrix-only", action="store_true")
    parser.add_argument("--concurrency-only", action="store_true")
    parser.add_argument("--only", help="run only cases whose name contains this substring")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        httpx.get(f"{args.base_url}/", timeout=5.0).raise_for_status()
    except httpx.HTTPError as exc:
        print(f"Cannot reach the server at {args.base_url} ({exc}).")
        print("Start it first:  uvicorn main:app --host 127.0.0.1 --port 8000")
        return 2

    print(f"Server: {args.base_url}")
    print("Building fixtures (this makes a couple of very large images, give it a moment)...")
    fixtures = build_fixtures()
    print(f"  {len(fixtures)} fixtures in {PHOTO_DIR}")

    results: list[Result] = []
    concurrency_problems: list[str] | None = None

    if not args.concurrency_only:
        cases = build_matrix()
        if args.only:
            cases = [c for c in cases if args.only in c.name]
        print(f"\nRunning {len(cases)} matrix cases against /create-post...\n")
        with httpx.Client(base_url=args.base_url, timeout=REQUEST_TIMEOUT) as client:
            for case in cases:
                result = run_case(client, case, fixtures)
                results.append(result)
                print(
                    f"  {result.outcome:<8} {case.name:<24} HTTP "
                    f"{str(result.status or '-'):<5} {result.seconds:6.2f}s"
                )
        print_table(results)

    if not args.matrix_only:
        concurrency_problems = concurrency_check(args.base_url)

    return print_summary(results, concurrency_problems)


if __name__ == "__main__":
    raise SystemExit(main())
