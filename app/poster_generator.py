"""
PropVibe - Poster Generator
===========================

Turns a set of property photos + listing details into a ready-to-post
1080x1080 social media poster (Instagram / Facebook square format).

HOW IT WORKS (the 30-second version for judges):

    1. We create a blank 1080x1080 navy canvas.
    2. We look at how many photos the agent uploaded and pick a layout:
         - fewer than 3 photos  -> TEMPLATE A: one big full-bleed hero photo
         - 3 or more photos     -> TEMPLATE B: a photo grid + solid navy band
    3. We paste the photos in, cropping them to fill their slot so nothing
       ever looks squashed or stretched.
    4. We draw the price / location / bed-bath text in the bottom-left.
    5. We save the result as a PNG and return the file path.

The only public function you need is `generate_poster()` at the bottom.

Requires: Pillow  (pip install Pillow)
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

logger = logging.getLogger("propvibe.poster_generator")

# ---------------------------------------------------------------------------
# DESIGN CONSTANTS
# Everything visual is defined here so the brand look can be tweaked in one
# place without digging through the drawing code.
# ---------------------------------------------------------------------------

CANVAS_SIZE = 1080  # Square poster: 1080 x 1080 px

# Brand colours
NAVY = (26, 43, 76)  # #1A2B4C - default band/background + price accent base
GOLD = (212, 175, 55)  # #D4AF37 - default price accent
WHITE = (255, 255, 255)  # location
LIGHT_GRAY = (200, 205, 215)  # bed / bath line
PLACEHOLDER_GRAY = (60, 68, 84)  # shown if a photo fails to load

# Template A: the dark fade sits over the bottom 350px of the hero photo
GRADIENT_HEIGHT = 350

# ---------------------------------------------------------------------------
# STYLE PALETTES
# Each caption style (see app.copy_generator.STYLE_TAGS) gets its own band
# colour, accent colour (price text + the "+N more" pill), and Template A fade
# shape, so the poster's visual look actually varies with style and not just
# the caption's wording. Keyed by *lowercase* style tag.
#
# style_tag=None or an unrecognised style falls back to DEFAULT_PALETTE (the
# original navy/gold look), so callers that don't care about style at all -
# /generate-poster, test_poster.py - keep their exact existing appearance.
# ---------------------------------------------------------------------------

DEFAULT_PALETTE = {
    "band": NAVY,
    "accent": GOLD,
    "gradient_height": GRADIENT_HEIGHT,
    "gradient_exponent": 2,  # eases in slowly, ramps up near the bottom
}

STYLE_PALETTES = {
    "modern": {
        "band": (17, 24, 39),  # near-black cool slate
        "accent": (196, 200, 209),  # cool platinum/silver
        "gradient_height": 300,  # tighter, cleaner fade
        "gradient_exponent": 2,
    },
    "warm": {
        "band": (74, 48, 34),  # warm espresso brown
        "accent": (224, 160, 90),  # copper/amber
        "gradient_height": 400,  # longer, softer fade
        "gradient_exponent": 2,
    },
    "bold": {
        "band": (15, 15, 20),  # near-black, max contrast
        "accent": (255, 87, 51),  # vivid red-orange
        "gradient_height": 300,  # short, punchy fade
        "gradient_exponent": 1,  # linear - darkens sooner, higher contrast
    },
}


def _palette_for(style_tag: str | None) -> dict:
    """Look up the palette for a style tag, falling back to DEFAULT_PALETTE."""
    if isinstance(style_tag, str) and style_tag.strip():
        palette = STYLE_PALETTES.get(style_tag.strip().lower())
        if palette is not None:
            return palette
    return DEFAULT_PALETTE

# Template B: photos live in the top 700px, text band is the bottom 380px
GRID_HEIGHT = 700
BAND_HEIGHT = CANVAS_SIZE - GRID_HEIGHT  # = 380
GRID_GAP = 8  # thin navy gutter between grid cells

# Text block placement (bottom-left corner of the poster)
MARGIN_X = 60  # distance from the left edge
MARGIN_BOTTOM = 70  # distance from the bottom edge

# Font sizes
PRICE_SIZE = 64
LOCATION_SIZE = 28
DETAILS_SIZE = 24

# Vertical breathing room between the three text lines
GAP_AFTER_PRICE = 16
GAP_AFTER_LOCATION = 10

# The photo-count threshold that decides which template we use
GRID_TEMPLATE_MIN_PHOTOS = 3
MAX_GRID_PHOTOS = 4  # a 2x2 grid holds 4; extras become a "+N more" badge

# Fonts live at <repo root>/assets/fonts/. We resolve the path relative to THIS
# file (not the working directory) so it also works once deployed on Railway.
FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
FONT_BOLD_PATH = FONT_DIR / "Poppins-Bold.ttf"
FONT_REGULAR_PATH = FONT_DIR / "Poppins-Regular.ttf"


# ---------------------------------------------------------------------------
# FONT LOADING
# ---------------------------------------------------------------------------


def _load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    """
    Load a TrueType font at the given size.

    FALLBACK NOTE: the Poppins .ttf files are committed under assets/fonts/, so
    the normal path is that this just works. If those files are ever missing
    (e.g. someone clones without Git LFS, or a slim Docker image strips them),
    we fall back to Pillow's built-in font instead of crashing. The poster will
    still generate correctly, it will just not be on-brand typographically.
    """
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        try:
            # Pillow >= 10.1 can scale its bundled default font.
            return ImageFont.load_default(size=size)
        except TypeError:
            # Very old Pillow: only a tiny fixed-size bitmap font is available.
            return ImageFont.load_default()


def fonts_are_available() -> bool:
    """True if the real Poppins files are on disk (handy for a health check)."""
    return FONT_BOLD_PATH.exists() and FONT_REGULAR_PATH.exists()


# ---------------------------------------------------------------------------
# NON-LATIN FALLBACK
#
# Poppins covers Latin and common punctuation and nothing else. A Chinese,
# Tamil or Jawi location - all perfectly normal in Malaysian listings - renders
# as a row of .notdef tofu boxes. When a line contains characters Poppins has no
# glyph for, we look for a system font that does.
#
# Nothing is downloaded and no dependency is added: these are the usual install
# locations on Windows, Linux and macOS. If none of them exist (a bare container
# with no fonts installed) we log it once and fall back to Poppins, i.e. exactly
# the old behaviour - tofu, but never a crash.
# ---------------------------------------------------------------------------

FALLBACK_FONT_PATHS = (
    # Windows
    "C:/Windows/Fonts/msyh.ttc",  # Microsoft YaHei - CJK + Latin
    "C:/Windows/Fonts/NotoSansSC-VF.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/seguisym.ttf",
    "C:/Windows/Fonts/arial.ttf",
    # Linux (Railway / Debian / Alpine images that ship fonts)
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
)

# A Private Use Area codepoint. No real font assigns a glyph to it, so whatever
# a font draws for it IS that font's .notdef - the placeholder box. Comparing a
# character's rendered bitmap against this tells us whether the font genuinely
# covers that character, without parsing the font's cmap table or pulling in
# fontTools just to ask.
_NOTDEF_PROBE = "\ue000"

# Fallback lookups are memoised per (path, size): resolving them means loading
# and rasterising, and _draw_text_block runs this for every line of every poster.
_FALLBACK_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont | None] = {}


def _covers(font, text: str) -> bool:
    """
    True if `font` has a real glyph for every character in `text`.

    Renders the .notdef probe once, then compares each character's bitmap
    against it. Whitespace is skipped - it legitimately rasterises to nothing.
    """
    try:
        notdef = bytes(font.getmask(_NOTDEF_PROBE))
    except Exception:  # noqa: BLE001 - a bitmap fallback font can't be probed
        return True

    for char in set(text):
        if char.isspace():
            continue
        try:
            if bytes(font.getmask(char)) == notdef:
                return False
        except Exception:  # noqa: BLE001 - unrenderable is its own answer
            return False
    return True


def _fallback_font(size: int, text: str):
    """The first system font at `size` that covers `text`, or None."""
    for path in FALLBACK_FONT_PATHS:
        if not Path(path).exists():
            continue

        cached = (path, size)
        if cached not in _FALLBACK_CACHE:
            try:
                _FALLBACK_CACHE[cached] = ImageFont.truetype(path, size)
            except OSError:
                _FALLBACK_CACHE[cached] = None

        font = _FALLBACK_CACHE[cached]
        if font is not None and _covers(font, text):
            return font

    return None


def _font_for(path: Path, size: int, text: str):
    """
    The brand font at `size`, or a system fallback when it can't render `text`.

    Latin text - which is almost everything - takes the fast path and gets
    Poppins exactly as before. The fallback is only consulted for characters
    Poppins genuinely lacks, and it is a regular weight regardless of which
    Poppins we started from: real glyphs in the wrong weight beat correct-weight
    boxes.
    """
    font = _load_font(path, size)
    if _covers(font, text):
        return font

    fallback = _fallback_font(size, text)
    if fallback is not None:
        return fallback

    logger.warning(
        "No installed font covers %r - it will render as placeholder boxes. "
        "Install a broad-coverage font (e.g. Noto Sans CJK) on this host.",
        text[:40],
    )
    return font


# ---------------------------------------------------------------------------
# IMAGE HELPERS
# ---------------------------------------------------------------------------


def _open_photo(path: str) -> tuple[Image.Image | None, str | None]:
    """
    Open a photo safely, reporting *why* if it can't be used.

    Returns ``(image, None)`` on success or ``(None, reason)`` on failure, where
    `reason` is a short phrase that reads correctly after "Photo 3 ...".

    The decompression-bomb case is called out separately because it is NOT a
    corrupt file: a 15000x15000 PNG of flat colour is under a megabyte on disk
    (so it sails past the upload size guard) but decodes to 225 megapixels, and
    Pillow refuses it with DecompressionBombError. That exception derives from
    Exception, not OSError, so it used to escape this function entirely and
    surface as a 500.
    """
    try:
        img = Image.open(path)
        # Phone photos carry an EXIF "rotate me" flag. Applying it here stops
        # portrait shots from coming out sideways.
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB"), None
    except Image.DecompressionBombError:
        return None, "is too large to process (too many pixels once decoded)"
    except (OSError, ValueError):
        return None, "could not be read as an image (corrupt, truncated, or not an image file)"


def _load_photo(path: str) -> Image.Image | None:
    """
    Open a photo safely.

    Returns None if the file is missing or unreadable, so one bad upload can
    never take down the whole poster job.
    """
    return _open_photo(path)[0]


def _crop_to_fill(img: Image.Image, width: int, height: int) -> Image.Image:
    """
    Resize a photo to exactly fill a width x height slot WITHOUT distorting it.

    This is the same behaviour as CSS `object-fit: cover`: we centre-crop the
    overflowing side, then scale. A wide photo loses some left/right, a tall
    photo loses some top/bottom, but circles stay circles.
    """
    source_ratio = img.width / img.height
    target_ratio = width / height

    if source_ratio > target_ratio:
        # Photo is too wide -> keep full height, trim the sides.
        crop_w = round(img.height * target_ratio)
        left = (img.width - crop_w) // 2
        box = (left, 0, left + crop_w, img.height)
    else:
        # Photo is too tall -> keep full width, trim top and bottom.
        crop_h = round(img.width / target_ratio)
        top = (img.height - crop_h) // 2
        box = (0, top, img.width, top + crop_h)

    # Pillow can crop and resize in one pass by passing `box`.
    return img.resize((width, height), Image.LANCZOS, box=box)


def _cell_image(path: str, width: int, height: int) -> Image.Image:
    """A cropped photo for one slot, or a flat grey block if it failed to load."""
    photo = _load_photo(path)
    if photo is None:
        return Image.new("RGB", (width, height), PLACEHOLDER_GRAY)
    return _crop_to_fill(photo, width, height)


def check_photos(paths: list[str]) -> tuple[list[str], list[str]]:
    """
    Split photo paths into the ones we can actually use and the ones we can't.

    Returns ``(usable, problems)``: `usable` is the subset Pillow decodes
    successfully, in the original order, and `problems` holds one
    ready-to-display sentence per rejected photo ("Photo 2 could not be read as
    an image ...").

    Callers are expected to run this BEFORE generate_poster() and drop the
    rejects. The grey PLACEHOLDER_GRAY block in _cell_image() is a last-resort
    safety net, not an acceptable result: a poster with a dead grey rectangle
    where a bedroom should be is not something anyone would publish, and when
    every photo fails it produces a poster that is nothing but grey.

    Note this decodes each photo once here and again during layout. That is a
    deliberate trade: for the endpoints it also means an all-unusable upload is
    rejected before we spend a web search and a caption call on it.
    """
    usable: list[str] = []
    problems: list[str] = []

    for index, path in enumerate(paths, start=1):
        image, reason = _open_photo(path)
        if image is None:
            problems.append(f"Photo {index} {reason}.")
        else:
            image.close()
            usable.append(path)

    return usable, problems


# ---------------------------------------------------------------------------
# TEMPLATE A - single full-bleed photo with a gradient fade
# ---------------------------------------------------------------------------


def _draw_gradient_overlay(canvas: Image.Image, palette: dict) -> None:
    """
    Fade the bottom `palette["gradient_height"]` pixels from transparent into
    the style's band colour.

    Why: the hero photo could be a bright white kitchen or a dark night shot.
    The fade guarantees the text underneath is readable either way.

    How: we build a 1-pixel-wide greyscale ramp (0 = transparent, 255 = opaque),
    stretch it to full width, and use it as the mask when pasting a colour
    block. `gradient_exponent` shapes the ramp: 2 (most styles) eases in
    slowly for a soft, natural fade; 1 (Bold) is linear, darkening sooner for a
    punchier, higher-contrast look.
    """
    height = palette["gradient_height"]
    exponent = palette["gradient_exponent"]

    ramp = Image.new("L", (1, height))
    for y in range(height):
        progress = y / (height - 1)
        ramp.putpixel((0, y), int(255 * (progress**exponent)))

    mask = ramp.resize((CANVAS_SIZE, height))
    band_block = Image.new("RGB", (CANVAS_SIZE, height), palette["band"])
    canvas.paste(band_block, (0, CANVAS_SIZE - height), mask)


def _build_template_a(canvas: Image.Image, photos: list[str], palette: dict) -> None:
    """One photo, edge to edge, with the dark gradient over the bottom."""
    hero = _cell_image(photos[0], CANVAS_SIZE, CANVAS_SIZE)
    canvas.paste(hero, (0, 0))
    _draw_gradient_overlay(canvas, palette)


# ---------------------------------------------------------------------------
# TEMPLATE B - photo grid above a solid navy band
# ---------------------------------------------------------------------------


def _grid_cells(photo_count: int) -> list[tuple[int, int, int, int]]:
    """
    Work out the (x, y, width, height) of each photo slot in the top 700px.

    Two arrangements, both capped at 4 photos:

      3 photos              4 photos
      +--------+----+       +----+----+
      |        |    |       |    |    |
      |  hero  +----+       +----+----+
      |        |    |       |    |    |
      +--------+----+       +----+----+

    The 3-photo "hero + two stacked" layout exists because a 2x2 grid with one
    empty square looks like a bug to a viewer.
    """
    half_w = (CANVAS_SIZE - GRID_GAP) // 2
    half_h = (GRID_HEIGHT - GRID_GAP) // 2
    right_x = half_w + GRID_GAP
    lower_y = half_h + GRID_GAP

    if photo_count == 3:
        return [
            (0, 0, half_w, GRID_HEIGHT),  # tall hero on the left
            (right_x, 0, half_w, half_h),  # top right
            (right_x, lower_y, half_w, half_h),  # bottom right
        ]

    # 4 or more -> a clean 2x2.
    return [
        (0, 0, half_w, half_h),
        (right_x, 0, half_w, half_h),
        (0, lower_y, half_w, half_h),
        (right_x, lower_y, half_w, half_h),
    ]


def _draw_more_badge(
    canvas: Image.Image, cell: tuple[int, int, int, int], extra: int, palette: dict
) -> None:
    """
    Stamp a "+N more" pill into the bottom-right corner of the final grid cell.

    Drawn on its own transparent layer so the pill can be semi-see-through and
    still let the photo underneath show through.
    """
    x, y, w, h = cell
    font = _load_font(FONT_BOLD_PATH, 26)
    label = f"+{extra} more"

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Measure the text so the pill hugs it regardless of how many digits N has.
    text_box = draw.textbbox((0, 0), label, font=font)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]

    pad_x, pad_y, inset = 20, 12, 18
    pill_w = text_w + pad_x * 2
    pill_h = text_h + pad_y * 2
    pill_x = x + w - inset - pill_w
    pill_y = y + h - inset - pill_h

    draw.rounded_rectangle(
        (pill_x, pill_y, pill_x + pill_w, pill_y + pill_h),
        radius=pill_h // 2,
        fill=palette["band"] + (225,),  # band colour at ~88% opacity
    )
    draw.text(
        (pill_x + pill_w // 2, pill_y + pill_h // 2),
        label,
        font=font,
        fill=palette["accent"],
        anchor="mm",
    )

    # Paste using the overlay's own alpha channel as the mask.
    canvas.paste(overlay, (0, 0), overlay)


def _build_template_b(canvas: Image.Image, photos: list[str], palette: dict) -> None:
    """Up to 4 photos in a grid; the bottom 380px stays the style's band colour."""
    shown = photos[:MAX_GRID_PHOTOS]
    cells = _grid_cells(len(shown))

    for path, (x, y, w, h) in zip(shown, cells):
        canvas.paste(_cell_image(path, w, h), (x, y))

    # The canvas starts life filled with the style's band colour, so the bottom
    # band and the thin gutters between cells are already the right colour -
    # nothing to draw.

    extra = len(photos) - MAX_GRID_PHOTOS
    if extra > 0:
        _draw_more_badge(canvas, cells[len(shown) - 1], extra, palette)


# ---------------------------------------------------------------------------
# TEXT OVERLAY (identical for both templates)
# ---------------------------------------------------------------------------


# A 1080px-wide poster fits a few dozen characters at these font sizes, so
# anything past this is guaranteed to be cut. Clipping to it before measuring
# keeps _truncate's cost flat no matter how long the input is.
TRUNCATE_SCAN_LIMIT = 400


def _truncate(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    """
    Shorten text with an ellipsis if it would run off the right edge.

    Finds the longest prefix that fits by binary search rather than by trimming
    one character at a time. The old one-at-a-time loop re-measured the whole
    remaining string on every step, so cost grew with the square of the input:
    a 3000-character location spent 5.8 seconds here, and a longer one would
    have looked to a caller exactly like a hung request.
    """
    # Measuring the whole string is itself linear in its length, so skip it once
    # we're past the point where it could conceivably fit - even 400 periods are
    # wider than the poster.
    if len(text) <= TRUNCATE_SCAN_LIMIT and draw.textlength(text, font=font) <= max_width:
        return text

    text = text[:TRUNCATE_SCAN_LIMIT]

    # Largest `size` where text[:size] + "..." still fits. Invariant: everything
    # at or below `low` fits, everything above `high` does not.
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if draw.textlength(text[:middle] + "...", font=font) <= max_width:
            low = middle
        else:
            high = middle - 1

    return text[:low].rstrip() + "..."


def _draw_text_block(
    canvas: Image.Image,
    price: str,
    location: str,
    details: str,
    palette: dict,
) -> None:
    """
    Draw price / location / details, stacked in the bottom-left corner.

    We measure the whole block first and work out where the TOP line starts, so
    the bottom line always lands exactly MARGIN_BOTTOM px above the edge no
    matter how tall the individual lines turn out to be.

    `details` is the pre-formatted third line (e.g. "3 Bed · 2 Bath", or a
    room-rental listing's "Master, Queen bed, Private bathroom") - callers
    decide its wording; this function just lays it out.
    """
    draw = ImageDraw.Draw(canvas)
    max_width = CANVAS_SIZE - (MARGIN_X * 2)

    # (text, font, colour, gap below this line). _font_for keeps Poppins for
    # Latin text and swaps in a system font only for a line Poppins can't render
    # - a Chinese or Tamil location, which is ordinary in Malaysian listings.
    lines = [
        (price, _font_for(FONT_BOLD_PATH, PRICE_SIZE, price), palette["accent"], GAP_AFTER_PRICE),
        (
            location,
            _font_for(FONT_REGULAR_PATH, LOCATION_SIZE, location),
            WHITE,
            GAP_AFTER_LOCATION,
        ),
        (details, _font_for(FONT_REGULAR_PATH, DETAILS_SIZE, details), LIGHT_GRAY, 0),
    ]

    # Measure every line ("la" anchor = left edge, top of the ascender).
    measured = []
    for text, font, colour, gap in lines:
        text = _truncate(draw, text, font, max_width)
        box = draw.textbbox((0, 0), text, font=font, anchor="la")
        measured.append((text, font, colour, gap, box[3] - box[1]))

    total_height = sum(height + gap for _, _, _, gap, height in measured)
    y = CANVAS_SIZE - MARGIN_BOTTOM - total_height

    for text, font, colour, gap, height in measured:
        draw.text((MARGIN_X, y), text, font=font, fill=colour, anchor="la")
        y += height + gap


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------


def generate_poster(
    photos: list[str],
    price: str,
    location: str,
    bedrooms: int,
    bathrooms: int,
    output_path: str,
    style_tag: str | None = None,
    details: str | None = None,
) -> str:
    """
    Generate a 1080x1080 property poster and save it as a PNG.

    Args:
        photos:      Local file paths to the property images, best photo first.
                     1-2 photos -> Template A (full-bleed hero).
                     3+ photos  -> Template B (grid + band).
        price:       Pre-formatted price string, e.g. "RM 450,000".
        location:    Short location line, e.g. "Mont Kiara, Kuala Lumpur".
        bedrooms:    Number of bedrooms. Ignored if `details` is given.
        bathrooms:   Number of bathrooms. Ignored if `details` is given.
        output_path: Where to write the PNG. Parent folders are created.
        style_tag:   Optional caption style ("Modern"/"Warm"/"Bold", see
            app.copy_generator.STYLE_TAGS) selecting the poster's colour
            palette via STYLE_PALETTES. None or an unrecognised style falls
            back to DEFAULT_PALETTE (the original navy/gold look) - callers
            that don't care about style keep their exact existing appearance.
        details:     Optional pre-formatted third line, overriding the default
            "{bedrooms} Bed · {bathrooms} Bath". For listing shapes that don't
            have bedroom/bathroom counts (e.g. a single room-rental listing),
            pass e.g. "Master, Queen bed, Private bathroom" instead. None (the
            default) keeps the original "X Bed · Y Bath" line, so existing
            callers are unaffected.

    Returns:
        The output_path that was written (same string you passed in).

    Raises:
        ValueError: if `photos` is empty.
    """
    if not photos:
        raise ValueError("generate_poster() needs at least one photo path.")

    palette = _palette_for(style_tag)

    # Start from the style's band colour. For Template B this IS the bottom
    # band and the grid gutters, so we never have to draw them explicitly.
    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), palette["band"])

    # ---- Automatic template selection -----------------------------------
    if len(photos) < GRID_TEMPLATE_MIN_PHOTOS:
        _build_template_a(canvas, photos, palette)
    else:
        _build_template_b(canvas, photos, palette)

    # ---- Shared text overlay --------------------------------------------
    details_line = details if details is not None else f"{bedrooms} Bed · {bathrooms} Bath"
    _draw_text_block(canvas, price, location, details_line, palette)

    # ---- Save -----------------------------------------------------------
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG")

    return output_path
