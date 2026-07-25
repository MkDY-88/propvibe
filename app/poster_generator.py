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

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

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
# IMAGE HELPERS
# ---------------------------------------------------------------------------


def _load_photo(path: str) -> Image.Image | None:
    """
    Open a photo safely.

    Returns None if the file is missing or unreadable, so one bad upload can
    never take down the whole poster job.
    """
    try:
        img = Image.open(path)
        # Phone photos carry an EXIF "rotate me" flag. Applying it here stops
        # portrait shots from coming out sideways.
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except (OSError, ValueError):
        return None


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


def _truncate(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    """Shorten text with an ellipsis if it would run off the right edge."""
    if draw.textlength(text, font=font) <= max_width:
        return text
    while text and draw.textlength(text + "...", font=font) > max_width:
        text = text[:-1]
    return text.rstrip() + "..."


def _draw_text_block(
    canvas: Image.Image,
    price: str,
    location: str,
    bedrooms: int,
    bathrooms: int,
    palette: dict,
) -> None:
    """
    Draw price / location / bed-bath, stacked in the bottom-left corner.

    We measure the whole block first and work out where the TOP line starts, so
    the bottom line always lands exactly MARGIN_BOTTOM px above the edge no
    matter how tall the individual lines turn out to be.
    """
    draw = ImageDraw.Draw(canvas)
    max_width = CANVAS_SIZE - (MARGIN_X * 2)

    # "3 Bed - 2 Bath", using a middle dot separator.
    details = f"{bedrooms} Bed · {bathrooms} Bath"

    # (text, font, colour, gap below this line)
    lines = [
        (price, _load_font(FONT_BOLD_PATH, PRICE_SIZE), palette["accent"], GAP_AFTER_PRICE),
        (location, _load_font(FONT_REGULAR_PATH, LOCATION_SIZE), WHITE, GAP_AFTER_LOCATION),
        (details, _load_font(FONT_REGULAR_PATH, DETAILS_SIZE), LIGHT_GRAY, 0),
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
) -> str:
    """
    Generate a 1080x1080 property poster and save it as a PNG.

    Args:
        photos:      Local file paths to the property images, best photo first.
                     1-2 photos -> Template A (full-bleed hero).
                     3+ photos  -> Template B (grid + band).
        price:       Pre-formatted price string, e.g. "RM 450,000".
        location:    Short location line, e.g. "Mont Kiara, Kuala Lumpur".
        bedrooms:    Number of bedrooms.
        bathrooms:   Number of bathrooms.
        output_path: Where to write the PNG. Parent folders are created.
        style_tag:   Optional caption style ("Modern"/"Warm"/"Bold", see
            app.copy_generator.STYLE_TAGS) selecting the poster's colour
            palette via STYLE_PALETTES. None or an unrecognised style falls
            back to DEFAULT_PALETTE (the original navy/gold look) - callers
            that don't care about style keep their exact existing appearance.

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
    _draw_text_block(canvas, price, location, bedrooms, bathrooms, palette)

    # ---- Save -----------------------------------------------------------
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG")

    return output_path
