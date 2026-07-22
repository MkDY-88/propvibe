"""
PropVibe - end-to-end test for the poster generator.

Run it from the repo root:

    python test_poster.py

It creates fake property photos (plain coloured rectangles with a big number on
them so you can tell which slot each one landed in), feeds them through
`generate_poster()`, and writes the results to test_output/ for you to eyeball.

Four posters are produced, one per interesting case:

    1_template_a_single.png   1 photo   -> Template A, full-bleed hero
    2_template_a_two.png      2 photos  -> Template A (still under the 3 threshold)
    3_template_b_three.png    3 photos  -> Template B, hero + two stacked
    4_template_b_four.png     4 photos  -> Template B, 2x2 grid
    5_template_b_badge.png    7 photos  -> Template B, 2x2 grid + "+3 more" badge

No real photos required - everything is generated on the fly.
"""

from pathlib import Path

from PIL import Image, ImageDraw

from app.poster_generator import (
    FONT_BOLD_PATH,
    _load_font,
    fonts_are_available,
    generate_poster,
)

# Where everything goes
OUTPUT_DIR = Path("test_output")
SAMPLE_DIR = OUTPUT_DIR / "sample_photos"

# Stand-in "property photos". Deliberately different sizes and aspect ratios so
# we can prove the crop-to-fill logic never squashes an image.
SAMPLE_SPECS = [
    ("photo1.jpg", (1600, 900), (46, 92, 138)),  # wide  - blue
    ("photo2.jpg", (900, 1600), (138, 84, 46)),  # tall  - brown
    ("photo3.jpg", (1200, 1200), (58, 110, 74)),  # square- green
    ("photo4.jpg", (1400, 1050), (120, 52, 92)),  # 4:3   - plum
    ("photo5.jpg", (1000, 1000), (140, 118, 40)),  # square- olive
    ("photo6.jpg", (1600, 1000), (60, 60, 90)),  # wide  - slate
    ("photo7.jpg", (1100, 1400), (100, 70, 70)),  # tall  - rose
]


def make_sample_photos() -> list[str]:
    """Create the placeholder images on disk and return their paths in order."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    for index, (name, size, colour) in enumerate(SAMPLE_SPECS, start=1):
        path = SAMPLE_DIR / name
        img = Image.new("RGB", size, colour)
        draw = ImageDraw.Draw(img)

        # A big number in the middle so the grid ordering is obvious at a glance.
        font = _load_font(FONT_BOLD_PATH, 220)
        draw.text(
            (size[0] // 2, size[1] // 2), str(index), font=font, fill=(255, 255, 255), anchor="mm"
        )

        img.save(path, format="JPEG", quality=90)
        paths.append(str(path))

    return paths


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if fonts_are_available():
        print("Fonts: using Poppins from assets/fonts/")
    else:
        print("Fonts: Poppins NOT found - falling back to Pillow's built-in font")

    photos = make_sample_photos()
    print(f"Created {len(photos)} sample photos in {SAMPLE_DIR}\n")

    # Each case: (filename, how many photos, listing details)
    cases = [
        ("1_template_a_single.png", 1, "RM 450,000", "Mont Kiara, Kuala Lumpur", 3, 2),
        ("2_template_a_two.png", 2, "RM 1,250,000", "Bangsar South, Kuala Lumpur", 4, 3),
        ("3_template_b_three.png", 3, "RM 780,000", "Puchong, Selangor", 3, 2),
        ("4_template_b_four.png", 4, "RM 2,100,000", "Damansara Heights, Kuala Lumpur", 5, 4),
        ("5_template_b_badge.png", 7, "RM 620,000", "Cyberjaya, Selangor", 2, 2),
    ]

    for filename, count, price, location, beds, baths in cases:
        result = generate_poster(
            photos=photos[:count],
            price=price,
            location=location,
            bedrooms=beds,
            bathrooms=baths,
            output_path=str(OUTPUT_DIR / filename),
        )
        template = "A (full-bleed)" if count < 3 else "B (grid)"
        print(f"  {count} photo(s) -> Template {template:<16} {result}")

    print(f"\nDone. Open the {OUTPUT_DIR}/ folder to check the posters.")


if __name__ == "__main__":
    main()
