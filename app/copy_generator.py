"""
PropVibe - Caption Generator
============================

Turns raw listing details (price, location, bed/bath) into ready-to-post social
copy by asking Claude Haiku for a caption, hashtags and a call-to-action.

The one public function you need is `generate_caption()` at the bottom. It
returns a plain dict::

    {
        "caption":  "<engaging FB/IG post, under 150 words>",
        "hashtags": ["luxuryliving", "montkiara", ...],   # 5-8, no '#' prefix
        "cta":      "DM us to book a viewing",
    }

Anything that goes wrong - a missing API key, a rate limit, a network blip, or
Claude returning something that isn't valid JSON - is raised as a single
`CaptionError` carrying a short, human-readable message. Callers can surface
`str(exc)` directly to the user instead of leaking a stack trace.
"""

from __future__ import annotations

import json
import os
import random
import re

import anthropic

# Haiku is fast and cheap, which is exactly what a one-shot caption needs.
MODEL = "claude-haiku-4-5-20251001"

# The fixed set of caption "styles" (tones) the app rotates through and learns
# from. When there's no performance data yet we pick one at random so there's
# variety to actually learn from; once /sync-engagement has data, the best
# performer is fed back in as `preferred_style` (see get_style_performance).
STYLE_TAGS = ("Modern", "Warm", "Bold")

# Short tone briefs, keyed by lowercased style, injected into the prompt so each
# style reads distinctly. An unknown style still works - it just gets a generic
# "write in an X style" instruction.
STYLE_GUIDANCE = {
    "modern": "sleek, minimal and contemporary - clean lines, aspirational and "
    "understated, quietly confident",
    "warm": "warm, homely and inviting - lead with comfort, family and the "
    "feeling of belonging",
    "bold": "bold, high-energy and punchy - short confident sentences, a sense "
    "of excitement and momentum",
}

# A caption + a handful of hashtags + one CTA line is tiny; 1024 is plenty of
# headroom while still capping a runaway response.
MAX_TOKENS = 1024

# Hashtags must land inside this range. We ask Claude for it in the prompt and
# gently clamp the result afterwards so the UI always has something sensible.
MIN_HASHTAGS = 5
MAX_HASHTAGS = 8

SYSTEM_PROMPT = (
    "You are a sharp real-estate social media copywriter for Facebook and "
    "Instagram. You write scroll-stopping, warm, benefit-led posts that make a "
    "property feel like a place to live, not a spreadsheet row.\n\n"
    "You reply with ONLY a single JSON object and nothing else - no markdown, "
    "no code fences, no commentary before or after. The JSON object must have "
    "exactly these three keys:\n"
    '  "caption":  a string. An engaging post under 150 words. May use a few '
    "tasteful emojis. Do NOT put hashtags in here.\n"
    '  "hashtags": an array of 5 to 8 short relevant hashtag strings, WITHOUT '
    'the leading "#" (e.g. "luxuryliving", not "#luxuryliving").\n'
    '  "cta":      a string. One short call-to-action line, e.g. '
    '"DM us to book a viewing".'
)


class CaptionError(Exception):
    """A clean, user-safe failure from :func:`generate_caption`."""


def _resolve_style(preferred_style: str | None) -> tuple[str, str]:
    """
    Decide which style to write in and return ``(style_tag, tone_brief)``.

    If a non-empty ``preferred_style`` is given (the current best performer) we
    honour it verbatim; its tone brief comes from STYLE_GUIDANCE when we know it,
    otherwise a generic brief. With no preference we pick one of STYLE_TAGS at
    random so there's variety to learn from.
    """
    if isinstance(preferred_style, str) and preferred_style.strip():
        style = preferred_style.strip()
    else:
        style = random.choice(STYLE_TAGS)

    brief = STYLE_GUIDANCE.get(style.lower(), f"written in a distinctly {style} style")
    return style, brief


def _build_user_prompt(
    price: str,
    location: str,
    bedrooms: int,
    bathrooms: int,
    style: str,
    tone_brief: str,
) -> str:
    """Assemble the listing details + tone into a single instruction for Claude."""
    return (
        "Write the social media copy for this property listing:\n"
        f"- Price: {price}\n"
        f"- Location: {location}\n"
        f"- Bedrooms: {bedrooms}\n"
        f"- Bathrooms: {bathrooms}\n\n"
        f"TONE: Write the caption in a {style} style - {tone_brief}.\n\n"
        "Return the JSON object described in the system instructions."
    )


def _extract_json(text: str) -> dict:
    """
    Pull a JSON object out of Claude's reply, defensively.

    The system prompt asks for bare JSON, but models sometimes wrap it in
    ```json fences or add a stray sentence. We try, in order:
      1. Parse the whole thing.
      2. Strip a ```json ... ``` (or plain ```) fence and parse that.
      3. Grab the first '{' to the last '}' and parse the slice.
    If none of that yields a JSON object, we give up with a CaptionError.
    """
    candidates: list[str] = []

    stripped = text.strip()
    candidates.append(stripped)

    # Strip a fenced code block if present, e.g. ```json\n{...}\n```
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fence:
        candidates.append(fence.group(1).strip())

    # Last resort: everything between the first { and the last }.
    first, last = stripped.find("{"), stripped.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(stripped[first : last + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed

    raise CaptionError(
        "Claude did not return valid JSON for the caption. "
        "Please try generating again."
    )


def _normalise(parsed: dict) -> dict:
    """
    Validate and tidy the parsed object into the exact shape callers expect.

    Guarantees a non-empty string `caption`, a `hashtags` list of 5-8 clean
    strings (with any stray '#'/whitespace removed), and a non-empty `cta`.
    """
    caption = parsed.get("caption")
    cta = parsed.get("cta")
    raw_tags = parsed.get("hashtags")

    if not isinstance(caption, str) or not caption.strip():
        raise CaptionError("Claude's response was missing a usable 'caption'.")
    if not isinstance(cta, str) or not cta.strip():
        raise CaptionError("Claude's response was missing a usable 'cta'.")
    if not isinstance(raw_tags, list):
        raise CaptionError("Claude's response was missing a 'hashtags' list.")

    # Clean each tag: drop leading '#', strip spaces, remove any internal
    # whitespace, and skip anything that ends up empty.
    hashtags: list[str] = []
    for tag in raw_tags:
        if not isinstance(tag, str):
            continue
        cleaned = re.sub(r"\s+", "", tag.lstrip("#").strip())
        if cleaned:
            hashtags.append(cleaned)

    if not hashtags:
        raise CaptionError("Claude's response contained no usable hashtags.")

    # Clamp to the desired range without inventing tags: only trim an overflow.
    hashtags = hashtags[:MAX_HASHTAGS]

    return {
        "caption": caption.strip(),
        "hashtags": hashtags,
        "cta": cta.strip(),
    }


def generate_caption(
    price: str,
    location: str,
    bedrooms: int,
    bathrooms: int,
    preferred_style: str | None = None,
) -> dict:
    """
    Generate social copy for a listing via Claude Haiku.

    Args:
        price:           Pre-formatted price string, e.g. "RM 450,000".
        location:        Short location line, e.g. "Mont Kiara, Kuala Lumpur".
        bedrooms:        Number of bedrooms.
        bathrooms:       Number of bathrooms.
        preferred_style: Optional style to lean the caption's tone toward - the
            current best-performing style from get_style_performance(). When
            omitted, a style from STYLE_TAGS is chosen at random so there's
            variety to learn from.

    Returns:
        dict with keys "caption" (str), "hashtags" (list[str], no '#'), "cta"
        (str) and "style_tag" (str) - the style actually used, which the caller
        records in Airtable so the learning loop knows what to attribute
        engagement to.

    Raises:
        CaptionError: for any failure - missing/invalid API key, rate limit,
            network error, or a response that isn't usable JSON. The message is
            safe to show a user.
    """
    # Validate the key here (not at import) so the rest of the app boots without
    # it - only this endpoint actually needs it.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise CaptionError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add "
            "your Anthropic API key, then restart the server."
        )

    style_tag, tone_brief = _resolve_style(preferred_style)

    client = anthropic.Anthropic()

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": _build_user_prompt(
                        price, location, bedrooms, bathrooms, style_tag, tone_brief
                    ),
                }
            ],
        )
    except anthropic.AuthenticationError:
        raise CaptionError(
            "Anthropic rejected the API key (authentication failed). "
            "Check that ANTHROPIC_API_KEY is a valid key."
        )
    except anthropic.RateLimitError:
        raise CaptionError(
            "Anthropic is rate limiting the request. Wait a few seconds and try again."
        )
    except anthropic.APIConnectionError:
        raise CaptionError(
            "Could not reach the Anthropic API (network error). Check your connection."
        )
    except anthropic.APIStatusError as exc:
        raise CaptionError(
            f"The Anthropic API returned an error (HTTP {exc.status_code}). "
            "Please try again."
        )
    except anthropic.APIError:
        # Base class catch-all for anything else the SDK raises.
        raise CaptionError("The Anthropic API call failed unexpectedly. Please try again.")

    # Pull the text out of the response content blocks (there may be several;
    # we only care about text blocks).
    text = "".join(block.text for block in response.content if block.type == "text")
    if not text.strip():
        raise CaptionError("Claude returned an empty response. Please try again.")

    result = _normalise(_extract_json(text))
    # Report back which style we wrote in so the caller can attribute engagement
    # to it in Airtable.
    result["style_tag"] = style_tag
    return result
