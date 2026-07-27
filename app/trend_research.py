"""
PropVibe - Trend Research
=========================

Asks Claude Haiku to do a quick web search for what's happening in a listing's
area right now (market movement, buyer interest, new amenities/infrastructure)
and hands back one or two plain-text sentences a caption can casually reference.

The one public function is `research_trend()`. It returns either a short string
or None - and it NEVER raises. A missing API key, a network blip, a timeout, or
a search that turns up nothing usable all come back as None, because this is an
enhancement to the caption, not a prerequisite for it. /create-post is expected
to carry on without it.

This is deliberately a SEPARATE Anthropic call from `copy_generator`. That module
asks for a strict JSON object; mixing a web-search tool loop into the same
request makes the JSON far more fragile (the model interleaves search results
and commentary with the answer). Two small, single-purpose calls are easier to
reason about and to fail independently.

Results are memo-ised per (location, day) for the life of the process, so
generating several posts for the same area in one session only searches once.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import date

import anthropic

logger = logging.getLogger("propvibe.trend_research")

# Same model as the caption call: fast and cheap is the whole point here.
MODEL = "claude-haiku-4-5-20251001"

# The basic web-search server tool. Haiku 4.5 takes this variant - the newer
# web_search_20260209 (with dynamic result filtering) needs an Opus/Sonnet-tier
# model, so asking for it here would just be an error.
WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    # One or two searches is plenty for a one-line note, and it caps both the
    # latency and the per-search cost of an optional extra.
    "max_uses": 2,
}

# A couple of sentences of prose. Small on purpose - if the model wants to write
# an essay we'd rather cut it off than pay for it.
MAX_TOKENS = 400

# Hard ceiling on how much of the reply we hand to the caption prompt, so a
# chatty response can't crowd out the actual listing details.
MAX_SUMMARY_CHARS = 300

# Anything shorter than this isn't a usable sentence - treat it as "found
# nothing" rather than feeding a fragment into the caption.
MIN_SUMMARY_CHARS = 25

# Wall-clock budget for the whole call. A web search plus a Haiku reply is
# normally a handful of seconds; this bounds the worst case so /create-post
# never hangs waiting on an optional extra. Paired with max_retries=0 below,
# this really is the ceiling (the SDK would otherwise retry and multiply it).
REQUEST_TIMEOUT_SECONDS = 15.0

# The model is told to reply with exactly this when the search turns up nothing
# worth mentioning, which is far more reliable than trying to sniff out phrases
# like "I couldn't find anything" after the fact.
NO_RESULT_SENTINEL = "NONE"

SYSTEM_PROMPT = (
    "You are a property market researcher for a Malaysian real estate agency.\n\n"
    "Given a location, run ONE quick web search and report anything currently "
    "notable about it: price movement, buyer or rental demand, new "
    "infrastructure or amenities, upcoming launches, or neighbourhood "
    "developments.\n\n"
    "Reply with ONE or TWO short plain-text sentences and nothing else - no "
    "preamble, no markdown, no bullet points, no JSON, no citations or URLs. "
    "Write it so it could be dropped casually into a social media post about a "
    "property there, and keep it factual - do not invent numbers.\n\n"
    f"If the search turns up nothing genuinely notable, reply with exactly "
    f"{NO_RESULT_SENTINEL} and nothing else. Saying nothing is better than "
    "padding with generic filler like 'a popular area with good amenities'."
)

# Cache for the lifetime of the process: {"<location>|<date>": summary or None}.
# Deliberately not persisted - a fresh search per server run is cheap enough,
# and the point is only to avoid re-searching the same area seconds apart while
# a user generates several posts. `None` entries are cached too, so a location
# with nothing to report doesn't retrigger a search on every post.
_CACHE: dict[str, str | None] = {}


def _cache_key(location: str) -> str:
    """Cache key for a location, scoped to today so trends don't go stale."""
    return f"{location.strip().lower()}|{date.today().isoformat()}"


def _extract_summary(content: list) -> str:
    """
    Pull the model's actual answer out of the response content blocks.

    A web-search turn interleaves blocks: the model may narrate ("Let me look
    that up"), then a `server_tool_use` block, then a `web_search_tool_result`,
    then the real answer. We want the last of those, so we take only the text
    that comes AFTER the final search-related block. If no search happened at
    all, every text block is fair game.

    Note the text blocks are CONCATENATED, not joined with a space: a cited
    answer arrives split into several adjacent fragments, and they already carry
    their own spacing. Joining with a space instead yields "in 2026 , with" and
    "tenant pool ." - _tidy() then collapses whitespace on the result.
    """
    last_search_index = -1
    for index, block in enumerate(content):
        if block.type in ("server_tool_use", "web_search_tool_result"):
            last_search_index = index

    def _join(blocks) -> str:
        return "".join(block.text for block in blocks if block.type == "text")

    text = _join(content[last_search_index + 1 :])
    if not text:
        # The model answered without searching (or only narrated before it) -
        # fall back to everything it said.
        text = _join(content)

    return text


def _tidy(text: str) -> str | None:
    """
    Normalise the model's reply into a single clean line, or None if unusable.

    Collapses whitespace, strips wrapping quotes, drops the "found nothing"
    sentinel, and trims to MAX_SUMMARY_CHARS on a word boundary.
    """
    cleaned = re.sub(r"\s+", " ", text).strip().strip('"').strip()
    if not cleaned:
        return None

    # The sentinel may come back bare or dressed up ("NONE." / "None").
    if cleaned.rstrip(".").strip().upper() == NO_RESULT_SENTINEL:
        return None

    if len(cleaned) < MIN_SUMMARY_CHARS:
        return None

    if len(cleaned) > MAX_SUMMARY_CHARS:
        cleaned = cleaned[:MAX_SUMMARY_CHARS].rsplit(" ", 1)[0].rstrip(",;:-") + "..."

    return cleaned


def research_trend(location: str) -> str | None:
    """
    Look up a short, current market note for `location` - or None.

    Args:
        location: Short location line, e.g. "Mont Kiara, Kuala Lumpur".

    Returns:
        One or two plain-text sentences suitable for a caption to reference, or
        None when there's no API key, the call fails, or the search turned up
        nothing worth saying. Never raises - callers can use the result directly
        without a try/except.
    """
    if not isinstance(location, str) or not location.strip():
        return None

    key = _cache_key(location)
    if key in _CACHE:
        logger.debug("Trend research cache hit for %r", location)
        return _CACHE[key]

    # Same deal as generate_caption(): validate at call time, not at import, so
    # the app still boots without a key - this endpoint just skips the extra.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("Skipping trend research: ANTHROPIC_API_KEY is not set.")
        return None

    summary: str | None = None
    try:
        # max_retries=0 keeps the worst case at REQUEST_TIMEOUT_SECONDS. A retry
        # would be nice-to-have, but not at the cost of doubling how long
        # /create-post can sit waiting on an optional enhancement.
        client = anthropic.Anthropic(
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=[WEB_SEARCH_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": (
                        "What's currently notable about the property market in "
                        f"{location.strip()}, Malaysia?"
                    ),
                }
            ],
        )
        summary = _tidy(_extract_summary(response.content))
    except Exception as exc:  # noqa: BLE001 - an optional extra must never fail the request
        # Covers everything the SDK can raise (missing/invalid key, rate limit,
        # timeout, connection error, unexpected response shape). We log it and
        # move on rather than caching, so a transient blip doesn't disable trend
        # research for this location for the rest of the process.
        logger.warning("Trend research failed for %r: %s", location, exc)
        return None

    if summary is None:
        logger.info("Trend research found nothing usable for %r", location)
    else:
        logger.info("Trend research for %r: %s", location, summary)

    _CACHE[key] = summary
    return summary
