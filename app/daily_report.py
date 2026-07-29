"""
PropVibe - End-of-Day Report
============================

Assembles today's self-learning numbers (posts published, style mix, per-style
engagement, current winner) and asks Claude Haiku to write them up as a short
plain-language summary a solopreneur would actually read at the end of the day.

Two public functions::

    collect_report_data() -> dict
        The raw numbers, read-only from Airtable. BEST-EFFORT: a read failure
        (or Airtable not being configured) degrades to empty sections with the
        problem noted in an "errors" list - it never raises, matching the
        posture of every other reporting read in this app.

    generate_daily_report(data: dict) -> str
        Hands those numbers to Claude Haiku and returns the written summary.
        Raises ReportError on any failure (missing key, rate limit, network,
        unusable JSON) - same single-clean-exception contract as
        copy_generator.CaptionError, so the endpoint can catch exactly one
        thing and degrade to a "report unavailable" message.

The Claude call follows copy_generator's strict-JSON-envelope pattern: ask for
ONLY a JSON object, parse it defensively (whole reply, then a stripped code
fence, then the outermost braces), and validate the one field we need.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import date

import anthropic

from app.airtable_client import (
    AirtableError,
    best_performing_style,
    get_posts,
    get_style_performance,
    is_configured,
)

logger = logging.getLogger("propvibe.daily_report")

# Same model choice (and reasoning) as copy_generator: a one-shot, few-paragraph
# write-up is exactly what Haiku is fast and cheap at.
MODEL = "claude-haiku-4-5-20251001"

# A few short paragraphs; 1024 caps a runaway response with headroom to spare.
MAX_TOKENS = 1024

# Posts whose style_tag column is blank/renamed still count toward the total,
# bucketed under this label so the distribution always sums to total_posts.
UNKNOWN_STYLE = "Unknown"

SYSTEM_PROMPT = (
    "You are writing a short end-of-day performance report for the solo "
    "operator of PropVibe - an app that publishes property listing posts to "
    "Facebook, tracks each post's engagement (clicks + likes + comments), and "
    "learns which caption style performs best so future posts lean toward the "
    "winner.\n\n"
    "You will be given today's real numbers as JSON:\n"
    '  "total_posts": how many posts have been published.\n'
    '  "style_distribution": how many posts were written in each style.\n'
    '  "style_performance": AVERAGE engagement per style, from real synced '
    "data.\n"
    '  "winning_style": the style the system currently considers best, or '
    "null when there is no engagement data yet.\n"
    '  "errors": problems reading the data, if any - mention briefly that '
    "some data was unavailable when this is non-empty.\n\n"
    "Write the way a busy solopreneur would want to read their own day: what "
    "happened, which style is winning and by how much, and what the system "
    "learned and will adapt (future captions lean toward the winning style). "
    "A few short paragraphs of plain language - no markdown, no headers, no "
    "bullet lists.\n\n"
    "Ground every figure strictly in the JSON you are given. NEVER invent, "
    "estimate or extrapolate a number that is not present in it. If the data "
    "is thin (no posts yet, or no engagement synced yet), say so honestly "
    "instead of padding.\n\n"
    "You reply with ONLY a single JSON object and nothing else - no markdown, "
    "no code fences, no commentary before or after. The JSON object must have "
    "exactly this one key:\n"
    '  "report": a string. The written summary described above.'
)


class ReportError(Exception):
    """A clean, user-safe failure from :func:`generate_daily_report`."""


def collect_report_data() -> dict:
    """
    Today's self-learning numbers, assembled read-only from Airtable.

    Returns::

        {
          "date": "2026-07-30",
          "airtable_configured": true,
          "total_posts": 6,
          "style_distribution": {"Warm": 3, "Modern": 2, "Bold": 1},
          "style_performance": {"Warm": 12.5, "Modern": 8.0},   # avg per style
          "winning_style": "Warm",                              # or null
          "errors": [],
        }

    ``style_performance`` reuses get_style_performance() - the exact numbers
    the caption generator learns from - and ``winning_style`` reuses
    best_performing_style() on that same dict, so the report can never
    disagree with the style /create-post would actually pick.

    Never raises: Airtable being unconfigured or unreadable degrades to empty
    sections with the problem recorded in ``errors``, the same posture as
    every other reporting read in this app.
    """
    data: dict = {
        "date": date.today().isoformat(),
        "airtable_configured": is_configured(),
        "total_posts": 0,
        "style_distribution": {},
        "style_performance": {},
        "winning_style": None,
        "errors": [],
    }

    if not data["airtable_configured"]:
        data["errors"].append(
            "Airtable is not configured (AIRTABLE_API_KEY / AIRTABLE_BASE_ID "
            "are missing), so there is nothing to report yet."
        )
        return data

    try:
        posts = get_posts()
    except AirtableError as exc:
        logger.warning("Daily report could not read Posts: %s", exc)
        data["errors"].append(str(exc))
        posts = []

    data["total_posts"] = len(posts)
    distribution: dict[str, int] = {}
    for post in posts:
        style = post["style_tag"] or UNKNOWN_STYLE
        distribution[style] = distribution.get(style, 0) + 1
    # Biggest style first, so the JSON (and the prompt built from it) leads
    # with what the day was mostly made of.
    data["style_distribution"] = dict(
        sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    )

    # Best-effort by design: returns {} (with a logged warning) on any failure.
    performance = get_style_performance()
    data["style_performance"] = dict(
        sorted(performance.items(), key=lambda item: item[1], reverse=True)
    )
    data["winning_style"] = best_performing_style(performance)

    return data


def _extract_json(text: str) -> dict:
    """
    Pull a JSON object out of Claude's reply, defensively.

    Same three-step approach as copy_generator._extract_json: parse the whole
    reply, then a stripped ```json fence, then the first-'{'-to-last-'}' slice.
    """
    candidates: list[str] = []

    stripped = text.strip()
    candidates.append(stripped)

    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fence:
        candidates.append(fence.group(1).strip())

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

    raise ReportError(
        "Claude did not return valid JSON for the daily report. "
        "Please try again in a moment."
    )


def generate_daily_report(data: dict) -> str:
    """
    Ask Claude Haiku to write the end-of-day summary for the assembled numbers.

    Args:
        data: A collect_report_data() result. Passed to Claude verbatim as the
            only source of truth - the prompt forbids inventing figures that
            are not in it.

    Returns:
        The written summary: a few short paragraphs of plain text, no markdown.

    Raises:
        ReportError: for any failure - missing/invalid API key, rate limit,
            network error, or a response that isn't usable JSON. The message is
            safe to show a user, and the /daily-report endpoint catches this to
            degrade to a "report unavailable" line rather than 500ing.
    """
    # Validate the key here (not at import) so the rest of the app boots
    # without it - same guard as copy_generator.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ReportError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add "
            "your Anthropic API key, then restart the server."
        )

    user_prompt = (
        "Here are today's real numbers:\n\n"
        f"{json.dumps(data, indent=2)}\n\n"
        "Return the JSON object described in the system instructions."
    )

    client = anthropic.Anthropic()

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except anthropic.AuthenticationError:
        raise ReportError(
            "Anthropic rejected the API key (authentication failed). "
            "Check that ANTHROPIC_API_KEY is a valid key."
        )
    except anthropic.RateLimitError:
        raise ReportError(
            "Anthropic is rate limiting the request. Wait a few seconds and try again."
        )
    except anthropic.APIConnectionError:
        raise ReportError(
            "Could not reach the Anthropic API (network error). Check your connection."
        )
    except anthropic.APIStatusError as exc:
        raise ReportError(
            f"The Anthropic API returned an error (HTTP {exc.status_code}). "
            "Please try again."
        )
    except anthropic.APIError:
        # Base class catch-all for anything else the SDK raises.
        raise ReportError("The Anthropic API call failed unexpectedly. Please try again.")

    text = "".join(block.text for block in response.content if block.type == "text")
    if not text.strip():
        raise ReportError("Claude returned an empty response. Please try again.")

    report = _extract_json(text).get("report")
    if not isinstance(report, str) or not report.strip():
        raise ReportError("Claude's response was missing a usable 'report'.")
    return report.strip()
