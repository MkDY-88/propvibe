"""
PropVibe - Lead Qualification Chatbot
=====================================

Answers questions from a lead who has clicked through from a published post,
and quietly works out whether they're a serious prospect while doing it.

The one public function is `qualify_lead()` at the bottom. It returns a plain
dict::

    {
        "reply":            "<what to say back to the lead>",
        "status":           "lead" | "prospect",
        "budget_signal":    "Around RM 1,200/month" | None,
        "timeline_signal":  "Wants to move in by August" | None,
    }

`status` is deliberately conservative: everyone starts as a "lead" and only
becomes a "prospect" once the conversation has given real signal on BOTH
budget fit and move-in timeline. The two *_signal fields are what the caller
logs to Airtable so a human can see why - they're never shown to the lead.

Anything that goes wrong - a missing API key, a rate limit, a network blip, or
Claude returning something that isn't valid JSON - is raised as a single
`ChatError` carrying a short, human-readable message, exactly like
`copy_generator.CaptionError`. Callers can surface `str(exc)` directly.
"""

from __future__ import annotations

import json
import os
import re

import anthropic

# Same model as the caption generator: a chat turn needs to come back fast, and
# Haiku is more than capable of a friendly leasing conversation.
MODEL = "claude-haiku-4-5-20251001"

# A conversational reply plus two short signal strings is small; 1024 gives
# plenty of headroom while still capping a runaway response.
MAX_TOKENS = 1024

# The only two values `status` is ever allowed to take. The frontend and the
# Airtable "Status" column both key off these exact strings.
STATUS_LEAD = "lead"
STATUS_PROSPECT = "prospect"

# How many past turns to send back to Claude. A landing-page chat is short by
# nature; this keeps a long-running tab from growing the prompt without bound
# while still leaving ample room for budget/timeline to have come up earlier.
MAX_HISTORY_TURNS = 20

SYSTEM_PROMPT = (
    "You are a friendly, low-pressure leasing assistant chatting with someone "
    "who just clicked through from a social media post about a rental "
    "property. Your job is to be genuinely helpful about the place and, along "
    "the way, get a natural sense of their budget and when they want to move "
    "in.\n\n"
    "HOW TO TALK:\n"
    "- Warm, casual and brief - two or three sentences is usually plenty. You "
    "are texting, not writing a brochure.\n"
    "- Answer what they actually asked first. Be helpful before you are "
    "curious.\n"
    "- Let budget and move-in timing come up the way they would in a real "
    "conversation - one light question at a time, and only when it fits. "
    "NEVER run through a checklist of questions, and never ask both in the "
    "same message.\n"
    "- Never pressure, never push for a viewing more than once, and never ask "
    "for their name, phone number or email address - someone else handles "
    "that later.\n"
    "- If you don't know a detail about the property, say so plainly and "
    "offer to find out. Do not invent facts about the unit, the building or "
    "the neighbourhood.\n\n"
    "You reply with ONLY a single JSON object and nothing else - no markdown, "
    "no code fences, no commentary before or after. The JSON object must have "
    "exactly these four keys:\n"
    '  "reply":           a string. What you say back to them.\n'
    '  "status":          the string "lead" or "prospect" - nothing else.\n'
    '  "budget_signal":   a short string summarising what they have revealed '
    "about their budget (e.g. \"Comfortable around RM 1,200/month\"), or null "
    "if they haven't given any.\n"
    '  "timeline_signal": a short string summarising when they want to move '
    '(e.g. "Looking to move in early August"), or null if they haven\'t said.'
    "\n\n"
    'STATUS RULES: default to "lead". Only return "prospect" once the '
    "conversation has given you REAL signal on BOTH their budget fit for this "
    "property AND their move-in timeline. A vague \"soon\" or \"depends\" is "
    'not real signal. When in doubt, stay on "lead".'
)


class ChatError(Exception):
    """A clean, user-safe failure from :func:`qualify_lead`."""


def _build_context_block(listing_context: dict | None) -> str:
    """
    Describe the property Claude is answering about, for the system prompt.

    With a listing we hand over its real details so the assistant can talk
    about that specific place. Without one (an old tracking link, a post from
    before we recorded listing indices) we tell it to be generally helpful and
    ask what they're after, rather than let it improvise a property.
    """
    if not listing_context:
        return (
            "\n\nTHE PROPERTY: you do not know which specific listing this "
            "person is looking at. Do not guess or invent one. Be generally "
            "helpful, and early on ask what kind of place they're looking "
            "for - area, size, that sort of thing."
        )

    lines = []
    for label, key in (
        ("Name", "condo_name"),
        ("Price", "price"),
        ("Address", "address"),
        ("Features", "features_text"),
    ):
        value = listing_context.get(key)
        if isinstance(value, str) and value.strip():
            lines.append(f"- {label}: {value.strip()}")

    if not lines:
        return (
            "\n\nTHE PROPERTY: you do not know which specific listing this "
            "person is looking at. Do not guess or invent one. Be generally "
            "helpful, and early on ask what kind of place they're looking for."
        )

    return (
        "\n\nTHE PROPERTY they are asking about - mention it naturally, and "
        "do not contradict or go beyond these details:\n" + "\n".join(lines)
    )


def _clean_history(history: list) -> list[dict]:
    """
    Turn caller-supplied history into messages the Anthropic API will accept.

    The history comes straight off a JSON request body, so treat every part of
    it as untrusted: skip anything that isn't a dict with a "user"/"assistant"
    role and non-empty string content, drop any leading assistant turns (the
    API requires the conversation to start with a user message), and keep only
    the most recent MAX_HISTORY_TURNS entries.
    """
    cleaned: list[dict] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant"):
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        if not cleaned and role != "user":
            # A conversation can't open on an assistant turn.
            continue
        cleaned.append({"role": role, "content": content.strip()})

    cleaned = cleaned[-MAX_HISTORY_TURNS:]
    # Trimming can leave an assistant message at the front - drop those too.
    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)
    return cleaned


def _extract_json(text: str) -> dict:
    """
    Pull a JSON object out of Claude's reply, defensively.

    Same three-step approach as copy_generator._extract_json: parse the whole
    thing, then a stripped ```json fence, then the slice from the first '{' to
    the last '}'. If none of that yields an object we give up with a ChatError.
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

    raise ChatError(
        "The assistant did not return valid JSON for this message. "
        "Please try sending it again."
    )


def _optional_signal(value) -> str | None:
    """A trimmed signal string, or None for anything blank/missing/non-string."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    # Models sometimes write the word rather than emitting a JSON null.
    if not cleaned or cleaned.lower() in ("null", "none", "n/a", "unknown"):
        return None
    return cleaned


def _normalise(parsed: dict) -> dict:
    """
    Validate and tidy the parsed object into the exact shape callers expect.

    Guarantees a non-empty `reply`, a `status` that is exactly "lead" or
    "prospect" (anything unrecognised falls back to "lead" - the conservative
    side), and two signal fields that are either a clean string or None.
    """
    reply = parsed.get("reply")
    if not isinstance(reply, str) or not reply.strip():
        raise ChatError("The assistant's response was missing a usable 'reply'.")

    raw_status = parsed.get("status")
    status = raw_status.strip().lower() if isinstance(raw_status, str) else ""
    if status != STATUS_PROSPECT:
        # Unknown/missing status means we haven't earned "prospect" yet.
        status = STATUS_LEAD

    return {
        "reply": reply.strip(),
        "status": status,
        "budget_signal": _optional_signal(parsed.get("budget_signal")),
        "timeline_signal": _optional_signal(parsed.get("timeline_signal")),
    }


def qualify_lead(
    history: list[dict],
    message: str,
    listing_context: dict | None,
) -> dict:
    """
    Answer a lead's chat message and score how qualified they are.

    Args:
        history:         The conversation so far, oldest first, as
            ``[{"role": "user"|"assistant", "content": str}, ...]``. Entries
            that aren't in that shape are ignored, so a caller can pass the
            request body through untouched.
        message:         The new message from the lead. Must be non-empty.
        listing_context: The property they're looking at - a dict with any of
            "condo_name", "price", "address" and "features_text". Pass None
            when the listing is unknown (e.g. an old tracking link) and the
            assistant will stay general and ask what they're after instead of
            inventing a property.

    Returns:
        dict with keys "reply" (str), "status" ("lead" or "prospect"),
        "budget_signal" (str | None) and "timeline_signal" (str | None). The
        signals are for the caller's own records - don't show them to the lead.

    Raises:
        ChatError: for any failure - missing/invalid API key, rate limit,
            network error, or a response that isn't usable JSON. The message is
            safe to show a user.
    """
    if not isinstance(message, str) or not message.strip():
        raise ChatError("Please type a message before sending.")

    # Validated here (not at import) so the rest of the app boots without it -
    # only the endpoints that actually call Claude need this key.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ChatError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add "
            "your Anthropic API key, then restart the server."
        )

    messages = _clean_history(history)
    messages.append({"role": "user", "content": message.strip()})

    client = anthropic.Anthropic()

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT + _build_context_block(listing_context),
            messages=messages,
        )
    except anthropic.AuthenticationError:
        raise ChatError(
            "Anthropic rejected the API key (authentication failed). "
            "Check that ANTHROPIC_API_KEY is a valid key."
        )
    except anthropic.RateLimitError:
        raise ChatError(
            "Anthropic is rate limiting the request. Wait a few seconds and try again."
        )
    except anthropic.APIConnectionError:
        raise ChatError(
            "Could not reach the Anthropic API (network error). Check your connection."
        )
    except anthropic.APIStatusError as exc:
        raise ChatError(
            f"The Anthropic API returned an error (HTTP {exc.status_code}). "
            "Please try again."
        )
    except anthropic.APIError:
        # Base class catch-all for anything else the SDK raises.
        raise ChatError("The Anthropic API call failed unexpectedly. Please try again.")

    text = "".join(block.text for block in response.content if block.type == "text")
    if not text.strip():
        raise ChatError("The assistant returned an empty response. Please try again.")

    return _normalise(_extract_json(text))
