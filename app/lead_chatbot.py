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

`extract_search_intent()` is the small companion call: it reads the same turn
and reports whether the lead is asking to see OTHER properties, plus any
budget/area/room-type hints to search on. The caller uses that to look up real
alternatives and hand them to `qualify_lead()` as `other_listings`, so "what
else have you got around this price?" is answered from the actual listing pool
instead of from the model's imagination.

Anything that goes wrong in `qualify_lead` - a missing API key, a rate limit, a
network blip, or Claude returning something that isn't valid JSON - is raised as
a single `ChatError` carrying a short, human-readable message, exactly like
`copy_generator.CaptionError`. Callers can surface `str(exc)` directly.
`extract_search_intent` is the opposite: it is only a hint, so it never raises
and degrades to "no search intent" instead.
"""

from __future__ import annotations

import json
import logging
import os
import re

import anthropic

logger = logging.getLogger("propvibe.lead_chatbot")

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
    "OTHER PROPERTIES: the property under THE PROPERTY below is the one they "
    "clicked through for - it stays the main subject of the conversation and "
    "the first one you mention. Only bring up other places if they ask what "
    "else is available, and then only ones listed under OTHER LISTINGS. NEVER "
    "invent, guess at or half-remember a property that is not written out in "
    "this prompt - not a name, not a price, not an area."
)

# Appended AFTER the property/alternatives blocks, so the output contract is
# the last thing the model reads. It used to sit in the middle of the prompt,
# which was survivable until OTHER LISTINGS started arriving behind it: ending
# on "here are places you may offer" made Haiku answer in the conversational
# voice that block is written in and drop the JSON envelope entirely, roughly
# one alternatives-turn in three (a 502 for the lead, mid-conversation).
OUTPUT_FORMAT_PROMPT = (
    "\n\nOUTPUT FORMAT - this governs every reply, no matter what the "
    "conversation is about:\n"
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
    'not real signal. When in doubt, stay on "lead".\n\n'
    "Even when you are turning someone down, have nothing to offer them, or "
    "are listing other places, the reply is still that one JSON object - the "
    "prose goes inside \"reply\"."
)

# Put into Claude's mouth as the start of its own turn, so the response can
# only continue a JSON object. Belt to OUTPUT_FORMAT_PROMPT's braces: a lead
# mid-conversation should never see a 502 because a reply came back as prose.
JSON_PREFILL = "{"

# Four small fields, so a fraction of a chat turn's budget is plenty - and a
# tight cap keeps this second call from adding noticeable latency to a reply.
SEARCH_INTENT_MAX_TOKENS = 200

# Intent lives in the last thing they said plus a little surrounding context
# ("what about a different area?" needs the area they were just discussing).
# Far fewer turns than the reply call needs, and cheaper for it.
SEARCH_INTENT_HISTORY_TURNS = 6

# What extract_search_intent returns whenever it cannot get a usable answer -
# i.e. "carry on exactly as before". Copied, never handed out directly, so a
# caller that mutates the result can't poison the next request.
_NO_SEARCH_INTENT = {
    "wants_alternatives": False,
    "max_price": None,
    "location_keyword": None,
    "room_type_keyword": None,
}

SEARCH_INTENT_PROMPT = (
    "You read one message from someone chatting about a rental property and "
    "decide whether they are asking to see OTHER properties, plus what they "
    "would want from one. You are not replying to them - you only extract.\n\n"
    "Reply with ONLY a single JSON object and nothing else - no markdown, no "
    "code fences, no commentary. Exactly these four keys:\n"
    '  "wants_alternatives": true if their latest message is asking about '
    "other places, more options, something cheaper, or somewhere else - false "
    "if they are asking about the property already under discussion, or "
    "chatting about anything else.\n"
    '  "max_price":          the most they want to pay per month, as a plain '
    "whole number of ringgit with no currency symbol, commas or text (1200, "
    'not "RM 1,200"). null if they have not named a figure. Do NOT guess one '
    'from a vague phrase like "cheaper" or "around the same".\n'
    '  "location_keyword":   one or two words that would appear in a street '
    'address of the area they want (e.g. "Cheras", "Jalan Ipoh", "Bangsar"). '
    "null if they have not named an area, or if they only said they want a "
    "different one without saying where.\n"
    '  "room_type_keyword":  one word for the kind of room or bed they want '
    '(e.g. "Master", "Medium", "Single", "Queen"). null if they have not '
    "said.\n\n"
    "Prefer null over a guess: a wrong keyword finds the wrong places, while "
    "null simply searches more broadly."
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


def _listing_summary(item: dict) -> str:
    """One compact line describing an alternative listing, or "" if unusable."""
    if not isinstance(item, dict):
        return ""

    name = item.get("condo_name")
    if not isinstance(name, str) or not name.strip():
        # Without a name the assistant has nothing safe to call it by.
        return ""

    parts = [name.strip()]
    for key in ("price", "address", "features_text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return " - ".join(parts)


def _build_alternatives_block(other_listings: list[dict] | None) -> str:
    """
    Describe the other listings we found for this turn, for the system prompt.

    An empty list and None are told apart on purpose only in wording, not in
    policy: either way the assistant has nothing real to offer, so it must say
    so instead of filling the gap. The listing_context property stays primary
    in both cases.
    """
    lines = [line for line in (_listing_summary(item) for item in other_listings or []) if line]

    if not lines:
        return (
            "\n\nOTHER LISTINGS: you have none to hand for this message. If "
            "they are asking what else is available, tell them honestly that "
            "you don't have anything else to show them right now (or nothing "
            "matching what they described) and offer to check with the team. "
            "Do NOT invent an alternative property, and do not imply one "
            "exists."
        )

    return (
        "\n\nOTHER LISTINGS - real, currently available places you MAY offer "
        "if they are asking about other options. Mention at most two of them, "
        "by name and price, and only ones from this list:\n"
        + "\n".join(f"- {line}" for line in lines)
    )


def _clean_history(history: list, max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    """
    Turn caller-supplied history into messages the Anthropic API will accept.

    The history comes straight off a JSON request body, so treat every part of
    it as untrusted: skip anything that isn't a dict with a "user"/"assistant"
    role and non-empty string content, drop any leading assistant turns (the
    API requires the conversation to start with a user message), and keep only
    the most recent `max_turns` entries.
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

    cleaned = cleaned[-max_turns:] if max_turns > 0 else []
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


def _as_bool(value) -> bool:
    """True only for a real True or the string "true" - never for "false"."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def _as_positive_int(value) -> int | None:
    """A positive whole number from an int/float/numeric string, else None."""
    if isinstance(value, bool):  # bool is an int subclass - not a price.
        return None
    if isinstance(value, (int, float)):
        number = int(value)
    elif isinstance(value, str):
        digits = re.sub(r"[^\d.]", "", value)  # tolerate "RM 1,200" / "1200/month"
        if not digits:
            return None
        try:
            number = int(float(digits))
        except ValueError:
            return None
    else:
        return None
    return number if number > 0 else None


def extract_search_intent(history: list[dict], message: str) -> dict:
    """
    Read a lead's message for "show me something else" and what they'd want.

    Returns exactly::

        {"wants_alternatives": bool, "max_price": int | None,
         "location_keyword": str | None, "room_type_keyword": str | None}

    This is a hint for the caller's listing search, not part of answering the
    lead, so it NEVER raises: a missing API key, a rate limit, a network blip
    or a reply that isn't usable JSON all log a warning and come back as
    "no search intent", which just leaves the reply as it was before. Same
    belt-and-braces shape as main._best_performing_style().
    """
    if not isinstance(message, str) or not message.strip():
        return dict(_NO_SEARCH_INTENT)

    try:
        messages = _clean_history(history, max_turns=SEARCH_INTENT_HISTORY_TURNS)
        messages.append({"role": "user", "content": message.strip()})

        response = anthropic.Anthropic().messages.create(
            model=MODEL,
            max_tokens=SEARCH_INTENT_MAX_TOKENS,
            system=SEARCH_INTENT_PROMPT,
            messages=messages,
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        parsed = _extract_json(text)

        return {
            "wants_alternatives": _as_bool(parsed.get("wants_alternatives")),
            "max_price": _as_positive_int(parsed.get("max_price")),
            "location_keyword": _optional_signal(parsed.get("location_keyword")),
            "room_type_keyword": _optional_signal(parsed.get("room_type_keyword")),
        }
    except Exception as exc:  # noqa: BLE001 - a hint must never break the turn
        logger.warning("Could not read search intent from the lead's message: %s", exc)
        return dict(_NO_SEARCH_INTENT)


def qualify_lead(
    history: list[dict],
    message: str,
    listing_context: dict | None,
    other_listings: list[dict] | None = None,
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
        other_listings:  Real alternatives the assistant may offer if this
            message is asking what else is available, each shaped like
            `listing_context`. `listing_context` stays the primary property
            either way. Pass None (the default) when they aren't asking for
            alternatives; None and an empty list both mean the assistant says
            it has nothing else to show rather than inventing something.

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
    messages.append({"role": "assistant", "content": JSON_PREFILL})

    client = anthropic.Anthropic()

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            # Order matters: the two dynamic blocks describe what it may talk
            # about, and OUTPUT_FORMAT_PROMPT closes on how to say it - see the
            # note above OUTPUT_FORMAT_PROMPT for what happened when it didn't.
            system=(
                SYSTEM_PROMPT
                + _build_context_block(listing_context)
                + _build_alternatives_block(other_listings)
                + OUTPUT_FORMAT_PROMPT
            ),
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

    # The prefilled "{" is ours, not the model's, so it isn't in the response -
    # put it back before parsing or every object would be missing its opening
    # brace. (Checked emptiness first, above, on what the model actually said.)
    return _normalise(_extract_json(JSON_PREFILL + text))
