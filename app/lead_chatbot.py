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

`extract_search_intent()` is the companion call: it reads the WHOLE
conversation and reports whether the lead needs to see OTHER properties, plus
the full set of preferences they have built up - budget, area, room, bed,
bathroom and must-have features. The filters accumulate rather than resetting
each turn, so a budget given early still applies to a later message that only
names an area, and changing their mind about one of them leaves the rest
standing. The caller feeds that to `listings_source.search_listings()` and
hands the ranked result to `qualify_lead()` as `other_listings`, so "what else
have you got around this price?" is answered from the actual listing pool
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
    "that later.\n\n"
    "WHAT YOU MAY TELL THEM:\n"
    "- Every detail written out below - name, price, full address, room type, "
    "bed, bathroom, parking - is yours to share, for the main property AND for "
    "any listing under OTHER LISTINGS. If they ask, answer with the actual "
    "detail. Never say you do not have something that is written down here, "
    "and never make them ask twice for something you were given.\n"
    "- If a detail genuinely is not written below (the floor, the wifi, "
    "deposit terms, when it is available), say in ONE short sentence that you "
    "do not have that detail, then carry on being useful - offer what you do "
    "know, or ask what else matters to them.\n"
    "- NEVER promise a human will follow up. There is no team behind this "
    "chat, no one to check with, nothing to get back to them about. Do not "
    'say "I\'ll check with the team", "let me find out", "I\'ll get back to '
    'you", "I can ask about that" or anything like them, ever. Do not point '
    'them at "someone who can check" either - you do not know that anyone '
    "will. What you do not have, you simply do not have.\n"
    "- If they push back on something you could not answer, do NOT apologise "
    "again or repeat yourself. Say it once, differently at most, and move the "
    "conversation forward.\n"
    "- NEVER invent or guess a property, price, address, feature or fact that "
    "is not written out below - not to be helpful, not as an example, not "
    "even hedged as a maybe.\n\n"
    "OTHER PROPERTIES: the property under THE PROPERTY below is the one they "
    "clicked through for - it stays the main subject of the conversation and "
    "the first one you mention. Only bring up other places if they ask what "
    "else is available, or follow up on one you have already mentioned, and "
    "then only ones listed under OTHER LISTINGS."
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

# Eight small fields, so a fraction of a chat turn's budget is plenty - and a
# tight cap keeps this second call from adding noticeable latency to a reply.
SEARCH_INTENT_MAX_TOKENS = 400

# This call re-derives the visitor's WHOLE requirement set every turn, so it
# needs the whole conversation - a budget mentioned early has to still be
# visible ten turns later. Same window as the reply call for that reason.
SEARCH_INTENT_HISTORY_TURNS = MAX_HISTORY_TURNS

# What extract_search_intent returns whenever it cannot get a usable answer -
# i.e. "carry on exactly as before". Copied, never handed out directly, so a
# caller that mutates the result can't poison the next request.
_NO_SEARCH_INTENT = {
    "wants_alternatives": False,
    "max_price": None,
    "target_price": None,
    "location_keyword": None,
    "room_type_keyword": None,
    "bed_type_keyword": None,
    "bathroom_type_keyword": None,
    "must_have_features": [],
}

# At most this many feature keywords are passed to the search. Every extra one
# is another constraint that has to be relaxed away, and a visitor who really
# has five hard requirements is better served by the first few than by a search
# that matches nothing and falls all the way down the relaxation ladder.
MAX_FEATURE_KEYWORDS = 4

SEARCH_INTENT_PROMPT = (
    "You read a conversation between a visitor and a leasing assistant, and "
    "report what the visitor is looking for RIGHT NOW. You are not replying to "
    "them - you only extract.\n\n"
    "THE WHOLE CONVERSATION, NOT JUST THE LAST MESSAGE. Build one single, "
    "current set of requirements from everything they have said:\n"
    "- A preference they gave earlier STILL APPLIES unless they have since "
    "changed or dropped it. If they said RM 900 four messages ago and now only "
    'ask "anything in Cheras?", the answer is RM 900 AND Cheras.\n'
    "- If they change their mind about something (\"actually, more like RM "
    '800"), the NEW value replaces the old one FOR THAT ONE FIELD ONLY. '
    "Everything else they said earlier still stands - do not clear it, and do "
    "not merge the old and new values.\n"
    "- Only leave a field null if they have never given it, or have explicitly "
    "said they no longer mind about it.\n"
    "- A QUESTION IS NOT A PREFERENCE. Record only what they say they want, "
    'need or are looking for. "Is the bathroom shared or private?" is asking '
    "what this one HAS - it does not mean they want either, so "
    'bathroom_type_keyword stays null. "How much is it?" is not a budget. Only '
    'something like "I\'d want my own bathroom" sets that field.\n\n'
    "Reply with ONLY a single JSON object and nothing else - no markdown, no "
    "code fences, no commentary. Exactly these eight keys:\n"
    '  "wants_alternatives":    true if answering their LATEST message needs '
    "properties other than the one they originally clicked through for. True "
    "when they ask what else is available or want something cheaper or "
    "elsewhere; true for a follow-up about an alternative already mentioned to "
    'them ("does that one have parking?", "tell me more about the second '
    'one"); and true when they simply DESCRIBE what they are after - naming a '
    "budget, an area, a room type or a must-have is itself asking to be shown "
    "something that fits, whether or not they use the words 'anything else'. "
    "False only when the latest message is purely about the original property "
    "(its rent, its bathroom, seeing it), is about their move-in timing, or is "
    "a greeting or small talk.\n"
    '  "max_price":             a monthly CEILING, as a plain whole number of '
    "ringgit with no symbol, commas or text (1200, not \"RM 1,200\"). Use this "
    'for "under X", "up to X", "no more than X", "my budget is X". null '
    "otherwise.\n"
    '  "target_price":          a monthly figure they named LOOSELY - "around '
    'X", "about X", "roughly X", "X or so". null otherwise. Set max_price OR '
    "target_price, never both, and never guess a number from a vague word like "
    '"cheaper" or "affordable" - leave both null instead.\n'
    '  "location_keyword":      one or two words that would appear in the '
    'street address of the area they want (e.g. "Cheras", "Mont Kiara", '
    '"Petaling Jaya"). null if they have not named an area, or only said they '
    "want a different one without saying where.\n"
    '  "room_type_keyword":     the kind of room (e.g. "Master", "Medium", '
    '"Single", "Studio", "Partitioned"). null if not said.\n'
    '  "bed_type_keyword":      the bed, and only if they mention it '
    'specifically (e.g. "Queen", "Single"). null if not said.\n'
    '  "bathroom_type_keyword": "Private" if they want their own bathroom '
    '(ensuite, attached, not sharing), "Shared" if they are happy sharing or '
    "asked for it. null if it has not come up.\n"
    '  "must_have_features":    a JSON array of short lowercase keywords for '
    'things they said they need, e.g. ["parking", "balcony", "big window"]. '
    "Use an empty array [] when they have not named any. Do not put budget, "
    "area, room type, bed or bathroom in here - they each have their own field "
    "above.\n\n"
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


def _build_alternatives_block(
    other_listings: list[dict] | None,
    alternatives_exact: bool = True,
    relaxed_criteria: list[str] | tuple[str, ...] | None = None,
) -> str:
    """
    Describe the other listings we found for this turn, for the system prompt.

    An empty list and None are told apart on purpose only in wording, not in
    policy: either way the assistant has nothing real to offer, so it must say
    so instead of filling the gap. The listing_context property stays primary
    in both cases.

    `alternatives_exact` is False when the search had to loosen what they asked
    for to find anything (see listings_source.SearchResult). That distinction
    has to reach the assistant, not be quietly flattened here: presenting a
    RM 800 room to someone who said "under RM 700" as though it matched is the
    one thing worse than having nothing to show them.
    """
    lines = [line for line in (_listing_summary(item) for item in other_listings or []) if line]

    if not lines:
        return (
            "\n\nOTHER LISTINGS: there are none. Either nothing in the pool "
            "matches what they described, or they have not asked. If they are "
            "asking what else is available, tell them plainly and in one "
            "sentence that you have nothing matching that - and do not offer "
            "to check, ask around or come back to them, because you cannot. "
            "You may ask whether they would flex on the budget or the area, "
            "since that would let you look again. NEVER invent an alternative "
            "property and never imply one exists."
        )

    header = (
        "\n\nOTHER LISTINGS - real, currently available places you MAY offer "
        "when they ask about other options or follow up on one of them. Only "
        "ever name places from this list. Mention at most two at a time so the "
        "reply stays short, but if they ask about any of them you may share "
        "ANY detail on its line - price, full address, room type, bed, "
        "bathroom, parking cost:\n"
    )

    if alternatives_exact:
        return header + "\n".join(f"- {line}" for line in lines)

    given_up = ", ".join(relaxed_criteria or []) or "some of what they asked for"
    return (
        "\n\nIMPORTANT - THESE ARE NOT EXACT MATCHES. Nothing available "
        f"matches everything they asked for, so the search loosened {given_up} "
        "to find the nearest things. You MUST say so before or as you offer "
        'them - something like "nothing matches that exactly, but the closest '
        'I have is..." - and be specific about where they fall short (over '
        "budget, a shared bathroom rather than private, and so on) by "
        "comparing what they asked for against the details below. NEVER "
        "present one of these as though it met the original request."
        + header
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


def _as_keyword_list(value) -> list[str]:
    """
    A short list of clean feature keywords from whatever the model emitted.

    Accepts the JSON array it is asked for, and also tolerates the single
    comma-separated string it sometimes sends instead. Anything else - a dict,
    a number, the literal word "null" - is simply no features rather than an
    error, because this whole call is a hint.
    """
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return []

    keywords: list[str] = []
    for item in items:
        cleaned = _optional_signal(item)
        if cleaned and cleaned.lower() not in (word.lower() for word in keywords):
            keywords.append(cleaned)
    return keywords[:MAX_FEATURE_KEYWORDS]


def extract_search_intent(history: list[dict], message: str) -> dict:
    """
    Read the WHOLE conversation for what this visitor is currently after.

    Returns exactly::

        {"wants_alternatives": bool,
         "max_price": int | None, "target_price": int | None,
         "location_keyword": str | None, "room_type_keyword": str | None,
         "bed_type_keyword": str | None, "bathroom_type_keyword": str | None,
         "must_have_features": list[str]}

    The filters are accumulated, not per-message: a budget given five turns ago
    still applies to a message that only names an area, and a change of mind
    ("actually, more like RM 800") replaces just that one field. That is why
    this sends the full history rather than the last couple of turns - the
    model is re-deriving the current requirement set each time, not reading one
    sentence in isolation.

    `wants_alternatives` gates whether the caller searches at all. It covers
    follow-up questions about an alternative already mentioned ("does that one
    have parking?"), because those need the same listings in front of the
    assistant to answer - and since the filters carry forward, re-running the
    search returns the same rows it was just talking about. It is also forced
    true whenever any concrete preference is on the table, model judgement or
    not: see the note at the `has_preferences` line for why erring towards
    searching is the safe direction.

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
        # Same trick as the reply call: start its turn mid-object so the only
        # thing it can do is finish the JSON. Cheap insurance on a call whose
        # failure mode is silently searching with no filters at all.
        messages.append({"role": "assistant", "content": JSON_PREFILL})

        response = anthropic.Anthropic().messages.create(
            model=MODEL,
            max_tokens=SEARCH_INTENT_MAX_TOKENS,
            system=SEARCH_INTENT_PROMPT,
            messages=messages,
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        parsed = _extract_json(JSON_PREFILL + text)

        max_price = _as_positive_int(parsed.get("max_price"))
        target_price = _as_positive_int(parsed.get("target_price"))
        if max_price is not None and target_price is not None:
            # Told to send one or the other; if it sends both, the ceiling is
            # the safer reading - it is the one a visitor would be annoyed to
            # see ignored.
            target_price = None

        features = _as_keyword_list(parsed.get("must_have_features"))
        location = _optional_signal(parsed.get("location_keyword"))
        room_type = _optional_signal(parsed.get("room_type_keyword"))
        bed_type = _optional_signal(parsed.get("bed_type_keyword"))
        bathroom_type = _optional_signal(parsed.get("bathroom_type_keyword"))

        # Stating a requirement IS asking to be shown something that meets it,
        # so any concrete preference turns the search on regardless of how the
        # model judged the question. It kept answering false for "I'm after a
        # master room in Mont Kiara under RM 700" - reasonably, since it is
        # never told WHICH property they clicked through for and so cannot tell
        # "somewhere else" from "this place" - and the assistant then told
        # someone we had nothing when we had a near miss to show them.
        #
        # Erring towards searching is the safe direction: a spare list of real
        # listings the assistant is told to keep secondary costs a turn
        # nothing, while a missed search makes it deny stock we actually have.
        # With no preferences at all this is still false, so a plain
        # single-property conversation searches exactly as little as before.
        has_preferences = any(
            (max_price, target_price, location, room_type, bed_type, bathroom_type, features)
        )

        return {
            "wants_alternatives": _as_bool(parsed.get("wants_alternatives")) or has_preferences,
            "max_price": max_price,
            "target_price": target_price,
            "location_keyword": location,
            "room_type_keyword": room_type,
            "bed_type_keyword": bed_type,
            "bathroom_type_keyword": bathroom_type,
            "must_have_features": features,
        }
    except Exception as exc:  # noqa: BLE001 - a hint must never break the turn
        logger.warning("Could not read search intent from the lead's message: %s", exc)
        return dict(_NO_SEARCH_INTENT)


def qualify_lead(
    history: list[dict],
    message: str,
    listing_context: dict | None,
    other_listings: list[dict] | None = None,
    alternatives_exact: bool = True,
    relaxed_criteria: list[str] | tuple[str, ...] | None = None,
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
        alternatives_exact: False when `other_listings` came back from a
            search that had to loosen the lead's criteria to find anything -
            i.e. listings_source.SearchResult.is_exact. The assistant is then
            told to offer them explicitly as near misses. Leaving this True for
            relaxed results would have it present a compromise as a match.
        relaxed_criteria: What that search gave up ("price range",
            "must-have features"), from SearchResult.relaxed_criteria, so the
            assistant can name the shortfall instead of waving at it. Ignored
            when `alternatives_exact` is True.

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
                + _build_alternatives_block(
                    other_listings, alternatives_exact, relaxed_criteria
                )
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
