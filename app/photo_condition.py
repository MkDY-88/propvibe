"""
PropVibe - Photo Condition Editor
==================================

Calls fal.ai's Nano Banana 2 Edit model to re-light a property photo for a
given time-of-day + weather combination, e.g. "evening" + "rainy".

Mirrors the env-at-call-time / custom-exception / explicit-timeout pattern
already used by app.copy_generator (Anthropic) and app.trend_research: the
FAL_API_KEY is read when a call is actually made, not at import, so the rest
of the app boots fine without it configured.

Every call here costs real money (fal.ai bills per image). This module never
decides whether to call fal.ai - that decision (cache first, always) belongs
to main.py's /toggle-photo-condition endpoint. This module only knows how to
make ONE edit call for a given (photo, time_of_day, weather) and turn the
result into bytes.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import httpx

TIME_OF_DAY_VALUES = ("morning", "evening", "night")
WEATHER_VALUES = ("sunny", "cloudy", "rainy")

FAL_EDIT_URL = "https://fal.run/fal-ai/nano-banana-2/edit"

# A real image edit can take tens of seconds. fal.run (as opposed to
# queue.fal.run) is a synchronous endpoint - it blocks until the edit is
# done and returns the result directly, so there's no polling to do here,
# just a generous timeout.
REQUEST_TIMEOUT_SECONDS = 90


class FalEditError(Exception):
    """A clean, user-safe failure from a fal.ai photo-edit call."""


TIME_OF_DAY_PROMPTS = {
    "morning": "early morning daylight, soft warm low-angle sunlight, clear bright sky",
    "evening": "golden-hour evening light, warm orange/pink low sun, long soft shadows",
    "night": "nighttime, dark sky, artificial/ambient lighting on visible surfaces, warm glowing lights",
}

WEATHER_PROMPTS = {
    "sunny": "clear sunny weather, bright blue sky, crisp sunlight",
    "cloudy": "overcast cloudy weather, soft diffuse light, grey sky",
    "rainy": "rainy weather, wet reflective surfaces, visible rain or rain-darkened sky",
}


def build_prompt(time_of_day: str, weather: str) -> str:
    """
    The single combined edit instruction for a (time_of_day, weather) pair.

    One prompt, one edit call - never chain a time-of-day edit and a weather
    edit as two separate calls, both to keep quality high (each extra edit
    pass compounds drift away from the original photo) and to keep spend
    bounded (one fal.ai call per combo, not two).
    """
    return (
        f"Edit this real-estate listing photo to show it during {time_of_day} "
        f"with {weather} weather: {TIME_OF_DAY_PROMPTS[time_of_day]}; "
        f"{WEATHER_PROMPTS[weather]}. CRITICAL: preserve the exact camera "
        "angle, framing, perspective, geometry, room layout and every "
        "object's position, size and shape exactly as in the original photo "
        "- do not move, add, remove, resize or redesign any furniture, "
        "structure, or object. Only change lighting, sky, and "
        "weather-related visual conditions (light colour/direction, "
        "shadows, sky appearance, wet/dry surfaces). Keep it photorealistic "
        "and keep it looking like the same real property."
    )


def _require_api_key() -> str:
    api_key = os.environ.get("FAL_API_KEY")
    if not api_key:
        raise FalEditError(
            "FAL_API_KEY is not set. Add it to your .env file "
            "(see .env.example), then restart the server."
        )
    return api_key


def _photo_data_uri(photo_path: str) -> str:
    """Base64-encode a local photo file into a data: URI fal.ai accepts inline."""
    raw = Path(photo_path).read_bytes()
    suffix = Path(photo_path).suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def edit_photo_condition(photo_path: str, time_of_day: str, weather: str) -> bytes:
    """
    Ask fal.ai's Nano Banana 2 Edit model to re-light one photo, and return
    the edited image bytes.

    `photo_path` must always be the ORIGINAL pool photo - never a
    previously-edited image. Chaining edits on top of edits would drift the
    result further from the real property on every pass and multiply spend
    with no cache benefit; the caller (main.py) is responsible for always
    passing the original.

    Raises:
        FalEditError: missing API key, a network failure, a non-2xx
            response, or a response we can't parse into an image URL. The
            message is safe to show a user / log directly.
    """
    if time_of_day not in TIME_OF_DAY_VALUES or weather not in WEATHER_VALUES:
        # Belt-and-braces: main.py validates these against the same tuples
        # before ever calling here, so this should be unreachable.
        raise FalEditError("Invalid time_of_day/weather combination.")

    api_key = _require_api_key()
    prompt = build_prompt(time_of_day, weather)
    image_uri = _photo_data_uri(photo_path)

    try:
        response = httpx.post(
            FAL_EDIT_URL,
            headers={"Authorization": f"Key {api_key}"},
            json={"prompt": prompt, "image_urls": [image_uri]},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except httpx.RequestError as exc:
        raise FalEditError(f"Could not reach fal.ai to edit the photo (network error): {exc}.")

    if not response.is_success:
        raise FalEditError(_error_message(response))

    try:
        payload = response.json()
    except ValueError:
        raise FalEditError("fal.ai returned an unreadable response.")

    images = payload.get("images")
    if not isinstance(images, list) or not images or not isinstance(images[0], dict):
        raise FalEditError("fal.ai's response did not include an edited image.")

    image_url = images[0].get("url")
    if not isinstance(image_url, str) or not image_url:
        raise FalEditError("fal.ai's response did not include an edited image URL.")

    try:
        image_response = httpx.get(image_url, timeout=REQUEST_TIMEOUT_SECONDS)
    except httpx.RequestError as exc:
        raise FalEditError(f"Could not download the edited image from fal.ai: {exc}.")

    if not image_response.is_success:
        raise FalEditError(f"Could not download the edited image from fal.ai (HTTP {image_response.status_code}).")

    return image_response.content


def _error_message(response: httpx.Response) -> str:
    """Pull fal.ai's own error text out of a failed response, or fall back to the status line."""
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("error") or payload.get("message")
        if isinstance(detail, str) and detail.strip():
            return f"fal.ai rejected the edit request: {detail.strip()}"
        if isinstance(detail, list) and detail:
            # FastAPI-style validation error shape: [{"msg": "...", ...}, ...]
            first = detail[0]
            if isinstance(first, dict) and isinstance(first.get("msg"), str):
                return f"fal.ai rejected the edit request: {first['msg']}"

    return f"fal.ai returned HTTP {response.status_code}."
