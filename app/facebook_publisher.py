"""
PropVibe - Facebook Publisher
=============================

Publishes a finished poster + caption to a Facebook Page using the Graph API's
photo-upload endpoint.

The one public function you need is :func:`publish_post`. Give it the path to an
image and the post text and it uploads the photo with that text as the caption::

    result = publish_post("poster.png", "Just listed! ...")
    # -> {"post_id": "1234_5678", "post_url": "https://www.facebook.com/1234_5678"}

Anything that goes wrong - a missing token/page id, a network blip, or Facebook
rejecting the upload - is raised as a single :class:`FacebookPublishError`
carrying a short, human-readable message (Facebook's own error text when the API
returns one). Callers can surface ``str(exc)`` directly instead of leaking a
stack trace.

HTTP is done with ``httpx`` (already a dependency of the anthropic SDK) so we
don't pull in a second HTTP client just for this one call.
"""

from __future__ import annotations

import os

import httpx

# Pin the Graph API version so a future default bump on Facebook's side can't
# silently change the request/response shape underneath us.
GRAPH_API_VERSION = "v21.0"

# Uploading a full-resolution poster over a slow link can take a moment, so give
# the request generous headroom before we treat it as a network failure.
REQUEST_TIMEOUT_SECONDS = 60


class FacebookPublishError(Exception):
    """A clean, user-safe failure from :func:`publish_post`."""


def _require_credentials() -> tuple[str, str]:
    """
    Read the Page token and id from the environment, or raise a clear error.

    Validated here (not at import) so the rest of the app boots without them -
    only publishing actually needs them. Mirrors the missing-key guard in
    ``copy_generator.generate_caption``.
    """
    token = os.environ.get("FACEBOOK_PAGE_ACCESS_TOKEN")
    page_id = os.environ.get("FACEBOOK_PAGE_ID")

    if not token:
        raise FacebookPublishError(
            "FACEBOOK_PAGE_ACCESS_TOKEN is not set. Add it to your .env file "
            "(see .env.example), then restart the server."
        )
    if not page_id:
        raise FacebookPublishError(
            "FACEBOOK_PAGE_ID is not set. Add it to your .env file "
            "(see .env.example), then restart the server."
        )

    return token, page_id


def _extract_error_message(response: httpx.Response) -> str:
    """
    Pull Facebook's own error text out of a failed Graph API response.

    Graph API errors come back as ``{"error": {"message": "...", ...}}``. We
    prefer that human-readable message; if the body isn't the shape we expect
    (e.g. an HTML error page or empty body) we fall back to the status line so
    the caller still gets something actionable rather than nothing.
    """
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()

    return f"Facebook returned HTTP {response.status_code}."


def publish_post(image_path: str, message: str) -> dict:
    """
    Publish a photo with a caption to the configured Facebook Page.

    Args:
        image_path: Path to the image file to upload (e.g. the generated poster).
        message:    The full post text, used as the photo's caption.

    Returns:
        dict with:
          "post_id":  the id of the published post.
          "post_url": a URL to view the post
                      (``https://www.facebook.com/<post_id>``).

    Raises:
        FacebookPublishError: for any failure - missing credentials, a missing
            image file, a network error, or Facebook rejecting the upload. The
            message is safe to show a user and carries Facebook's own error text
            when the API returns one.
    """
    token, page_id = _require_credentials()

    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{page_id}/photos"

    try:
        image_file = open(image_path, "rb")
    except OSError as exc:
        raise FacebookPublishError(
            f"Could not open the image to publish ({image_path!r}): {exc.strerror or exc}."
        )

    try:
        response = httpx.post(
            url,
            data={"caption": message, "access_token": token},
            files={"source": image_file},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except httpx.RequestError as exc:
        raise FacebookPublishError(
            f"Could not reach Facebook to publish the post (network error): {exc}."
        )
    finally:
        image_file.close()

    if not response.is_success:
        # Surface Facebook's actual complaint (bad token, expired token, photo
        # too large, ...) rather than a generic status code.
        raise FacebookPublishError(_extract_error_message(response))

    try:
        payload = response.json()
    except ValueError:
        raise FacebookPublishError(
            "Facebook accepted the upload but returned an unreadable response."
        )

    # The /photos endpoint returns "post_id" (the viewable page-post id, e.g.
    # "<pageid>_<storyid>") alongside "id" (the photo object id). Prefer the
    # former; fall back to the photo id so we always hand back something.
    post_id = payload.get("post_id") or payload.get("id")
    if not post_id:
        raise FacebookPublishError(
            "Facebook accepted the upload but did not return a post id."
        )

    return {
        "post_id": post_id,
        "post_url": f"https://www.facebook.com/{post_id}",
    }


def _summary_count(section: object) -> int:
    """Pull ``.summary.total_count`` out of a Graph API edge, defaulting to 0."""
    if isinstance(section, dict):
        summary = section.get("summary")
        if isinstance(summary, dict):
            count = summary.get("total_count")
            if isinstance(count, (int, float)) and not isinstance(count, bool):
                return int(count)
    return 0


def get_post_engagement(post_id: str) -> dict:
    """
    Fetch the current like and comment counts for a published post.

    Uses the Graph API summary counts in a single request::

        GET /{post_id}?fields=likes.summary(true),comments.summary(true)

    Args:
        post_id: The Facebook post id (as returned by :func:`publish_post`).

    Returns:
        dict with integer "likes" and "comments" counts.

    Raises:
        FacebookPublishError: for a missing token, a network error, or Facebook
            returning an error (carrying Facebook's own message). Safe to show a
            user. Callers that sync many posts should catch this per-post so one
            failure doesn't abort the whole run.
    """
    token, _ = _require_credentials()

    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{post_id}"
    params = {
        "fields": "likes.summary(true),comments.summary(true)",
        "access_token": token,
    }

    try:
        response = httpx.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    except httpx.RequestError as exc:
        raise FacebookPublishError(
            f"Could not reach Facebook to read engagement (network error): {exc}."
        )

    if not response.is_success:
        raise FacebookPublishError(_extract_error_message(response))

    try:
        payload = response.json()
    except ValueError:
        raise FacebookPublishError(
            "Facebook returned an unreadable response when reading engagement."
        )

    return {
        "likes": _summary_count(payload.get("likes")),
        "comments": _summary_count(payload.get("comments")),
    }
