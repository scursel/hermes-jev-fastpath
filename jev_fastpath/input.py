"""Read-only extraction of the latest user text from a provider request.

The request is never mutated and never flattened: multimodal, binary, or otherwise
non-plain-text input returns ``None`` so the turn falls through to the real provider.
"""

from __future__ import annotations

from typing import Any, Mapping


def _part_get(part: Any, key: str) -> Any:
    if isinstance(part, Mapping):
        return part.get(key)
    return getattr(part, key, None)


def _joined_text(content: Any, text_types: tuple[str, ...]) -> str | None:
    """Accept only plain string content or a list of uniformly typed text parts."""
    if isinstance(content, str):
        return content
    if not isinstance(content, (list, tuple)):
        return None
    pieces: list[str] = []
    for part in content:
        part_type = _part_get(part, "type")
        text = _part_get(part, "text")
        if part_type not in text_types or not isinstance(text, str):
            return None
        pieces.append(text)
    if not pieces:
        return None
    return "\n".join(pieces)


def _converse_text(content: Any) -> str | None:
    """Accept only plain string content or Converse ``{"text": ...}`` blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, (list, tuple)):
        return None
    pieces: list[str] = []
    for part in content:
        text = _part_get(part, "text")
        if not isinstance(text, str) or _part_get(part, "reasoningContent") is not None:
            return None
        pieces.append(text)
    if not pieces:
        return None
    return "\n".join(pieces)


def _latest_user_content(messages: Any) -> Any:
    if not isinstance(messages, (list, tuple)):
        return None
    for message in reversed(messages):
        if not isinstance(message, Mapping):
            continue
        if message.get("role") == "user":
            return message.get("content")
    return None


def extract_latest_user_text(request: Any, api_mode: str) -> str | None:
    """Return the latest plain user text for the protocol, or ``None`` when unsupported.

    Supported shapes mirror what each Hermes transport actually builds:

    - ``chat_completions`` / ``anthropic_messages``: ``request["messages"]`` with string
      or ``{"type": "text", "text": ...}`` content;
    - ``codex_responses``: ``request["input"]`` as a string or message items with
      ``input_text`` parts;
    - ``bedrock_converse``: Converse ``messages`` with ``{"text": ...}`` blocks.
    """
    if not isinstance(request, Mapping):
        return None
    if api_mode in ("chat_completions", "anthropic_messages"):
        content = _latest_user_content(request.get("messages"))
        return _joined_text(content, ("text",))
    if api_mode == "codex_responses":
        input_items = request.get("input")
        if isinstance(input_items, str):
            return input_items
        if not isinstance(input_items, (list, tuple)):
            return None
        for item in reversed(input_items):
            if isinstance(item, Mapping) and item.get("role") == "user":
                return _joined_text(item.get("content"), ("input_text", "text"))
        return None
    if api_mode == "bedrock_converse":
        content = _latest_user_content(request.get("messages"))
        return _converse_text(content)
    return None
