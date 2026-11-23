"""Shared network policy. Environment is checked at each public lookup boundary."""
import os
import re


def offline():
    return os.environ.get("RAVEN_EARS_OFFLINE", "").lower() in ("1", "true", "yes", "on")


def require_online(action):
    if offline():
        raise RuntimeError(f"{action} disabled by RAVEN_EARS_OFFLINE=1")


def local_audio_path(path, parser=None):
    """Reject URL/protocol inputs before any reader or metadata lookup sees them."""
    value = os.fspath(path)
    drive = re.match(r"^[A-Za-z]:[/\\]", value)
    if ("://" in value or value.startswith(("//", "\\\\")) or
            (re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value) and not drive)):
        message = "audio input must be a local file; URLs and protocol inputs are not supported"
        if parser is not None:
            parser.error(message)
        raise ValueError(message)
    return value
