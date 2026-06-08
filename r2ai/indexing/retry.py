"""Retry helpers for transient network and Qdrant gateway failures."""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}
RETRYABLE_MESSAGE_PARTS = (
    "bad gateway",
    "gateway timeout",
    "temporarily unavailable",
    "too many requests",
    "timeout",
    "timed out",
    "connection reset",
    "connection aborted",
    "connection refused",
    "connection error",
    "remote end closed",
    "server disconnected",
    "service unavailable",
)


def retry_request(
    operation: Callable[[], T],
    *,
    label: str,
    attempts: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
) -> T:
    """Run a request with exponential backoff for transient gateway/network errors."""
    max_attempts = attempts if attempts is not None else _env_int("R2AI_REQUEST_RETRIES", 12)
    delay = base_delay if base_delay is not None else _env_float("R2AI_REQUEST_BACKOFF_SECONDS", 2.0)
    delay_cap = max_delay if max_delay is not None else _env_float("R2AI_REQUEST_MAX_BACKOFF_SECONDS", 90.0)
    max_attempts = max(1, max_attempts)

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt >= max_attempts or not is_retryable_request_error(exc):
                raise
            sleep_for = min(delay_cap, delay * (2 ** (attempt - 1)))
            sleep_for *= random.uniform(0.75, 1.25)
            print(
                f"{label} failed with transient request error on attempt {attempt}/{max_attempts}: "
                f"{exc}. Retrying in {sleep_for:.1f}s",
                flush=True,
            )
            time.sleep(sleep_for)

    raise RuntimeError("unreachable retry state")


def is_retryable_request_error(exc: Exception) -> bool:
    """Best-effort detection for request errors raised by qdrant-client/httpx/urllib3."""
    status_code = _status_code(exc)
    message = str(exc).lower()
    if status_code in RETRYABLE_STATUS_CODES:
        return True
    if status_code == 404 and "bad gateway" in message:
        return True
    type_name = type(exc).__name__.lower()
    return any(part in message for part in RETRYABLE_MESSAGE_PARTS) or any(
        part in type_name for part in ("timeout", "connection", "network")
    )


def _status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value
    return None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default
