import json
import re
import threading
from functools import lru_cache

from langchain_anthropic import ChatAnthropic

from app.config import get_settings

# USD per million tokens, input/output. Sonnet 5 is on intro pricing until
# 2026-08-31 ($3/$15 after).
PRICES = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


class BudgetExceeded(RuntimeError):
    pass


class Meter:
    """The graph can loop, and a runaway retry cycle is the easy way to spend
    more than you meant to. Every call is priced and the cap is enforced."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.calls = 0
            self.input_tokens = 0
            self.output_tokens = 0
            self.cost = 0.0

    def record(self, message, limit=None):
        usage = getattr(message, "usage_metadata", None) or {}
        model = (getattr(message, "response_metadata", None) or {}).get("model", "")
        # Anthropic returns the resolved id, which may carry a suffix we don't price.
        price_in, price_out = next(
            (p for name, p in PRICES.items() if model.startswith(name)), (5.00, 25.00)
        )
        into, out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)

        with self._lock:
            self.calls += 1
            self.input_tokens += into
            self.output_tokens += out
            self.cost += (into * price_in + out * price_out) / 1_000_000
            spent = self.cost

        if limit is not None and spent >= limit:
            raise BudgetExceeded(f"spent ${spent:.2f} of the ${limit:.2f} cap after {self.calls} calls")

    def snapshot(self):
        with self._lock:
            return {
                "calls": self.calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cost_usd": round(self.cost, 4),
            }


meter = Meter()


@lru_cache(maxsize=8)
def get_llm(model=None, max_tokens=None):
    s = get_settings()
    return ChatAnthropic(
        model=model or s.model,
        max_tokens=max_tokens or s.max_tokens,
        timeout=180,
        max_retries=3,
    )


def chat_llm():
    s = get_settings()
    return get_llm(s.chat_model, s.chat_max_tokens)


def invoke(llm, messages):
    """Every model call goes through here so spend is metered and capped."""
    reply = llm.invoke(messages)
    meter.record(reply, limit=get_settings().max_spend_usd)
    return reply


def text_of(msg):
    """Responses come back as a list of content blocks once thinking is on, so
    .content isn't always a string."""
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)


def extract_json(text, default=None):
    """Models wrap JSON in prose or fences more often than you'd like. Pull out
    the first balanced object and parse it; return `default` rather than raising
    so one malformed response doesn't kill the run."""
    if not text:
        return default

    cleaned = _FENCE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1]
    if not starts:
        return default
    start = min(starts)

    opening = cleaned[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = escaped = False

    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start : i + 1])
                except json.JSONDecodeError:
                    return default
    return default
