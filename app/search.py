"""Search backends.

`local` indexes the markdown files under data/corpus so the pipeline (and the
eval harness) runs without network access. `tavily` hits the real web.
"""

import logging
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from app.config import get_settings

log = logging.getLogger(__name__)


def truncate(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "was", "were",
    "for", "on", "at", "by", "with", "what", "which", "how", "does", "do", "did",
    "it", "its", "as", "that", "this", "from", "be", "we", "our",
}


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


def search(query: str, k: int | None = None) -> list[dict]:
    s = get_settings()
    k = k or s.results_per_query
    if s.search_backend == "tavily":
        return _tavily(query, k)
    return _local(query, k)


# --- local corpus -----------------------------------------------------------

@lru_cache(maxsize=1)
def _chunks() -> list[dict]:
    """Split every corpus doc on its `##` headings. One chunk per section."""
    root = Path(get_settings().corpus_dir)
    out: list[dict] = []

    for path in sorted(root.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        doc_title = raw.splitlines()[0].lstrip("# ").strip() if raw else path.stem

        sections = re.split(r"^## +", raw, flags=re.M)
        for section in sections[1:]:
            heading, _, body = section.partition("\n")
            body = body.strip()
            if not body:
                continue
            out.append(
                {
                    "title": f"{doc_title} \u2014 {heading.strip()}",
                    "url": f"corpus://{path.name}#{_slug(heading)}",
                    "text": body,
                    "terms": Counter(tokenize(f"{heading} {body}")),
                }
            )

    if not out:
        log.warning("no corpus documents found under %s", root)
    return out


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _local(query: str, k: int) -> list[dict]:
    terms = tokenize(query)
    if not terms:
        return []

    scored = []
    for chunk in _chunks():
        score = sum(chunk["terms"].get(t, 0) for t in terms)
        # reward covering distinct query terms, not just repeating one of them
        coverage = sum(1 for t in set(terms) if chunk["terms"].get(t))
        if score:
            scored.append((coverage * 10 + score, chunk))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {
            "title": chunk["title"],
            "url": chunk["url"],
            "snippet": truncate(chunk["text"], 900),
        }
        for _, chunk in scored[:k]
    ]


# --- tavily -----------------------------------------------------------------

def _tavily(query: str, k: int) -> list[dict]:
    from tavily import TavilyClient

    key = get_settings().tavily_api_key
    if not key:
        log.warning("SEARCH_BACKEND=tavily but TAVILY_API_KEY is unset; returning nothing")
        return []

    try:
        res = TavilyClient(api_key=key).search(query, max_results=k, search_depth="advanced")
    except Exception as exc:  # network/quota problems shouldn't abort the graph
        log.warning("tavily search failed for %r: %s", query, exc)
        return []

    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": truncate(r.get("content", ""), 900),
        }
        for r in res.get("results", [])
    ]
