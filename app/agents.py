"""The four agents and the graph that wires them together."""

import json
import logging
from functools import lru_cache
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

from app.config import get_settings
from app.db import recall_reports
from app.llm import extract_json, get_llm, invoke, text_of
from app.search import search

log = logging.getLogger(__name__)

MAX_TOOL_STEPS = 6


class ResearchState(TypedDict, total=False):
    session_id: int
    question: str
    subquestions: list[str]
    queries: list[str]  # what the retriever should look up next
    evidence: list[dict]
    prior: list[dict]
    draft: str
    claims: list[dict]
    confidence: float
    revisions: int


PLANNER = """You plan research. Break the user's question into the smallest set of
sub-questions that together answer it, and write one search query per sub-question.

Rules:
- 2 to 5 sub-questions. Fewer is better when the question is narrow.
- Search queries are keyword-style, not full sentences.
- If prior findings already cover a sub-question, leave it out.

Reply with JSON only:
{"subquestions": ["..."], "queries": ["..."]}"""

RETRIEVER = """You gather evidence for a research report.

Call web_search once per query you were given. If a result set is thin or off
target, rephrase and search again. Call recall_prior_work when the question
sounds like a follow-up to something the session already covered.

Stop as soon as every sub-question has at least one supporting source. Then
reply with a one-line summary of what you found. Do not write the report."""

SYNTHESIZER = """You write the research report.

- Answer the question directly in the first sentence, then support it.
- Cite with bracketed numbers matching the source list, e.g. [2]. Every factual
  sentence needs at least one citation.
- Only state what the sources say. If they do not cover part of the question,
  say so plainly instead of filling the gap.
- Markdown, no title heading, under 600 words."""

REVISE = """A previous draft had unsupported claims. Rewrite the report so every
claim is backed by the sources below. Drop claims you cannot support.

Problems found:
{problems}"""

CHECKER = """You fact-check a draft report against the sources it cites.

Pull out every factual claim (skip hedges and framing sentences). For each one,
decide:
  supported     - a source states it
  unsupported   - no source states it, or the citation points somewhere else
  contradicted  - a source says otherwise

Judge only against the sources given. Your own knowledge is not evidence.

For anything not supported, write a search query that would settle it.

Reply with JSON only:
{"claims": [{"text": "...", "verdict": "supported", "note": "", "sources": [1]}],
 "follow_up_queries": ["..."]}"""


def plan(state):
    question = state["question"]
    prior = recall_reports(state["session_id"], question)

    context = ""
    if prior:
        summaries = "\n".join(f"- Q: {p['question']}\n  A: {p['answer'][:400]}" for p in prior)
        context = f"\n\nPrior findings in this session:\n{summaries}"

    reply = invoke(
        get_llm(),
        [SystemMessage(content=PLANNER), HumanMessage(content=f"Question: {question}{context}")],
    )
    parsed = extract_json(text_of(reply), default={}) or {}

    subquestions = [s for s in parsed.get("subquestions", []) if isinstance(s, str)]
    queries = [q for q in parsed.get("queries", []) if isinstance(q, str)]
    if not queries:
        log.warning("planner returned no queries, falling back to the raw question")
        queries = [question]

    return {
        "subquestions": subquestions or [question],
        "queries": queries[:5],
        "prior": prior,
        "evidence": [],
        "revisions": 0,
    }


def retrieve(state):
    session_id = state["session_id"]
    collected = []
    seen = {e["url"] for e in state.get("evidence", [])}

    @tool
    def web_search(query: str) -> str:
        """Search the corpus/web for a keyword query. Returns titles, urls and snippets."""
        hits = search(query)
        for h in hits:
            if h["url"] not in seen:
                seen.add(h["url"])
                collected.append(h)
        return json.dumps(hits, indent=2) if hits else "No results."

    @tool
    def recall_prior_work(query: str) -> str:
        """Look up earlier reports from this session that touch on `query`."""
        prior = recall_reports(session_id, query)
        return json.dumps(prior, indent=2) if prior else "Nothing on file."

    registry = {"web_search": web_search, "recall_prior_work": recall_prior_work}
    llm = get_llm().bind_tools(list(registry.values()))
    messages = [SystemMessage(content=RETRIEVER), HumanMessage(content=_task(state))]

    for _ in range(MAX_TOOL_STEPS):
        ai = invoke(llm, messages)
        messages.append(ai)
        if not ai.tool_calls:
            break
        for call in ai.tool_calls:
            fn = registry.get(call["name"])
            if fn is None:
                result = f"Unknown tool {call['name']}."
            else:
                try:
                    result = fn.invoke(call["args"])
                except Exception as exc:
                    log.warning("tool %s failed: %s", call["name"], exc)
                    result = f"Tool error: {exc}"
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    else:
        log.warning("retriever hit the tool-step ceiling with %d sources", len(collected))

    # Fall back to a direct search if the model never called a tool.
    if not collected and not state.get("evidence"):
        for q in state.get("queries", [])[:3]:
            for hit in search(q):
                if hit["url"] not in seen:
                    seen.add(hit["url"])
                    collected.append(hit)

    evidence = list(state.get("evidence", []))
    for hit in collected:
        evidence.append(
            {
                "rank": len(evidence) + 1,
                "title": hit.get("title", ""),
                "url": hit.get("url", ""),
                "snippet": hit.get("snippet", ""),
            }
        )
    return {"evidence": evidence, "queries": []}


def _task(state):
    lines = [f"Question: {state['question']}", "", "Sub-questions:"]
    lines += [f"- {s}" for s in state.get("subquestions", [])]
    lines += ["", "Suggested queries:"]
    lines += [f"- {q}" for q in state.get("queries", [])]
    if state.get("evidence"):
        lines += [
            "",
            f"You already have {len(state['evidence'])} sources. "
            "This pass is to close the specific gaps listed above only.",
        ]
    return "\n".join(lines)


def synthesize(state):
    sources = "\n\n".join(
        f"[{e['rank']}] {e['title']}\n{e['url']}\n{e['snippet']}" for e in state.get("evidence", [])
    ) or "(no sources found)"

    parts = [f"Question: {state['question']}", "", "Sources:", sources]

    failed = [c for c in state.get("claims", []) if c["verdict"] != "supported"]
    if failed:
        problems = "\n".join(f"- {c['text']} ({c['verdict']}: {c['note']})" for c in failed)
        parts += ["", REVISE.format(problems=problems)]

    reply = invoke(
        get_llm(), [SystemMessage(content=SYNTHESIZER), HumanMessage(content="\n".join(parts))]
    )
    return {"draft": text_of(reply).strip()}


def check(state):
    sources = "\n\n".join(
        f"[{e['rank']}] {e['title']}\n{e['snippet']}" for e in state.get("evidence", [])
    ) or "(no sources)"

    reply = invoke(
        get_llm(),
        [
            SystemMessage(content=CHECKER),
            HumanMessage(content=f"Draft:\n{state['draft']}\n\nSources:\n{sources}"),
        ],
    )
    parsed = extract_json(text_of(reply), default={}) or {}

    claims = []
    for raw in parsed.get("claims", []):
        if not isinstance(raw, dict) or not raw.get("text"):
            continue
        verdict = raw.get("verdict", "unsupported")
        if verdict not in {"supported", "unsupported", "contradicted"}:
            verdict = "unsupported"
        claims.append({"text": raw["text"], "verdict": verdict, "note": raw.get("note", "") or ""})

    if not claims:
        # Nothing parsed back. Treat it as unverified rather than a pass.
        log.warning("fact-checker returned no claims for question %r", state["question"])
        return {"claims": [], "confidence": 0.0, "queries": []}

    supported = sum(1 for c in claims if c["verdict"] == "supported")
    queries = [q for q in parsed.get("follow_up_queries", []) if isinstance(q, str)][:3]
    return {
        "claims": claims,
        "confidence": round(supported / len(claims), 3),
        "queries": queries,
    }


def route_after_check(state):
    s = get_settings()
    if state.get("confidence", 0.0) >= s.confidence_floor:
        return "accept"
    if state.get("revisions", 0) >= s.max_revisions:
        return "accept"
    if not state.get("queries"):
        # Nothing new to look up, so another retrieval pass would be identical.
        return "accept"
    return "retry"


def bump_revision(state):
    return {"revisions": state.get("revisions", 0) + 1}


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("planner", plan)
    g.add_node("retriever", retrieve)
    g.add_node("synthesizer", synthesize)
    g.add_node("checker", check)
    g.add_node("revise", bump_revision)

    g.set_entry_point("planner")
    g.add_edge("planner", "retriever")
    g.add_edge("retriever", "synthesizer")
    g.add_edge("synthesizer", "checker")
    g.add_conditional_edges("checker", route_after_check, {"retry": "revise", "accept": END})
    g.add_edge("revise", "retriever")

    return g.compile()
