"""End-to-end graph runs with a scripted model, so the wiring and the retry edge
are covered without calling Anthropic."""

import json

import pytest
from langchain_core.messages import AIMessage

from app import agents
from app.agents import build_graph
from app.config import get_settings
from app.llm import BudgetExceeded, meter


class ScriptedLLM:
    """Returns queued replies in order. Raises if a node asks for one more than
    the script provides, which is usually a wiring bug rather than a bad script."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        if not self.replies:
            raise AssertionError(f"unscripted model call (#{self.calls + 1})")
        self.calls += 1
        return self.replies.pop(0)


@pytest.fixture
def script(monkeypatch):
    def install(replies):
        llm = ScriptedLLM(replies)
        monkeypatch.setattr(agents, "get_llm", lambda *a, **k: llm)
        meter.reset()
        return llm

    return install


def plan_reply(queries):
    return AIMessage(content=json.dumps({"subquestions": ["what are the SLOs"], "queries": queries}))


def search_call(query, call_id="call_1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": "web_search", "args": {"query": query}, "id": call_id, "type": "tool_call"}],
    )


def check_reply(claims, follow_up=()):
    return AIMessage(content=json.dumps({"claims": claims, "follow_up_queries": list(follow_up)}))


def test_happy_path(script):
    script(
        [
            plan_reply(["tracking api latency SLO"]),
            search_call("tracking api latency SLO p99"),
            AIMessage(content="Gathered the SLO section."),
            AIMessage(content="p99 is under 250 ms [1]."),
            check_reply([{"text": "p99 is under 250 ms", "verdict": "supported", "note": ""}]),
        ]
    )

    state = build_graph().invoke({"session_id": 0, "question": "What are the tracking API SLOs?"})

    assert state["confidence"] == 1.0
    assert state["revisions"] == 0
    assert state["draft"] == "p99 is under 250 ms [1]."
    assert state["evidence"], "the web_search tool call should have collected sources"
    assert state["evidence"][0]["rank"] == 1


def test_unsupported_claim_triggers_one_more_retrieval_pass(script):
    llm = script(
        [
            plan_reply(["sev1 incidents 2024"]),
            search_call("sev1 incidents 2024"),
            AIMessage(content="First pass done."),
            AIMessage(content="There were nine Sev-1 incidents [1]."),
            check_reply(
                [{"text": "There were nine Sev-1 incidents", "verdict": "contradicted", "note": "source says seven"}],
                follow_up=["incident review 2024 sev-1 count"],
            ),
            search_call("incident review 2024 sev-1 count", call_id="call_2"),
            AIMessage(content="Second pass done."),
            AIMessage(content="There were seven Sev-1 incidents [1]."),
            check_reply([{"text": "There were seven Sev-1 incidents", "verdict": "supported", "note": ""}]),
        ]
    )

    state = build_graph().invoke({"session_id": 0, "question": "How many Sev-1 incidents in 2024?"})

    assert state["revisions"] == 1
    assert state["confidence"] == 1.0
    assert "seven" in state["draft"]
    assert not llm.replies, "the whole script should have been consumed"


def test_retriever_falls_back_to_direct_search_when_no_tool_is_called(script):
    script(
        [
            plan_reply(["deploy freeze window"]),
            AIMessage(content="I'll answer from memory."),  # no tool call
            AIMessage(content="The freeze runs 1 December to 5 January [1]."),
            check_reply([{"text": "freeze runs 1 December to 5 January", "verdict": "supported", "note": ""}]),
        ]
    )

    state = build_graph().invoke({"session_id": 0, "question": "When is the deploy freeze?"})

    assert state["evidence"], "fallback search should still populate evidence"


def test_budget_cap_aborts_the_graph(script, monkeypatch):
    """The cap has to stop a run mid-graph, not just report afterwards."""
    monkeypatch.setattr(get_settings(), "max_spend_usd", 0.10)

    expensive = AIMessage(
        content='{"subquestions": ["a"], "queries": ["a"]}',
        response_metadata={"model": "claude-opus-5"},
        usage_metadata={"input_tokens": 100_000, "output_tokens": 0, "total_tokens": 100_000},
    )
    script([expensive, expensive, expensive])

    with pytest.raises(BudgetExceeded) as exc:
        build_graph().invoke({"session_id": 0, "question": "anything at all"})

    assert "0.50" in str(exc.value)  # one $0.50 call blew the $0.10 cap
    assert meter.snapshot()["calls"] == 1
    meter.reset()
