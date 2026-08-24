import pytest

from app.agents import route_after_check
from app.llm import BudgetExceeded, Meter, extract_json, text_of
from app.search import search, tokenize


def test_extract_json_handles_fences_and_prose():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert extract_json('Here is the plan.\n\n{"queries": ["one"]}\n\nHope that helps.')["queries"] == ["one"]
    assert extract_json('{"note": "a } brace"}') == {"note": "a } brace"}
    assert extract_json("no json here", default={}) == {}


def test_text_of_handles_content_blocks():
    class Fake:
        content = [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "answer "},
            {"type": "text", "text": "continues"},
        ]

    assert text_of(Fake()) == "answer continues"


def test_tokenize_drops_stopwords():
    assert tokenize("What is the latency of the API") == ["latency", "api"]


def test_local_search_finds_the_slo_section():
    hits = search("tracking api latency SLO p99", k=3)
    assert any("250 ms" in h["snippet"] for h in hits)
    assert all(h["url"].startswith("corpus://") for h in hits)


def test_local_search_ranks_on_term_coverage():
    assert "incidents-2024" in search("billing double charge idempotency", k=2)[0]["url"]


def test_corpus_loads_regardless_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert search("deploy freeze December", k=2)


def test_routing_on_the_retry_edge():
    assert route_after_check({"confidence": 0.9, "revisions": 0, "queries": ["x"]}) == "accept"
    assert route_after_check({"confidence": 0.5, "revisions": 0, "queries": ["x"]}) == "retry"
    assert route_after_check({"confidence": 0.1, "revisions": 2, "queries": ["x"]}) == "accept"
    # no follow-up queries means another pass would repeat the last one
    assert route_after_check({"confidence": 0.1, "revisions": 0, "queries": []}) == "accept"


def fake_reply(model, input_tokens, output_tokens):
    class Reply:
        usage_metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        response_metadata = {"model": model}

    return Reply()


def test_meter_prices_and_accumulates():
    m = Meter()
    m.record(fake_reply("claude-sonnet-5", 1_000_000, 1_000_000))
    assert m.snapshot()["cost_usd"] == pytest.approx(12.0)  # $2 in + $10 out

    m.record(fake_reply("claude-opus-5", 100_000, 10_000))
    assert m.snapshot()["cost_usd"] == pytest.approx(12.75)
    assert m.snapshot()["calls"] == 2


def test_meter_raises_once_the_cap_is_crossed():
    m = Meter()
    m.record(fake_reply("claude-opus-5", 100_000, 0), limit=5.0)  # $0.50
    with pytest.raises(BudgetExceeded):
        m.record(fake_reply("claude-opus-5", 1_000_000, 0), limit=5.0)  # +$5.00


def test_meter_survives_a_response_without_usage():
    class Bare:
        pass

    m = Meter()
    m.record(Bare())
    assert m.snapshot()["cost_usd"] == 0.0
