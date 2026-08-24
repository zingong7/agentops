# AgentOps

Ask a research question, get a cited report. Four agents split the work: a
planner breaks the question up, a retriever gathers evidence with tool calls, a
synthesizer writes the draft, and a fact-checker verifies every claim against
the sources that were actually retrieved. Anything it can't support goes back to
the retriever with new search queries.

LangGraph for the pipeline, FastAPI on top, Postgres for state.

```
question -> planner -> retriever -> synthesizer -> checker -> done
                 ^                                    |
                 +---- unsupported claims ------------+
```

The retry edge is the part worth looking at. The checker only counts a claim as
supported if a retrieved source says it, and it writes follow-up queries for the
ones that fail, so a weak first pass gets targeted searches instead of a reroll
of the same prompt. `CONFIDENCE_FLOOR` and `MAX_REVISIONS` bound the loop.

## Results

Ten questions with hand-written ground truth, `claude-sonnet-5`, 2026-08-23:
**92.6% factual accuracy** (25/27 facts), 23s median per question, $0.45 for the
whole run.

Nine of ten were perfect. Both misses were on the same question and were metric
artifacts, not errors: the report said "December 1 to January 5" where the
ground truth expected "1 December". On facts it was 27/27. I left the ground
truth strict rather than teaching the matcher about date order; loosening a
metric until it agrees with your output is how you end up trusting it when it's
wrong.

One question fell below the confidence floor, took the retry edge, went from 4
sources to 8, and the second draft covered everything.

`/chat` measured separately over 7 turns in one session on `claude-haiku-4-5`:
**1.5s mean, 0.97s median**, range 0.63-3.31s. The slowest turn was the first
one, which is also the one that pulled a full report out of memory; later turns
answer from context that's already assembled. One of seven crossed 3 seconds, so
treat the mean as the honest figure rather than a ceiling.

A full `/research` run through the API took 27s end to end, including writing
the report, its 5 sources and 7 claim verdicts to the database.

## Running it

```bash
cp .env.example .env      # needs ANTHROPIC_API_KEY
docker compose up --build
```

Or locally against SQLite, no Postgres needed. Set
`DATABASE_URL=sqlite+pysqlite:///./agentops.db` and:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Interactive docs at http://localhost:8000/docs. Tables are created on startup;
no migrations yet.

`POST /research` runs the full graph. It takes minutes, so it queues the run and
returns an id to poll at `/runs/{id}`. `POST /chat` is a single model call
against session memory (conversation history plus reports already produced in
that session) for asking follow-ups without re-running anything.

## Search

`SEARCH_BACKEND=local` (the default) indexes the markdown in `data/corpus/`, a
handful of invented internal docs for a logistics company. That's what makes the
eval reproducible without a search key. `SEARCH_BACKEND=tavily` swaps in live web
search.

## Eval

```bash
python -m eval.run_eval
python -m eval.run_eval --only slo-latency
```

`eval/questions.json` holds the questions and the facts a correct answer has to
contain. Scoring is string containment, which is blunt. It catches a number
being invented or dropped and misses paraphrase, so treat it as a regression
guard on prompt changes rather than a quality score. Results land in
`eval/results.json`.

## Cost

Every model call goes through the meter in `app/llm.py`, which prices it from
the response's token usage and raises once the total crosses `MAX_SPEND_USD`.
The graph loops, and a runaway retry cycle is the easy way to overspend, so the
cap is enforced rather than advisory. It's checked after each call returns, so a
run can overshoot by one call. Prices are hardcoded and need updating when
Anthropic's rates change.

Levers, cheapest first: `ANTHROPIC_MODEL`, `MAX_REVISIONS`,
`RESULTS_PER_QUERY`, `MAX_TOKENS`.

## Tests

```bash
pytest -q
```

Runs against SQLite and never calls Anthropic. `test_pipeline.py` drives the
whole graph with a scripted model, including the retry edge and the budget
abort.

## Known gaps

- Background runs use FastAPI's `BackgroundTasks`, so a restart drops in-flight
  work. The `runs` table is shaped for a real queue but there isn't one.
- No auth.
- `MAX_TOKENS` covers thinking plus visible output on the 5-series models. If the
  planner or checker returns unparseable JSON, that's why.
- The retriever stops at six tool steps and returns partial evidence rather than
  looping; the checker's confidence reflects it.
