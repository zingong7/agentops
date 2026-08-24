import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from app.agents import build_graph
from app.config import get_settings
from app.db import (
    ChatSession,
    Claim,
    Message,
    Report,
    Run,
    Source,
    get_db,
    init_db,
    recall_reports,
    recent_messages,
    session_scope,
)
from app.llm import chat_llm, invoke, text_of

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)

CHAT_SYSTEM = """You are the conversational front end of a research assistant.

You have the conversation so far and, when relevant, findings from research runs
in this session. Answer from those. If the question needs new research, say so
and suggest the exact question to run rather than guessing at an answer.

Keep replies short."""


class SessionIn(BaseModel):
    title: str = "untitled"


class SessionOut(BaseModel):
    id: int
    title: str
    created_at: datetime
    model_config = {"from_attributes": True}


class MessageOut(BaseModel):
    role: str
    content: str
    created_at: datetime
    model_config = {"from_attributes": True}


class ChatIn(BaseModel):
    session_id: int
    message: str = Field(min_length=1)


class ChatOut(BaseModel):
    session_id: int
    reply: str
    latency_ms: int
    used_reports: list[int] = []


class ResearchIn(BaseModel):
    question: str = Field(min_length=5)
    session_id: int | None = None


class RunOut(BaseModel):
    id: int
    session_id: int
    question: str
    status: str
    error: str | None = None
    report_id: int | None = None
    started_at: datetime
    finished_at: datetime | None = None
    model_config = {"from_attributes": True}


class SourceOut(BaseModel):
    rank: int
    title: str
    url: str
    snippet: str
    model_config = {"from_attributes": True}


class ClaimOut(BaseModel):
    text: str
    verdict: str
    note: str
    model_config = {"from_attributes": True}


class ReportOut(BaseModel):
    id: int
    session_id: int
    question: str
    answer: str
    confidence: float
    revisions: int
    created_at: datetime
    sources: list[SourceOut] = []
    claims: list[ClaimOut] = []
    model_config = {"from_attributes": True}


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="AgentOps", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
def healthz(db: DbSession = Depends(get_db)):
    db.execute(select(1))
    return {"status": "ok"}


@app.post("/sessions", response_model=SessionOut, status_code=201)
def create_session(body: SessionIn, db: DbSession = Depends(get_db)):
    session = ChatSession(title=body.title[:200])
    db.add(session)
    db.commit()
    return session


@app.get("/sessions", response_model=list[SessionOut])
def list_sessions(limit: int = 50, db: DbSession = Depends(get_db)):
    return db.scalars(select(ChatSession).order_by(ChatSession.id.desc()).limit(limit)).all()


@app.get("/sessions/{session_id}/messages", response_model=list[MessageOut])
def session_messages(session_id: int, db: DbSession = Depends(get_db)):
    _require_session(db, session_id)
    return db.scalars(
        select(Message).where(Message.session_id == session_id).order_by(Message.id)
    ).all()


@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn, db: DbSession = Depends(get_db)):
    """One model call against session memory. Deliberately not the research
    graph -- this is the interactive path and it stays on a latency budget."""
    _require_session(db, body.session_id)
    started = time.perf_counter()

    history = recent_messages(body.session_id, get_settings().history_turns)
    prior = recall_reports(body.session_id, body.message)

    messages = [SystemMessage(content=CHAT_SYSTEM)]
    if prior:
        findings = "\n\n".join(
            f"Report #{p['report_id']} (confidence {p['confidence']})\n"
            f"Q: {p['question']}\nA: {p['answer']}"
            for p in prior
        )
        messages.append(SystemMessage(content=f"Findings on file:\n\n{findings}"))

    for row in history:
        cls = HumanMessage if row.role == "user" else AIMessage
        messages.append(cls(content=row.content))
    messages.append(HumanMessage(content=body.message))

    reply = text_of(invoke(chat_llm(), messages)).strip()
    latency_ms = int((time.perf_counter() - started) * 1000)

    db.add_all(
        [
            Message(session_id=body.session_id, role="user", content=body.message),
            Message(session_id=body.session_id, role="assistant", content=reply),
        ]
    )
    db.commit()

    return ChatOut(
        session_id=body.session_id,
        reply=reply,
        latency_ms=latency_ms,
        used_reports=[p["report_id"] for p in prior],
    )


@app.post("/research", response_model=RunOut, status_code=202)
def start_research(body: ResearchIn, tasks: BackgroundTasks, db: DbSession = Depends(get_db)):
    """A full run takes minutes, so this queues it and returns a run id to poll."""
    if body.session_id is None:
        session = ChatSession(title=body.question[:200])
        db.add(session)
        db.flush()
        session_id = session.id
    else:
        _require_session(db, body.session_id)
        session_id = body.session_id

    run = Run(session_id=session_id, question=body.question, status="queued")
    db.add(run)
    db.commit()

    tasks.add_task(run_research, run.id)
    return run


@app.get("/runs/{run_id}", response_model=RunOut)
def get_run(run_id: int, db: DbSession = Depends(get_db)):
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return run


@app.get("/reports/{report_id}", response_model=ReportOut)
def get_report(report_id: int, db: DbSession = Depends(get_db)):
    report = db.get(Report, report_id)
    if report is None:
        raise HTTPException(404, "report not found")
    report.sources.sort(key=lambda s: s.rank)
    return report


@app.get("/sessions/{session_id}/reports", response_model=list[ReportOut])
def session_reports(session_id: int, db: DbSession = Depends(get_db)):
    _require_session(db, session_id)
    return db.scalars(
        select(Report).where(Report.session_id == session_id).order_by(Report.id.desc())
    ).all()


def _require_session(db: DbSession, session_id: int):
    session = db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    return session


def run_research(run_id):
    """Background task: run the graph and write the result. Owns its own
    sessions because it outlives the request that queued it."""
    with session_scope() as db:
        run = db.get(Run, run_id)
        if run is None:
            log.error("run %s disappeared before it started", run_id)
            return
        run.status = "running"
        session_id, question = run.session_id, run.question

    try:
        final = build_graph().invoke({"session_id": session_id, "question": question})
    except Exception as exc:
        log.exception("run %s failed", run_id)
        with session_scope() as db:
            run = db.get(Run, run_id)
            run.status = "error"
            run.error = str(exc)[:2000]
            run.finished_at = datetime.now(timezone.utc)
        return

    with session_scope() as db:
        report = Report(
            session_id=session_id,
            question=question,
            answer=final.get("draft", ""),
            confidence=final.get("confidence", 0.0),
            revisions=final.get("revisions", 0),
        )
        db.add(report)
        db.flush()

        db.add_all(
            Source(
                report_id=report.id,
                rank=e["rank"],
                title=e["title"][:300],
                url=e["url"][:500],
                snippet=e["snippet"],
            )
            for e in final.get("evidence", [])
        )
        db.add_all(
            Claim(report_id=report.id, text=c["text"], verdict=c["verdict"], note=c["note"])
            for c in final.get("claims", [])
        )

        run = db.get(Run, run_id)
        run.status = "done"
        run.report_id = report.id
        run.finished_at = datetime.now(timezone.utc)
