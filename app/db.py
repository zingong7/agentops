"""Engine, tables, and the queries the agents use to read session memory."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.orm import Session as DbSession

from app.config import get_settings
from app.search import tokenize, truncate


class Base(DeclarativeBase):
    pass


class ChatSession(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200), default="untitled")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    messages: Mapped[list["Message"]] = relationship(back_populates="session", cascade="all, delete")
    reports: Mapped[list["Report"]] = relationship(back_populates="session", cascade="all, delete")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped[ChatSession] = relationship(back_populates="messages")


class Run(Base):
    """One execution of the graph. Separate from Report so a failed run still
    leaves a trace with the error on it."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    question: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|done|error
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_id: Mapped[int | None] = mapped_column(ForeignKey("reports.id"), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    revisions: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped[ChatSession] = relationship(back_populates="reports")
    sources: Mapped[list["Source"]] = relationship(back_populates="report", cascade="all, delete")
    claims: Mapped[list["Claim"]] = relationship(back_populates="report", cascade="all, delete")


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    report_id: Mapped[int] = mapped_column(ForeignKey("reports.id", ondelete="CASCADE"), index=True)
    rank: Mapped[int] = mapped_column(Integer)  # the [n] used in the answer text
    title: Mapped[str] = mapped_column(String(300))
    url: Mapped[str] = mapped_column(String(500))
    snippet: Mapped[str] = mapped_column(Text)

    report: Mapped[Report] = relationship(back_populates="sources")


class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[int] = mapped_column(primary_key=True)
    report_id: Mapped[int] = mapped_column(ForeignKey("reports.id", ondelete="CASCADE"), index=True)
    text: Mapped[str] = mapped_column(Text)
    verdict: Mapped[str] = mapped_column(String(16))  # supported | unsupported | contradicted
    note: Mapped[str] = mapped_column(Text, default="")

    report: Mapped[Report] = relationship(back_populates="claims")


_engine = create_engine(get_settings().database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(_engine)


def get_db() -> Iterator[DbSession]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[DbSession]:
    """For background work that runs outside a request."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def recall_reports(session_id: int, query: str, limit: int = 3) -> list[dict]:
    """Prior reports from this session that overlap the query, so a follow-up
    doesn't re-research what we already answered."""
    terms = set(tokenize(query))
    with session_scope() as db:
        rows = db.scalars(
            select(Report).where(Report.session_id == session_id).order_by(Report.id.desc()).limit(25)
        ).all()

    hits = []
    for r in rows:
        overlap = len(terms & set(tokenize(f"{r.question} {r.answer}")))
        if overlap:
            hits.append((overlap, r))

    hits.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {
            "report_id": r.id,
            "question": r.question,
            "answer": truncate(r.answer, 1200),
            "confidence": r.confidence,
        }
        for _, r in hits[:limit]
    ]


def recent_messages(session_id: int, limit: int) -> list[Message]:
    with session_scope() as db:
        rows = db.scalars(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.id.desc())
            .limit(limit)
        ).all()
    return list(reversed(rows))
