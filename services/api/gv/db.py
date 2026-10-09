"""Structured project database (SQLAlchemy).

Works on SQLite out of the box and on PostgreSQL/PostGIS via DATABASE_URL.
Uploaded PDFs are kept on disk under GV_DATA_DIR; every extracted value keeps
a pointer back to (document, page, bounding box).
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

DATA_DIR = Path(os.environ.get("GV_DATA_DIR", "./data")).resolve()
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DATA_DIR / 'groundvision.db'}")


def now():
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Base(DeclarativeBase):
    pass


class ProjectRow(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    model: Mapped[dict] = mapped_column(JSON)  # gv.models.Project as JSON
    overrides: Mapped[dict] = mapped_column(JSON, default=dict)  # per-sheet calibration + datum
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)


class DocumentRow(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(300))
    path: Mapped[str] = mapped_column(Text)
    pages: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="queued")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    vision_used: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class SheetReportRow(Base):
    """Raw per-page extraction, kept so the project can be re-assembled after calibration changes."""

    __tablename__ = "sheet_reports"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    page: Mapped[int] = mapped_column(Integer)
    sheet_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    category: Mapped[str] = mapped_column(String(40))
    report: Mapped[dict] = mapped_column(JSON)


class EditRow(Base):
    """Every confirmation or correction a person makes, replayed after re-assembly."""

    __tablename__ = "edits"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    edit: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "uploads").mkdir(exist_ok=True)
engine = create_engine(DATABASE_URL, future=True, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(engine, expire_on_commit=False)


def init_db():
    Base.metadata.create_all(engine)


def get_project_row(db: Session, pid: str) -> ProjectRow | None:
    return db.get(ProjectRow, pid)


def documents_for(db: Session, pid: str) -> list[DocumentRow]:
    return list(db.scalars(select(DocumentRow).where(DocumentRow.project_id == pid).order_by(DocumentRow.created_at)))


def reports_for(db: Session, pid: str) -> list[tuple[str, dict]]:
    rows = db.execute(select(SheetReportRow).join(DocumentRow).where(DocumentRow.project_id == pid).order_by(SheetReportRow.document_id, SheetReportRow.page))
    return [(r.document_id, r.report) for (r,) in rows]


def edits_for(db: Session, pid: str) -> list[dict]:
    return [r.edit for r in db.scalars(select(EditRow).where(EditRow.project_id == pid).order_by(EditRow.id))]
