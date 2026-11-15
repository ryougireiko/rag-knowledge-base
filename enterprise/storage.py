"""基于 SQLAlchemy 的文档元数据存储。

支持 MySQL（Docker/部署环境）和 SQLite（本地离线演示、自动化测试）。
Chroma 仍然只负责向量索引；Agent Checkpoint 仍由 LangGraph SQLite Saver 管理。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    create_engine,
    delete,
    event,
    func,
    select,
    update,
)
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

log = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy ORM 基类。"""


class DocumentRecord(Base):
    """原始文档元数据。"""

    __tablename__ = "docs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    filename: Mapped[str] = mapped_column(String(180), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    characters: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )


class ChunkRecord(Base):
    """文档切分片段；metadata_json 映射到数据库列 metadata。"""

    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_doc_id", "doc_id"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("docs.id", ondelete="CASCADE"), nullable=False
    )
    idx: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False)


class AppStateRecord(Base):
    """存放轻量级应用状态，例如检索数据版本号。"""

    __tablename__ = "app_state"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)


class DocumentStore:
    """文档登记库：提供 docs/chunks CRUD、事务和检索版本号。"""

    REVISION_KEY = "retrieval_revision"

    def __init__(self, database_url: str | Path):
        # 兼容旧版 DocumentStore(Path(...)) 调用方式。
        if isinstance(database_url, Path):
            database_url = URL.create(
                "sqlite+pysqlite", database=str(database_url.resolve())
            ).render_as_string(hide_password=False)

        self.database_url = str(database_url)
        engine_options: dict[str, Any] = {"pool_pre_ping": True, "future": True}
        if self.database_url.startswith("sqlite"):
            engine_options["connect_args"] = {"check_same_thread": False}
        else:
            # MySQL 连接池：连接取出前先 ping，空闲连接定期回收。
            engine_options.update(pool_size=5, max_overflow=10, pool_recycle=1800)

        self.engine = create_engine(self.database_url, **engine_options)
        if self.engine.dialect.name == "sqlite":
            @event.listens_for(self.engine, "connect")
            def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)
        Base.metadata.create_all(self.engine)
        self._ensure_revision_record()

    def _ensure_revision_record(self) -> None:
        """并发安全地初始化版本号记录，不覆盖已有版本。"""
        table = AppStateRecord.__table__
        if self.engine.dialect.name == "mysql":
            from sqlalchemy.dialects.mysql import insert as dialect_insert

            statement = dialect_insert(table).values(key=self.REVISION_KEY, value=0)
            statement = statement.on_duplicate_key_update(value=table.c.value)
            with self.engine.begin() as connection:
                connection.execute(statement)
        elif self.engine.dialect.name == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as dialect_insert

            statement = dialect_insert(table).values(key=self.REVISION_KEY, value=0)
            statement = statement.on_conflict_do_nothing(index_elements=[table.c.key])
            with self.engine.begin() as connection:
                connection.execute(statement)
        else:
            # 其他 SQLAlchemy 方言使用通用 ORM 路径；项目默认只部署 MySQL / SQLite。
            with self.Session.begin() as session:
                if session.get(AppStateRecord, self.REVISION_KEY) is None:
                    session.add(AppStateRecord(key=self.REVISION_KEY, value=0))

    def _increment_revision(self, session) -> None:
        """在文档变更事务内原子递增版本，避免 Redis 返回旧检索结果。"""
        session.execute(
            update(AppStateRecord)
            .where(AppStateRecord.key == self.REVISION_KEY)
            .values(value=AppStateRecord.value + 1)
        )

    @staticmethod
    def _document_dict(record: DocumentRecord) -> dict[str, Any]:
        created_at = record.created_at
        return {
            "id": record.id,
            "filename": record.filename,
            "sha256": record.sha256,
            "characters": record.characters,
            "created_at": created_at.isoformat(sep=" ") if created_at else None,
        }

    def get_doc_by_sha(self, sha: str) -> dict[str, Any] | None:
        """按内容哈希查询文档，用于上传去重。"""
        with self.Session() as session:
            record = session.scalar(select(DocumentRecord).where(DocumentRecord.sha256 == sha))
            return self._document_dict(record) if record else None

    def add(self, doc: dict, chunks: list[dict]) -> None:
        """在一个数据库事务中插入文档及全部片段，并递增检索版本。"""
        created_at = doc.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

        with self.Session.begin() as session:
            session.add(
                DocumentRecord(
                    id=doc["id"],
                    filename=doc["filename"],
                    sha256=doc["sha256"],
                    characters=doc["characters"],
                    **({"created_at": created_at} if created_at else {}),
                )
            )
            # 先写入父表，再写入带外键的子表，确保跨数据库顺序一致。
            session.flush()
            session.add_all(
                [
                    ChunkRecord(
                        id=chunk["id"],
                        doc_id=chunk["doc_id"],
                        idx=chunk["idx"],
                        text=chunk["text"],
                        metadata_json=chunk["metadata"],
                    )
                    for chunk in chunks
                ]
            )
            self._increment_revision(session)

    def docs(self) -> list[dict[str, Any]]:
        """列出文档及片段数量，按创建时间倒序。"""
        chunk_counts = (
            select(ChunkRecord.doc_id, func.count(ChunkRecord.id).label("chunk_count"))
            .group_by(ChunkRecord.doc_id)
            .subquery()
        )
        statement = (
            select(DocumentRecord, func.coalesce(chunk_counts.c.chunk_count, 0))
            .outerjoin(chunk_counts, chunk_counts.c.doc_id == DocumentRecord.id)
            .order_by(DocumentRecord.created_at.desc())
        )
        with self.Session() as session:
            rows = session.execute(statement).all()
            return [
                {**self._document_dict(record), "chunks": int(count)}
                for record, count in rows
            ]

    def chunks(self) -> list[dict[str, Any]]:
        """读取全部片段，并返回与旧 SQLite 实现兼容的字典结构。"""
        with self.Session() as session:
            records = session.scalars(
                select(ChunkRecord).order_by(ChunkRecord.doc_id, ChunkRecord.idx)
            ).all()
            return [
                {
                    "id": record.id,
                    "doc_id": record.doc_id,
                    "idx": record.idx,
                    "text": record.text,
                    "metadata": record.metadata_json or {},
                }
                for record in records
            ]

    def delete(self, doc_id: str) -> bool:
        """删除文档及其片段；删除和版本递增处于同一事务。"""
        with self.Session.begin() as session:
            record = session.get(DocumentRecord, doc_id)
            if record is None:
                return False
            # 显式删除片段，避免不同数据库对 ORM 级联行为的差异。
            session.execute(delete(ChunkRecord).where(ChunkRecord.doc_id == doc_id))
            session.delete(record)
            self._increment_revision(session)
            return True

    def touch_revision(self) -> int:
        """在向量索引重建或同步完成后递增版本，使旧检索缓存自然失效。"""
        with self.Session.begin() as session:
            self._increment_revision(session)
        return self.revision()

    def revision(self) -> int:
        """获取文档数据版本；缓存键使用它来隔离文档变更前后的结果。"""
        with self.Session() as session:
            value = session.scalar(
                select(AppStateRecord.value).where(AppStateRecord.key == self.REVISION_KEY)
            )
            return int(value or 0)

    def ping(self) -> bool:
        """检查数据库连接是否可用。"""
        try:
            with self.engine.connect() as connection:
                connection.exec_driver_sql("SELECT 1")
            return True
        except Exception:
            log.exception("数据库健康检查失败")
            return False

    def close(self) -> None:
        """释放 SQLAlchemy 连接池。"""
        self.engine.dispose()
