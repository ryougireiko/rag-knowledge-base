"""仅从环境变量加载配置，不将任何密钥提交到 Git。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

load_dotenv()  # 加载 .env；默认不覆盖 shell/容器显式传入的环境变量。


def yes(key: str, default: str = "false") -> bool:
    """读取布尔型环境变量。"""
    return os.getenv(key, default).strip().lower() in {"1", "yes", "true", "on"}


def build_database_url(root: Path) -> str:
    """优先使用 DATABASE_URL，其次读取 DB_* 变量，最后回退到本地 SQLite。"""
    explicit_url = os.getenv("DATABASE_URL", "").strip()
    if explicit_url:
        return explicit_url

    host = os.getenv("DB_HOST", "").strip()
    if host:
        user = quote_plus(os.getenv("DB_USER", "rag_user"))
        password = quote_plus(os.getenv("DB_PASSWORD", ""))
        database = quote_plus(os.getenv("DB_NAME", "enterprise_rag"))
        port = int(os.getenv("DB_PORT", "3306"))
        return (
            f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
            "?charset=utf8mb4"
        )

    return f"sqlite+pysqlite:///{(root / 'documents.sqlite3').as_posix()}"


@dataclass(frozen=True)
class Settings:
    """不可变的应用配置项集合。"""

    data_dir: Path
    sqlite_path: Path          # LangGraph Checkpoint 与 SQLite 向量测试后端仍使用该目录
    checkpoint_path: Path
    chroma_path: Path
    embedding_cache: Path
    database_url: str          # 文档元数据数据库：MySQL 或 SQLite
    redis_url: str             # 空字符串表示禁用 Redis 缓存
    redis_cache_ttl: int
    embedding_model: str
    embedding_backend: str
    vector_backend: str
    llm_backend: str
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    api_token: str
    max_upload_mb: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    rerank_enabled: bool
    rerank_model: str
    max_agent_steps: int

    @classmethod
    def from_env(cls) -> "Settings":
        """从环境变量构建配置；本地默认 SQLite，Compose 显式启用 MySQL/Redis。"""
        root = Path(os.getenv("APP_DATA_DIR", "./.local_data")).resolve()
        demo = yes("DEMO_MODE")
        fields = cls(
            data_dir=root,
            sqlite_path=root / "documents.sqlite3",
            checkpoint_path=root / "agent_checkpoints.sqlite3",
            chroma_path=root / "chroma",
            embedding_cache=Path(
                os.getenv("MODEL_CACHE_DIR", str(root / "models"))
            ).resolve(),
            database_url=build_database_url(root),
            redis_url=os.getenv("REDIS_URL", "").strip(),
            redis_cache_ttl=int(os.getenv("REDIS_CACHE_TTL", "300")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            embedding_backend=os.getenv(
                "EMBEDDING_BACKEND", "hash" if demo else "bge"
            ).lower(),
            vector_backend=os.getenv(
                "VECTOR_BACKEND", "sqlite" if demo else "chroma"
            ).lower(),
            llm_backend=os.getenv("LLM_BACKEND", "mock" if demo else "deepseek").lower(),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_token=os.getenv("APP_API_TOKEN", ""),
            max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "8")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "450")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "60")),
            top_k=int(os.getenv("TOP_K", "5")),
            rerank_enabled=yes("RERANK_ENABLED"),
            rerank_model=os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base"),
            max_agent_steps=int(os.getenv("MAX_AGENT_STEPS", "12")),
        )

        if fields.chunk_size <= 0 or not (0 <= fields.chunk_overlap < fields.chunk_size):
            raise ValueError("Invalid CHUNK_SIZE/CHUNK_OVERLAP")
        if fields.embedding_backend not in {"hash", "bge"}:
            raise ValueError("Invalid EMBEDDING_BACKEND")
        if fields.vector_backend not in {"sqlite", "chroma"}:
            raise ValueError("Invalid VECTOR_BACKEND")
        if fields.llm_backend not in {"mock", "deepseek"}:
            raise ValueError("Invalid LLM_BACKEND")
        if fields.max_upload_mb <= 0 or not (1 <= fields.top_k <= 20):
            raise ValueError("Invalid upload/query limits")
        if fields.redis_cache_ttl <= 0:
            raise ValueError("REDIS_CACHE_TTL must be greater than zero")

        fields.data_dir.mkdir(parents=True, exist_ok=True)
        return fields
