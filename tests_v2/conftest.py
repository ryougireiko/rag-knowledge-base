import pytest

from enterprise.config import Settings
from enterprise.service import Services


@pytest.fixture
def services(monkeypatch, tmp_path):
    """测试显式锁定轻量后端，避免被开发机 .env 中的真实配置污染。"""
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+pysqlite:///{tmp_path / 'documents.sqlite3'}")
    monkeypatch.setenv("REDIS_URL", "")
    monkeypatch.setenv("EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("VECTOR_BACKEND", "sqlite")
    monkeypatch.setenv("LLM_BACKEND", "mock")
    monkeypatch.setenv("APP_API_TOKEN", "unit-test-secret")
    monkeypatch.setenv("RERANK_ENABLED", "false")

    config = Settings.from_env()
    svc = Services(config)
    yield svc
    svc.close()
