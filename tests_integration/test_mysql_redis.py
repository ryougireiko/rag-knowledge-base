"""真实 MySQL / Redis 集成测试；本地未配置服务时自动跳过。"""
from __future__ import annotations

import os
from uuid import uuid4

import pytest

from enterprise.cache import RedisCache
from enterprise.storage import DocumentStore

DATABASE_URL = os.getenv("TEST_DATABASE_URL", "").strip()
REDIS_URL = os.getenv("TEST_REDIS_URL", "").strip()

pytestmark = pytest.mark.skipif(
    not DATABASE_URL or not REDIS_URL,
    reason="需要 TEST_DATABASE_URL 和 TEST_REDIS_URL 指向真实 MySQL / Redis 服务",
)


def test_mysql_transaction_revision_and_redis_json_roundtrip():
    store = DocumentStore(DATABASE_URL)
    cache = RedisCache(REDIS_URL)
    suffix = uuid4().hex
    doc_id = suffix
    cache_key = f"integration:test:{suffix}"
    try:
        assert store.ping() is True
        assert cache.ping() is True

        before = store.revision()
        store.add(
            {
                "id": doc_id,
                "filename": "integration.md",
                "sha256": suffix + suffix,
                "characters": 8,
            },
            [
                {
                    "id": suffix,
                    "doc_id": doc_id,
                    "idx": 0,
                    "text": "集成测试文本",
                    "metadata": {"filename": "integration.md", "start_char": 0, "end_char": 6},
                }
            ],
        )
        assert store.revision() == before + 1
        assert store.get_doc_by_sha(suffix + suffix)["id"] == doc_id
        chunks_by_id = {chunk["id"]: chunk for chunk in store.chunks()}
        assert chunks_by_id[suffix]["metadata"]["filename"] == "integration.md"

        payload = [{"doc_id": doc_id, "score": 0.9}]
        assert cache.set_json(cache_key, payload, ttl=30) is True
        assert cache.get_json(cache_key) == payload

        assert store.delete(doc_id) is True
        assert store.revision() == before + 2
    finally:
        try:
            cache.client.delete(cache_key)
        finally:
            cache.close()
            store.close()
