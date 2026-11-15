from types import SimpleNamespace

from enterprise.cache import RedisCache
from enterprise.retrieval import Retriever
from enterprise.storage import DocumentStore


class FakeRedis:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key] = value
        return True

    def ping(self):
        return True

    def close(self):
        pass


class CountingVector:
    def __init__(self):
        self.calls = 0

    def search(self, vector, limit):
        self.calls += 1
        return []


class FakeEmbedder:
    def encode(self, texts, query=False):
        return [[0.0, 1.0] for _ in texts]


def _add_doc(store, doc_id, text):
    store.add(
        {"id": doc_id, "filename": f"{doc_id}.md", "sha256": f"sha-{doc_id}", "characters": len(text)},
        [{
            "id": f"chunk-{doc_id}", "doc_id": doc_id, "idx": 0, "text": text,
            "metadata": {"filename": f"{doc_id}.md", "chunk_index": 0, "start_char": 0, "end_char": len(text)},
        }],
    )


def test_retriever_uses_redis_cache_and_database_revision(tmp_path):
    store = DocumentStore(tmp_path / "documents.sqlite3")
    vector = CountingVector()
    redis_client = FakeRedis()
    cache = RedisCache("redis://unused", client=redis_client)
    settings = SimpleNamespace(
        top_k=5, rerank_enabled=False, rerank_model="unused", embedding_cache=tmp_path,
        redis_cache_ttl=300, embedding_backend="hash", embedding_model="test-model",
        vector_backend="sqlite",
    )
    retriever = Retriever(store, vector, FakeEmbedder(), settings, cache=cache)
    try:
        _add_doc(store, "doc-1", "上海差旅住宿标准为每晚四百元")
        first = retriever.search("上海住宿标准", strategy="bm25")
        second = retriever.search("上海住宿标准", strategy="bm25")
        assert first == second
        assert first[0]["doc_id"] == "doc-1"

        # 文档写入会在 MySQL/SQLite 事务中递增版本，旧缓存键不再被读取。
        _add_doc(store, "doc-2", "上海住宿标准需要部门审批")
        third = retriever.search("上海住宿标准", strategy="bm25")
        assert len(third) == 2
    finally:
        cache.close()
        store.close()


def test_redis_cache_falls_back_when_redis_is_unavailable():
    class BrokenRedis:
        def get(self, key):
            raise ConnectionError("redis is down")

        def set(self, key, value, ex=None):
            raise ConnectionError("redis is down")

        def ping(self):
            raise ConnectionError("redis is down")

        def close(self):
            pass

    cache = RedisCache("redis://unused", client=BrokenRedis())
    assert cache.get_json("key") is None
    assert cache.set_json("key", {"ok": True}, 10) is False
    assert cache.ping() is False
