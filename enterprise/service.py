"""应用服务装配：统一管理配置、数据库、缓存、向量索引和 AI 服务。"""
from __future__ import annotations

from enterprise.cache import make_cache
from enterprise.config import Settings
from enterprise.embedding import make_embedder
from enterprise.ingest import IngestionService
from enterprise.llm import LLM
from enterprise.rag import RAGService
from enterprise.retrieval import Retriever
from enterprise.storage import DocumentStore
from enterprise.vector_index import make_vector


class Services:
    """服务聚合容器：应用生命周期内只创建一组共享服务。"""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.store = DocumentStore(self.settings.database_url)
        self.cache = make_cache(self.settings)
        self.vector = make_vector(self.settings)
        self.embedder = make_embedder(self.settings)
        self.ingest = IngestionService(
            self.store, self.vector, self.embedder, self.settings
        )
        self.retriever = Retriever(
            self.store, self.vector, self.embedder, self.settings, cache=self.cache
        )
        self.llm = LLM(self.settings)
        self.rag = RAGService(self.retriever, self.llm)

    def health(self) -> dict:
        """返回数据库和缓存的健康状态，不暴露连接串或凭据。"""
        database_ok = self.store.ping()
        if not self.cache.enabled:
            cache_status = "disabled"
        else:
            cache_status = "ok" if self.cache.ping() else "unavailable"
        return {
            "status": "ok" if database_ok else "degraded",
            "database": "ok" if database_ok else "unavailable",
            "cache": cache_status,
        }

    def close(self) -> None:
        """按依赖顺序释放缓存、向量索引和数据库连接。"""
        self.cache.close()
        self.vector.close()
        self.store.close()
