"""应用服务在进程中仅实例化一次。"""
from __future__ import annotations
from enterprise.config import Settings
from enterprise.storage import DocumentStore
from enterprise.vector_index import make_vector
from enterprise.embedding import make_embedder
from enterprise.ingest import IngestionService
from enterprise.retrieval import Retriever
from enterprise.llm import LLM
from enterprise.rag import RAGService


class Services:
    """服务聚合容器：统一装配配置、存储、检索、模型等各组件。"""
    def __init__(self, settings: Settings | None = None):
        # 未传入配置时从环境变量加载
        self.settings = settings or Settings.from_env()
        self.store = DocumentStore(self.settings.sqlite_path)  # 文档登记库
        self.vector = make_vector(self.settings)  # 向量索引后端
        self.embedder = make_embedder(self.settings)  # 嵌入后端
        self.ingest = IngestionService(self.store,self.vector,self.embedder,self.settings)  # 入库服务
        self.retriever = Retriever(self.store,self.vector,self.embedder,self.settings)  # 检索器
        self.llm = LLM(self.settings)  # 大模型封装
        self.rag = RAGService(self.retriever,self.llm)  # RAG 服务

    def close(self):
        """释放资源：关闭向量索引与文档存储连接。"""
        self.vector.close()
        self.store.close()