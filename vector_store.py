import chromadb # chroma数据库的Python客户端库，让我们用代码操作chroma的数据库
from chromadb.config import Settings
import hashlib


class ChromaVectorStore:
    """Chroma 向量存储，负责向量的持久化和检索"""

    def __init__(self, persist_directory: str, collection_name: str):
        """
        初始化 Chroma 向量存储

        Args:
            persist_directory: 持久化目录路径，如 "vector_db"
            collection_name: 集合名称，如 "rag_knowledge_base"
        """
        # 创建持久化客户端
        self.client = chromadb.PersistentClient( #这个函数如果原来没有名字的话他会自己生成一个名字的
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)  # 关闭遥测 别把数据发出去
        )

        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度 后续再运行的时候会被忽略掉
        )

    @staticmethod
    def _generate_chunk_id(filename: str, chunk_index: int) -> str:
        """生成稳定的 Chunk ID，格式：文件名::chunk_序号"""
        return f"{filename}::chunk_{chunk_index}"

    def upsert_documents(self, documents: list, embeddings: list) -> int:
        """
        插入或更新文档向量（不会重复插入）

        Args:
            documents: Document 对象列表
            embeddings: 对应的嵌入向量列表

        Returns:
            插入的文档数量
        """
        ids = []
        metadatas = []
        contents = []
        embeddings_list = []

        for doc, embedding in zip(documents, embeddings):
            # 生成唯一 ID
            chunk_id = self._generate_chunk_id(
                doc.metadata.get("filename", "unknown"),
                doc.metadata.get("chunk_index", 0)
            )
            ids.append(chunk_id)

            # 准备 metadata（Chroma 要求值是基本类型：str, int, float, bool）
            metadata = {
                "filename": str(doc.metadata.get("filename", "")),
                "chunk_index": int(doc.metadata.get("chunk_index", 0)),
                "start_char": int(doc.metadata.get("start_char", 0)),
                "end_char": int(doc.metadata.get("end_char", 0))
            }
            metadatas.append(metadata)

            # 文档内容和向量
            contents.append(doc.page_content)
            embeddings_list.append(embedding)

        # 使用 upsert，有则更新，无则插入
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            documents=contents,
            metadatas=metadatas
        )

        return len(ids)

    def search(self, query_embedding: list, top_k: int = 5) -> dict:
        """
        搜索最相似的文档

        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量

        Returns:
            包含 ids, documents, metadatas, distances 的字典
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # Chroma 返回的是嵌套列表，我们取第一层
        return {
            "ids": results["ids"][0],
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0]
        }

    def count(self) -> int:
        """返回集合中的文档总数"""
        return self.collection.count()