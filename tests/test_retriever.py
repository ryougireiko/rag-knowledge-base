"""测试 retriever 模块：余弦相似度 + Top-K 检索。

完全不依赖真实模型，使用手写 numpy 向量测试。
"""

import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from retriever import cosine_similarity, retrieve_top_k, SearchResult
from document import Document


# ============================================
# 测试数据准备
# ============================================

@pytest.fixture
def sample_documents():
    """创建 3 个测试用 Document"""
    return [
        Document(
            page_content="RAG 技术介绍",
            metadata={"filename": "doc1.txt", "chunk_index": 0}
        ),
        Document(
            page_content="Embedding 向量表示",
            metadata={"filename": "doc2.txt", "chunk_index": 1}
        ),
        Document(
            page_content="意大利披萨制作方法",
            metadata={"filename": "doc3.txt", "chunk_index": 2}
        ),
    ]


@pytest.fixture
def sample_vectors():
    """创建 3 个文档向量（2维，方便手算验证）"""
    return np.array([
        [1.0, 0.0],   # 文档1：沿 x 轴
        [0.0, 1.0],   # 文档2：沿 y 轴
        [-1.0, 0.0],  # 文档3：沿 -x 轴
    ])


# ============================================
# 余弦相似度测试
# ============================================

class TestCosineSimilarity:
    """测试 cosine_similarity 函数"""

    def test_same_vector(self):
        """相同向量相似度应为 1"""
        query = np.array([1.0, 0.0])
        docs = np.array([[1.0, 0.0]])
        result = cosine_similarity(query, docs)
        assert np.isclose(result[0], 1.0)

    def test_orthogonal_vectors(self):
        """垂直向量相似度应为 0"""
        query = np.array([1.0, 0.0])
        docs = np.array([[0.0, 1.0]])
        result = cosine_similarity(query, docs)
        assert np.isclose(result[0], 0.0)

    def test_opposite_vectors(self):
        """相反向量相似度应为 -1"""
        query = np.array([1.0, 0.0])
        docs = np.array([[-1.0, 0.0]])
        result = cosine_similarity(query, docs)
        assert np.isclose(result[0], -1.0)

    def test_non_normalized_vectors(self):
        """非归一化向量也能正确计算"""
        query = np.array([3.0, 4.0])    # 模长 = 5
        docs = np.array([[6.0, 8.0]])   # 模长 = 10，方向相同
        result = cosine_similarity(query, docs)
        assert np.isclose(result[0], 1.0)

    def test_multiple_documents(self):
        """多个文档返回正确形状"""
        query = np.array([1.0, 0.0])
        docs = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        result = cosine_similarity(query, docs)
        assert result.shape == (3,)
        assert np.isclose(result[0], 1.0)
        assert np.isclose(result[1], 0.0)
        assert np.isclose(result[2], -1.0)

    def test_query_not_1d_raises(self):
        """查询向量不是一维时报错"""
        query = np.array([[1.0, 0.0]])  # (1, 2) 不是 (2,)
        docs = np.array([[1.0, 0.0]])
        with pytest.raises(ValueError, match="一维"):
            cosine_similarity(query, docs)

    def test_docs_not_2d_raises(self):
        """文档向量不是二维时报错"""
        query = np.array([1.0, 0.0])
        docs = np.array([1.0, 0.0])  # (2,) 不是 (n, 2)
        with pytest.raises(ValueError, match="二维"):
            cosine_similarity(query, docs)

    def test_dimension_mismatch_raises(self):
        """向量维度不一致时报错"""
        query = np.array([1.0, 0.0, 0.0])       # 3维
        docs = np.array([[1.0, 0.0]])            # 2维
        with pytest.raises(ValueError, match="维度不一致"):
            cosine_similarity(query, docs)

    def test_zero_query_raises(self):
        """零查询向量时报错"""
        query = np.array([0.0, 0.0])
        docs = np.array([[1.0, 0.0]])
        with pytest.raises(ValueError, match="零向量"):
            cosine_similarity(query, docs)

    def test_zero_document_not_crash(self):
        """零文档向量不会导致除零崩溃"""
        query = np.array([1.0, 0.0])
        docs = np.array([[0.0, 0.0]])
        # 不应抛出异常
        result = cosine_similarity(query, docs)
        assert result[0] == 0.0


# ============================================
# retrieve_top_k 测试
# ============================================

class TestRetrieveTopK:
    """测试 retrieve_top_k 函数"""

    def test_basic_retrieval(self, sample_documents, sample_vectors):
        """基本检索：查询 [1,0]，最相似的应该是文档1"""
        query = np.array([1.0, 0.0])
        results = retrieve_top_k(query, sample_documents, sample_vectors, top_k=3)

        assert len(results) == 3
        # 第一个应该和查询方向一致
        assert results[0].score == 1.0
        assert results[0].document.metadata["filename"] == "doc1.txt"

    def test_descending_order(self, sample_documents, sample_vectors):
        """分数从高到低排列"""
        query = np.array([1.0, 0.0])
        results = retrieve_top_k(query, sample_documents, sample_vectors, top_k=3)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_smaller_than_total(self, sample_documents, sample_vectors):
        """top_k 小于文档总数时只返回 K 个"""
        query = np.array([1.0, 0.0])
        results = retrieve_top_k(query, sample_documents, sample_vectors, top_k=2)

        assert len(results) == 2
        # 应该是最相似的两个
        assert results[0].score == 1.0
        assert results[1].score == 0.0

    def test_top_k_larger_than_total(self, sample_documents, sample_vectors):
        """top_k 大于文档数时返回全部"""
        query = np.array([1.0, 0.0])
        results = retrieve_top_k(query, sample_documents, sample_vectors, top_k=10)

        assert len(results) == 3  # 总共只有 3 个文档

    def test_rank_starts_from_one(self, sample_documents, sample_vectors):
        """rank 从 1 开始"""
        query = np.array([1.0, 0.0])
        results = retrieve_top_k(query, sample_documents, sample_vectors, top_k=3)

        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[2].rank == 3

    def test_empty_documents(self):
        """空文档返回空列表"""
        query = np.array([1.0, 0.0])
        results = retrieve_top_k(query, [], np.array([]).reshape(0, 2), top_k=3)
        assert results == []

    def test_top_k_invalid_raises(self, sample_documents, sample_vectors):
        """top_k 非法时报错"""
        query = np.array([1.0, 0.0])
        with pytest.raises(ValueError, match="top_k 必须大于 0"):
            retrieve_top_k(query, sample_documents, sample_vectors, top_k=0)

    def test_mismatched_counts_raises(self, sample_documents, sample_vectors):
        """文档数量和向量数量不一致时报错"""
        query = np.array([1.0, 0.0])
        # 只给 2 个向量但有 3 个文档
        with pytest.raises(ValueError, match="不一致"):
            retrieve_top_k(query, sample_documents, sample_vectors[:2], top_k=3)

    def test_metadata_preserved(self, sample_documents, sample_vectors):
        """Metadata 被保留"""
        query = np.array([1.0, 0.0])
        results = retrieve_top_k(query, sample_documents, sample_vectors, top_k=3)

        assert results[0].document.metadata["filename"] == "doc1.txt"
        assert results[0].document.metadata["chunk_index"] == 0

    def test_original_document_not_modified(self, sample_documents, sample_vectors):
        """原始 Document 没有被修改"""
        original_content = sample_documents[0].page_content
        original_metadata = sample_documents[0].metadata.copy()

        query = np.array([1.0, 0.0])
        retrieve_top_k(query, sample_documents, sample_vectors, top_k=3)

        assert sample_documents[0].page_content == original_content
        assert sample_documents[0].metadata == original_metadata

    def test_result_is_search_result(self, sample_documents, sample_vectors):
        """返回的是 SearchResult 对象"""
        query = np.array([1.0, 0.0])
        results = retrieve_top_k(query, sample_documents, sample_vectors, top_k=1)

        assert isinstance(results[0], SearchResult)
        assert hasattr(results[0], 'document')
        assert hasattr(results[0], 'score')
        assert hasattr(results[0], 'rank')

    def test_stable_sort_equal_scores(self):
        """相同分数时排序稳定"""
        docs = [
            Document(page_content="A", metadata={}),
            Document(page_content="B", metadata={}),
        ]
        vectors = np.array([[1.0, 0.0], [1.0, 0.0]])  # 完全相同

        query = np.array([1.0, 0.0])
        results = retrieve_top_k(query, docs, vectors, top_k=2)

        # 两个分数应该相等
        assert np.isclose(results[0].score, results[1].score)
        # 原始顺序保持（稳定排序）
        assert results[0].document.page_content == "A"
        assert results[1].document.page_content == "B"