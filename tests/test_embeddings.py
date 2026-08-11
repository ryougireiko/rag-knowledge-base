"""测试 embeddings 模块。

使用 Fake 模型，不下载真实模型。
"""

import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from embeddings import EmbeddingModel
from document import Document


# ============================================
# Fake 模型
# ============================================

class FakeSentenceTransformer:
    """模拟 SentenceTransformer，返回固定向量"""

    def __init__(self, model_name=None):
        self.model_name = model_name
        self.encode_calls = []  # 记录调用历史

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        """返回固定形状的向量"""
        self.encode_calls.append(texts)

        if isinstance(texts, list):
            # 批量编码：每个文本返回 4 维向量
            return np.array([[0.1, 0.2, 0.3, 0.4]] * len(texts))
        else:
            # 单个文本：返回一维向量
            return np.array([0.1, 0.2, 0.3, 0.4])


# ============================================
# 测试数据
# ============================================

@pytest.fixture
def fake_model():
    """创建 Fake 模型"""
    return FakeSentenceTransformer()


@pytest.fixture
def embedding_model(fake_model):
    """创建使用 Fake 模型的 EmbeddingModel"""
    return EmbeddingModel(
        model_name="fake-model",
        query_instruction="测试前缀：",
        model=fake_model
    )


@pytest.fixture
def sample_documents():
    """创建测试文档"""
    return [
        Document(page_content="文档1内容", metadata={"index": 0}),
        Document(page_content="文档2内容", metadata={"index": 1}),
        Document(page_content="文档3内容", metadata={"index": 2}),
    ]


# ============================================
# 测试
# ============================================

class TestEmbeddingModel:
    """测试 EmbeddingModel"""

    def test_init_with_real_model_name(self):
        """使用模型名初始化（不实际加载）"""
        # 注入 fake 模型避免下载
        model = EmbeddingModel(
            model_name="BAAI/bge-small-zh-v1.5",
            model=FakeSentenceTransformer()
        )
        assert model.model_name == "BAAI/bge-small-zh-v1.5"

    def test_embed_documents_returns_2d_array(self, embedding_model, sample_documents):
        """多个文档返回二维数组"""
        result = embedding_model.embed_documents(sample_documents)
        assert result.ndim == 2
        assert result.shape == (3, 4)  # 3个文档，每个4维

    def test_embed_documents_row_count(self, embedding_model, sample_documents):
        """返回行数等于文档数量"""
        result = embedding_model.embed_documents(sample_documents)
        assert result.shape[0] == len(sample_documents)

    def test_embed_query_returns_1d_array(self, embedding_model):
        """embed_query 返回一维数组"""
        result = embedding_model.embed_query("测试问题")
        assert result.ndim == 1
        assert result.shape == (4,)

    def test_query_instruction_added(self, fake_model):
        """查询指令被正确添加"""
        model = EmbeddingModel(
            model_name="test",
            query_instruction="前缀：",
            model=fake_model
        )
        model.embed_query("问题")
        # 检查传给 encode 的文本是否加了前缀
        assert "前缀：问题" in fake_model.encode_calls[-1]

    def test_documents_no_query_instruction(self, fake_model, sample_documents):
        """文档没有添加查询指令"""
        model = EmbeddingModel(
            model_name="test",
            query_instruction="前缀：",
            model=fake_model
        )
        model.embed_documents(sample_documents)
        # 文档文本不应该有前缀
        for text in fake_model.encode_calls[-1]:
            assert not text.startswith("前缀：")

    def test_empty_query_raises(self, embedding_model):
        """空查询报错"""
        with pytest.raises(ValueError, match="不能为空"):
            embedding_model.embed_query("")

        with pytest.raises(ValueError, match="不能为空"):
            embedding_model.embed_query("   ")

    def test_empty_documents_returns_empty_array(self, embedding_model):
        """空文档列表返回空数组"""
        result = embedding_model.embed_documents([])
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_model_not_recreated(self, fake_model, sample_documents):
        """模型对象没有在每次调用时重新创建"""
        model = EmbeddingModel(model_name="test", model=fake_model)

        # 多次调用
        model.embed_documents(sample_documents)
        model.embed_query("问题1")
        model.embed_query("问题2")

        # Fake 模型记录了 3 次调用（说明用的是同一个模型对象）
        assert len(fake_model.encode_calls) == 3