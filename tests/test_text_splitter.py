"""测试文本切片功能。"""

import pytest
from text_splitter import split_document, split_documents
from document import Document


class TestSplitDocument:
    """测试单文档切片。"""

    @pytest.fixture
    def short_doc(self):
        return Document(
            page_content="短文本，不到100字符。",
            metadata={"filename": "test.txt"}
        )

    @pytest.fixture
    def long_doc(self):
        return Document(
            page_content="测试内容。" * 200,  # 约1200字符
            metadata={"filename": "long.txt"}
        )

    def test_short_single_chunk(self, short_doc):
        """短文本只产生1个Chunk"""
        chunks = split_document(short_doc, chunk_size=100, chunk_overlap=20)
        assert len(chunks) == 1

    def test_long_multiple_chunks(self, long_doc):
        """长文本产生多个Chunk"""
        chunks = split_document(long_doc, chunk_size=200, chunk_overlap=50)
        assert len(chunks) > 1

    def test_overlap(self, long_doc):
        """相邻Chunk有重叠"""
        chunks = split_document(long_doc, chunk_size=200, chunk_overlap=50)
        if len(chunks) > 1:
            first_end = chunks[0].metadata["end_char"]
            second_start = chunks[1].metadata["start_char"]
            assert second_start < first_end

    def test_last_chunk_not_lost(self, long_doc):
        """最后一段到达文本末尾"""
        chunks = split_document(long_doc, chunk_size=200, chunk_overlap=50)
        assert chunks[-1].metadata["end_char"] == len(long_doc.page_content)

    def test_chunk_index_continuous(self, long_doc):
        """chunk_index连续递增"""
        chunks = split_document(long_doc, chunk_size=200, chunk_overlap=50)
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i

    def test_metadata_preserved(self, long_doc):
        """原始metadata被保留"""
        chunks = split_document(long_doc, chunk_size=200, chunk_overlap=50)
        for chunk in chunks:
            assert chunk.metadata["filename"] == "long.txt"

    def test_start_end_char(self, long_doc):
        """start_char和end_char正确"""
        chunks = split_document(long_doc, chunk_size=200, chunk_overlap=50)
        for chunk in chunks:
            s = chunk.metadata["start_char"]
            e = chunk.metadata["end_char"]
            assert 0 <= s < e <= len(long_doc.page_content)
            assert len(chunk.page_content) == e - s

    def test_invalid_chunk_size(self, long_doc):
        """chunk_size<=0抛异常"""
        with pytest.raises(ValueError):
            split_document(long_doc, chunk_size=0, chunk_overlap=10)
        with pytest.raises(ValueError):
            split_document(long_doc, chunk_size=-1, chunk_overlap=10)

    def test_invalid_overlap(self, long_doc):
        """overlap非法抛异常"""
        with pytest.raises(ValueError):
            split_document(long_doc, chunk_size=100, chunk_overlap=-1)
        with pytest.raises(ValueError):
            split_document(long_doc, chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError):
            split_document(long_doc, chunk_size=100, chunk_overlap=150)

    def test_empty_text(self):
        """空文本返回空列表"""
        doc = Document(page_content="", metadata={})
        chunks = split_document(doc, chunk_size=100, chunk_overlap=20)
        assert chunks == []

    def test_no_infinite_loop(self):
        """不会死循环"""
        doc = Document(page_content="a" * 10, metadata={})
        chunks = split_document(doc, chunk_size=3, chunk_overlap=2)
        assert len(chunks) > 0


class TestSplitDocuments:
    """测试多文档切片。"""

    def test_multiple_docs(self):
        """多个文档都切片，来源文件名正确"""
        docs = [
            Document(page_content="A" * 500, metadata={"filename": "a.txt"}),
            Document(page_content="B" * 500, metadata={"filename": "b.txt"}),
        ]
        chunks = split_documents(docs, chunk_size=200, chunk_overlap=50)
        filenames = {c.metadata["filename"] for c in chunks}
        assert filenames == {"a.txt", "b.txt"}