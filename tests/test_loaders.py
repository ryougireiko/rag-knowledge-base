"""测试文档加载功能。"""

import os
import tempfile
import pytest
from loaders import load_text_file, load_documents, clean_text
from document import Document


class TestCleanText:
    """测试文本清洗。"""

    def test_unify_line_endings(self):
        """Windows换行符统一为\n"""
        text = "line1\r\nline2\rline3"
        result = clean_text(text)
        assert "\r\n" not in result
        assert "\r" not in result

    def test_strip_whitespace(self):
        """去除首尾空白"""
        text = "  \n  hello  \n  "
        result = clean_text(text)
        assert result == "hello"

    def test_compress_newlines(self):
        """3个以上空行压缩为2个"""
        text = "a\n\n\n\nb"
        result = clean_text(text)
        assert result == "a\n\nb"


class TestLoadTextFile:
    """测试单文件加载。"""

    def test_load_txt(self):
        """加载TXT文件"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, encoding='utf-8'
        ) as f:
            f.write("测试内容")
            path = f.name

        try:
            doc = load_text_file(path)
            assert isinstance(doc, Document)
            assert doc.page_content == "测试内容"
            assert doc.metadata["file_type"] == ".txt"
        finally:
            os.unlink(path)

    def test_load_md(self):
        """加载Markdown文件"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False, encoding='utf-8'
        ) as f:
            f.write("# 标题\n内容")
            path = f.name

        try:
            doc = load_text_file(path)
            assert doc.page_content == "# 标题\n内容"
            assert doc.metadata["file_type"] == ".md"
        finally:
            os.unlink(path)

    def test_metadata(self):
        """元数据包含必要字段"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, encoding='utf-8'
        ) as f:
            f.write("test")
            path = f.name

        try:
            doc = load_text_file(path)
            assert "source" in doc.metadata
            assert "filename" in doc.metadata
            assert "file_type" in doc.metadata
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        """文件不存在时抛出FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            load_text_file("不存在的文件.txt")


class TestLoadDocuments:
    """测试批量加载。"""

    def test_load_from_dir(self):
        """从目录加载，跳过不支持的格式"""
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "a.txt"), 'w', encoding='utf-8') as f:
                f.write("文档A")
            with open(os.path.join(d, "b.md"), 'w', encoding='utf-8') as f:
                f.write("文档B")
            with open(os.path.join(d, "c.pdf"), 'w', encoding='utf-8') as f:
                f.write("不支持")

            docs = load_documents(d)
            assert len(docs) == 2

    def test_empty_dir(self):
        """空目录返回空列表"""
        with tempfile.TemporaryDirectory() as d:
            docs = load_documents(d)
            assert docs == []

    def test_nonexistent_dir(self):
        """不存在的目录抛异常"""
        with pytest.raises(FileNotFoundError):
            load_documents("不存在的目录")

    def test_skip_empty_files(self):
        """空文件被跳过"""
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "empty.txt"), 'w', encoding='utf-8') as f:
                f.write("   \n  ")
            with open(os.path.join(d, "ok.txt"), 'w', encoding='utf-8') as f:
                f.write("有内容")

            docs = load_documents(d)
            assert len(docs) == 1