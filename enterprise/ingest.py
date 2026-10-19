"""上传 -> 解析 -> 复用之前的切分器 -> 登记 -> 向量化 -> 持久化索引。"""
from __future__ import annotations
from io import BytesIO
import hashlib
import re
from pathlib import Path
from uuid import uuid4

from enterprise.legacy_bridge import Document, split_document

# 允许上传的文件后缀
ALLOWED = {'.txt', '.md', '.markdown', '.pdf'}


def parse_file(filename: str, data: bytes, max_bytes: int) -> str:
    """解析上传文件为纯文本，校验类型、大小与编码。"""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED:
        raise ValueError('仅支持 .txt/.md/.markdown/.pdf')
    if not data or len(data) > max_bytes:
        raise ValueError('文件为空或超出上传大小限制')
    if suffix == '.pdf':
        from pypdf import PdfReader
        try:
            reader = PdfReader(BytesIO(data))
            if len(reader.pages) > 200: raise ValueError('PDF 页数超限')
            text = '\n'.join(page.extract_text() or '' for page in reader.pages)
        except ValueError: raise
        except Exception as exc: raise ValueError('PDF 解析失败或已加密') from exc
    else:
        try: text = data.decode('utf-8-sig')
        except UnicodeDecodeError as exc: raise ValueError('文件必须使用 UTF-8 编码') from exc
    # 统一换行符并压缩多余空行
    text = re.sub(r'\n{3,}', '\n\n', text.replace('\r\n', '\n')).strip()
    if not text: raise ValueError('未抽取到可索引文本；扫描件 PDF 暂不支持 OCR')
    return text


class IngestionService:
    """入库服务：负责文档解析、切分、去重与索引写入。"""
    def __init__(self, store, vector, embedder, settings):
        self.store, self.vector, self.embedder, self.settings = store, vector, embedder, settings

    def upload(self, filename: str, content: bytes) -> dict:
        """上传文档：按内容哈希去重，切分后写入登记库与向量库。"""
        safe_filename = Path(filename.replace('\\','/')).name[:180]  # 清洗路径并截断，防目录穿越
        text = parse_file(safe_filename, content, self.settings.max_upload_mb * 1024 * 1024)
        sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
        existing = self.store.get_doc_by_sha(sha)
        if existing:  # 命中相同内容，直接返回已存在文档
            return {'id': existing['id'], 'filename': existing['filename'],
                    'duplicate': True, 'chunks': len([c for c in self.store.chunks() if c['doc_id']==existing['id']])}
        doc_id = uuid4().hex
        source_doc = Document(page_content=text, metadata={'filename':safe_filename, 'doc_id':doc_id})
        split = split_document(source_doc, self.settings.chunk_size, self.settings.chunk_overlap)
        # 为每个片段生成稳定 id 并附带定位元数据
        rows = [{'id': hashlib.sha256(f'{doc_id}:{i}'.encode()).hexdigest()[:32],
                 'doc_id':doc_id, 'idx':i, 'text':chunk.page_content,
                 'metadata':{'filename':safe_filename, 'chunk_index':i,
                             'start_char':chunk.metadata['start_char'],
                             'end_char':chunk.metadata['end_char']}}
                 for i,chunk in enumerate(split)]
        vectors = self.embedder.encode([row['text'] for row in rows])
        self.store.add({'id':doc_id,'filename':safe_filename,'sha256':sha,'characters':len(text)},rows)
        try:
            self.vector.upsert(rows, vectors)
        except Exception:
            self.store.delete(doc_id)  # 向量写入失败则回滚登记，保持一致性
            raise
        return {'id':doc_id,'filename':safe_filename,'duplicate':False,'chunks':len(rows)}

    def delete(self, doc_id: str) -> bool:
        """删除文档：先删向量再删登记；写入中断时管理员可执行 rebuild。"""
        # Remove vector first; registry deletion follows. Admin may run rebuild on interrupted write.
        exists = any(d['id']==doc_id for d in self.store.docs())
        if not exists: return False
        self.vector.delete(doc_id)
        return self.store.delete(doc_id)

    def rebuild(self) -> int:
        """依据登记库中的全部片段重建向量索引，返回片段数。"""
        chunks = self.store.chunks()
        if chunks:
            self.vector.upsert(chunks, self.embedder.encode([c['text'] for c in chunks]))
        return len(chunks)