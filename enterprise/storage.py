"""小型项目的 SQLite 文档登记库，采用显式事务管理。"""
from __future__ import annotations
import sqlite3
import json
from threading import RLock
from pathlib import Path


class DocumentStore:
    """文档登记库：管理 docs 与 chunks 两张表的读写。"""
    def __init__(self, path: Path):
        self.lock = RLock()  # 可重入锁，保证多线程访问安全
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # 查询结果可按列名访问
        # 初始化时启用外键并建表建索引
        with self.lock, self.conn:
            self.conn.execute('PRAGMA foreign_keys=ON')
            self.conn.executescript('''
                CREATE TABLE IF NOT EXISTS docs (
                  id TEXT PRIMARY KEY, filename TEXT NOT NULL, sha256 TEXT UNIQUE NOT NULL,
                  characters INTEGER NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS chunks (
                  id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
                  idx INTEGER NOT NULL, text TEXT NOT NULL, metadata TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chunks_doc_id ON chunks(doc_id);
            ''')

    def get_doc_by_sha(self, sha: str) -> dict | None:
        """按内容哈希查询文档，用于上传去重。"""
        with self.lock:
            r = self.conn.execute('SELECT * FROM docs WHERE sha256=?', (sha,)).fetchone()
        return dict(r) if r else None

    def add(self, doc: dict, chunks: list[dict]) -> None:
        """在单个事务中插入文档记录及其全部片段。"""
        with self.lock, self.conn:
            self.conn.execute('INSERT INTO docs(id,filename,sha256,characters) VALUES (?,?,?,?)',
                              (doc['id'], doc['filename'], doc['sha256'], doc['characters']))
            # metadata 以 JSON 字符串存储
            self.conn.executemany('INSERT INTO chunks(id,doc_id,idx,text,metadata) VALUES (?,?,?,?,?)',
                [(c['id'], c['doc_id'], c['idx'], c['text'], json.dumps(c['metadata'], ensure_ascii=False))
                  for c in chunks])

    def docs(self) -> list[dict]:
        """列出全部文档，附带各自的片段数量，按创建时间倒序。"""
        with self.lock:
            rows = self.conn.execute('''SELECT d.*, COUNT(c.id) AS chunks FROM docs d
                LEFT JOIN chunks c ON c.doc_id=d.id GROUP BY d.id ORDER BY d.created_at DESC''').fetchall()
        return [dict(r) for r in rows]

    def chunks(self) -> list[dict]:
        """读取全部片段，并将 metadata 反序列化回字典。"""
        with self.lock:
            rows = self.conn.execute('SELECT * FROM chunks ORDER BY doc_id, idx').fetchall()
        return [{**dict(r), 'metadata': json.loads(r['metadata'])} for r in rows]

    def delete(self, doc_id: str) -> bool:
        """删除文档；外键级联会自动清除其片段，返回是否命中。"""
        with self.lock, self.conn:
            cur = self.conn.execute('DELETE FROM docs WHERE id=?', (doc_id,))
            return cur.rowcount > 0

    def close(self) -> None:
        """关闭数据库连接。"""
        with self.lock:
            self.conn.close()