"""将旧版 SQLite 文档元数据迁移到当前配置的 MySQL 数据库。

用法（Docker Compose 启动 MySQL 后，在项目容器中执行）：
    python scripts/migrate_sqlite_to_mysql.py --source .local_data/documents.sqlite3

该脚本只迁移 docs/chunks 元数据，不迁移 Chroma/SQLite 向量文件；向量 ID 保持不变，
因此原有向量索引可继续使用。执行前建议备份 .local_data 和 MySQL volume。
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from enterprise.config import Settings
from enterprise.storage import DocumentStore


def migrate(source: Path, store: DocumentStore) -> tuple[int, int]:
    """返回 (导入文档数, 跳过文档数)，按 SHA-256 实现重复执行安全。"""
    if not source.is_file():
        raise FileNotFoundError(f"SQLite 源数据库不存在：{source}")

    connection = sqlite3.connect(str(source))
    connection.row_factory = sqlite3.Row
    imported = 0
    skipped = 0
    try:
        document_rows = connection.execute(
            "SELECT id, filename, sha256, characters, created_at FROM docs ORDER BY created_at"
        ).fetchall()
        for row in document_rows:
            document = dict(row)
            if store.get_doc_by_sha(document["sha256"]):
                skipped += 1
                continue

            chunk_rows = connection.execute(
                "SELECT id, doc_id, idx, text, metadata FROM chunks WHERE doc_id=? ORDER BY idx",
                (document["id"],),
            ).fetchall()
            chunks = [
                {
                    "id": chunk["id"],
                    "doc_id": chunk["doc_id"],
                    "idx": chunk["idx"],
                    "text": chunk["text"],
                    "metadata": json.loads(chunk["metadata"]),
                }
                for chunk in chunk_rows
            ]
            store.add(document, chunks)
            imported += 1
    finally:
        connection.close()
    return imported, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="迁移 SQLite 文档元数据到 MySQL")
    parser.add_argument("--source", type=Path, required=True, help="旧 documents.sqlite3 文件路径")
    args = parser.parse_args()

    settings = Settings.from_env()
    if not settings.database_url.startswith("mysql+"):
        raise SystemExit(
            "当前 DATABASE_URL 不是 MySQL。请通过 DB_HOST/DB_USER/DB_PASSWORD/DB_NAME 配置 MySQL。"
        )

    store = DocumentStore(settings.database_url)
    try:
        imported, skipped = migrate(args.source, store)
        print(f"迁移完成：导入 {imported} 个文档，跳过 {skipped} 个已存在文档。")
        print("注意：向量索引未迁移；如果原向量目录仍在，原有向量 ID 可继续使用。")
    finally:
        store.close()


if __name__ == "__main__":
    main()
