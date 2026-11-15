import sqlite3

from enterprise.storage import DocumentStore


def test_existing_v2_sqlite_schema_remains_readable(tmp_path):
    database = tmp_path / "documents.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE docs (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            sha256 TEXT UNIQUE NOT NULL,
            characters INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
            idx INTEGER NOT NULL,
            text TEXT NOT NULL,
            metadata TEXT NOT NULL
        );
        CREATE INDEX chunks_doc_id ON chunks(doc_id);
        INSERT INTO docs(id, filename, sha256, characters)
        VALUES ('legacy-doc', 'legacy.md', 'legacy-sha', 6);
        INSERT INTO chunks(id, doc_id, idx, text, metadata)
        VALUES ('legacy-chunk', 'legacy-doc', 0, '旧版数据',
                '{"filename":"legacy.md","start_char":0,"end_char":4}');
        """
    )
    connection.commit()
    connection.close()

    store = DocumentStore(database)
    try:
        assert store.ping() is True
        assert store.get_doc_by_sha("legacy-sha")["id"] == "legacy-doc"
        assert store.chunks()[0]["metadata"]["filename"] == "legacy.md"
        assert store.revision() == 0
    finally:
        store.close()
