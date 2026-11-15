from enterprise.storage import DocumentStore


def _doc(doc_id="doc-1", sha="sha-1"):
    return {"id": doc_id, "filename": "policy.md", "sha256": sha, "characters": 12}


def _chunks(doc_id="doc-1"):
    return [
        {
            "id": f"chunk-{doc_id}",
            "doc_id": doc_id,
            "idx": 0,
            "text": "差旅住宿标准为 400 元。",
            "metadata": {"filename": "policy.md", "chunk_index": 0, "start_char": 0, "end_char": 12},
        }
    ]


def test_document_store_crud_and_revision(tmp_path):
    store = DocumentStore(tmp_path / "documents.sqlite3")
    try:
        assert store.revision() == 0
        store.add(_doc(), _chunks())
        assert store.revision() == 1
        assert store.get_doc_by_sha("sha-1")["id"] == "doc-1"
        assert store.docs()[0]["chunks"] == 1
        assert store.chunks()[0]["metadata"]["filename"] == "policy.md"

        assert store.delete("doc-1") is True
        assert store.revision() == 2
        assert store.docs() == []
        assert store.chunks() == []
        assert store.delete("missing") is False
        assert store.revision() == 2
    finally:
        store.close()


def test_document_store_rolls_back_document_and_chunks_together(tmp_path):
    store = DocumentStore(tmp_path / "documents.sqlite3")
    try:
        store.add(_doc(), _chunks())
        before_revision = store.revision()
        try:
            # 重复主键触发数据库异常，事务应回滚，不能留下半条数据。
            store.add(_doc(doc_id="doc-2", sha="sha-2"), _chunks(doc_id="doc-1"))
        except Exception:
            pass
        assert store.get_doc_by_sha("sha-2") is None
        assert store.revision() == before_revision
        assert len(store.docs()) == 1
    finally:
        store.close()
