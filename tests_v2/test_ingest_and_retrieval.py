import pytest
from enterprise.ingest import parse_file
from enterprise.retrieval import bm25, rrf


def test_parse_rejects_executable():
    with pytest.raises(ValueError):
        parse_file('evil.exe',b'hello',100)


def test_parse_rejects_empty():
    with pytest.raises(ValueError):
        parse_file('empty.txt',b'',100)


def test_reuses_original_splitter_if_present():
    from enterprise.legacy_bridge import split_document,Document
    chunks=split_document(Document(page_content='abcdefghij',metadata={'filename':'a.txt'}),5,1)
    assert len(chunks)==3
    assert chunks[1].metadata['start_char']==4


def test_upload_dedup_delete_and_persistent_index(services):
    svc=services
    data=('普通员工到上海出差住宿标准为每晚400元，报销需要发票和审批记录。'*10).encode()
    res=svc.ingest.upload('policy.md',data)
    assert res['chunks']>0
    assert svc.ingest.upload('policy_copy.md',data)['duplicate'] is True
    assert len(svc.store.docs())==1
    assert svc.retriever.search('上海住宿标准',top_k=3)
    assert svc.ingest.delete(res['id']) is True
    assert svc.store.docs()==[]
    assert svc.retriever.search('上海住宿标准',top_k=3)==[]


def test_rag_sources_not_invented(services):
    answer=services.rag.answer('不存在的内容')
    assert answer['sources']==[]
    services.ingest.upload('policy.txt','上海普通员工住宿费400元每晚'.encode())
    data=services.rag.answer('上海住宿')
    assert data['sources'][0]['source_id']=='S1'
    assert '演示模式' in data['answer']


def test_bm25_and_rrf():
    docs=[{'id':'1','text':'上海 差旅住宿'} ,{'id':'2','text':'Python Docker 部署'}]
    assert bm25('上海住宿',docs)[0][0]=='1'
    ranked=rrf([('1',1.0),('2',.5)],[('2',1.0),('1',.2)])
    assert {id for id,_ in ranked}=={'1','2'}
