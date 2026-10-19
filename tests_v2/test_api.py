from fastapi.testclient import TestClient
from enterprise.api import create_app


def test_api_auth_and_upload(services):
    app=create_app(services)
    with TestClient(app) as client:
        assert client.get('/health').status_code==200
        assert client.get('/api/documents').status_code==401
        head={'Authorization':'Bearer unit-test-secret'}
        res=client.post('/api/documents',headers=head,
            files={'file':('travel.md','普通员工上海出差住宿标准400元'.encode('utf-8'),'text/markdown')})
        assert res.status_code==200, res.text
        res=client.post('/api/query',headers=head,json={'query':'上海住宿标准？','strategy':'hybrid'})
        assert res.status_code==200, res.text
        assert res.json()['sources']
        assert client.post('/api/agent',headers=head,json={'question':'计算','thread_id':'abc'}).status_code==503
        assert client.delete('/api/documents/'+res.json()['sources'][0]['doc_id'],headers=head).status_code==200


def test_input_rejects_invalid_top_k(services):
    with TestClient(create_app(services)) as client:
        response=client.post('/api/query',headers={'Authorization':'Bearer unit-test-secret'},
            json={'query':'hello','top_k':999})
        assert response.status_code==422
