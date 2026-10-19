import os
from pathlib import Path
import pytest
from enterprise.config import Settings
from enterprise.service import Services


@pytest.fixture
def services(monkeypatch,tmp_path):
    monkeypatch.setenv('DEMO_MODE','true')
    monkeypatch.setenv('APP_DATA_DIR',str(tmp_path))
    monkeypatch.setenv('APP_API_TOKEN','unit-test-secret')
    monkeypatch.setenv('RERANK_ENABLED','false')
    config=Settings.from_env()
    svc=Services(config)
    yield svc
    svc.close()
