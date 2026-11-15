from enterprise.config import build_database_url


def test_build_database_url_escapes_mysql_password(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DB_HOST", "mysql")
    monkeypatch.setenv("DB_PORT", "3306")
    monkeypatch.setenv("DB_USER", "rag_user")
    monkeypatch.setenv("DB_PASSWORD", "p@ss/word#1")
    monkeypatch.setenv("DB_NAME", "enterprise_rag")

    url = build_database_url(tmp_path)
    assert url.startswith("mysql+pymysql://rag_user:")
    assert "p%40ss%2Fword%231@mysql:3306/enterprise_rag" in url


def test_build_database_url_falls_back_to_sqlite(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DB_HOST", raising=False)
    url = build_database_url(tmp_path)
    assert url.startswith("sqlite+pysqlite:///")
    assert "documents.sqlite3" in url
