from unittest.mock import patch

import pytest

from ecommerce_ml.db import get_connection


def test_db_connection_uses_database_url_environment(monkeypatch):
    database_url = (
        "postgresql://user:password@ep-example.neon.tech/"
        "database?sslmode=require"
    )
    monkeypatch.setenv("DATABASE_URL", database_url)

    with patch("ecommerce_ml.db.psycopg.connect") as connect:
        get_connection()

    connect.assert_called_once_with(database_url)


def test_db_connection_requires_neon_ssl_url(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://user:password@localhost/database",
    )

    with pytest.raises(RuntimeError, match="Neon PostgreSQL host"):
        get_connection()


def test_db_connection():
	with get_connection() as conn, conn.cursor() as cur:
		cur.execute("SELECT 1")

		assert cur.fetchone() == (1, )
			