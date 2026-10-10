from ecommerce_ml.configs.config import Settings


def test_default_settings():
    settings = Settings()

    assert settings.database_host == "localhost"
    assert settings.database_port == 5432
    assert settings.database_name == "ecommerce"
    assert settings.database_password == ""


def test_settings_from_environment(monkeypatch):
    monkeypatch.setenv("DB_HOST", "test-db")
    monkeypatch.setenv("DB_PORT", "5433")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test-password")

    settings = Settings.from_env()

    assert settings.database_host == "test-db"
    assert settings.database_port == 5433
    assert settings.database_password == "test-password"