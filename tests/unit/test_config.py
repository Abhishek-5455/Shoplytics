from ecommerce_ml.configs.config import Settings


def test_default_settings():
    settings = Settings()

    assert settings.database_url == ""


def test_settings_from_environment(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://test-user:test-password@ep-example.neon.tech/app?sslmode=require",
    )

    settings = Settings.from_env()

    assert settings.database_url == (
        "postgresql://test-user:test-password@ep-example.neon.tech/app?sslmode=require"
    )