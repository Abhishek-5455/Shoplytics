from urllib.parse import parse_qs, urlsplit

import psycopg

from ecommerce_ml.configs.config import Settings


def get_connection():
    database_url = Settings.from_env().database_url
    if not database_url:
        raise RuntimeError("Set DATABASE_URL to your Neon connection URL.")

    try:
        parsed_url = urlsplit(database_url)
        sslmode = parse_qs(parsed_url.query).get("sslmode", [""])[0]
    except ValueError as exc:
        raise RuntimeError("DATABASE_URL is not a valid Neon URL.") from exc

    if (
        parsed_url.scheme not in {"postgres", "postgresql"}
        or not parsed_url.hostname
        or not parsed_url.hostname.endswith(".neon.tech")
        or sslmode not in {"require", "verify-ca", "verify-full"}
    ):
        raise RuntimeError(
            "DATABASE_URL must point to a Neon PostgreSQL host and require SSL."
        )

    return psycopg.connect(database_url)