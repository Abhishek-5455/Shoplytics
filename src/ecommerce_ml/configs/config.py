import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_host: str = "localhost"
    database_port: int = 5432
    database_name: str = "ecommerce"
    database_user: str = "ecommerce_user"
    database_password: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_host=os.getenv("DB_HOST", "localhost"),
            database_port=int(os.getenv("DB_PORT", "5432")),
            database_name=os.getenv("POSTGRES_DB", "ecommerce"),
            database_user=os.getenv("POSTGRES_USER", "ecommerce_user"),
            database_password=os.getenv("POSTGRES_PASSWORD", ""),
        )


settings = Settings.from_env()