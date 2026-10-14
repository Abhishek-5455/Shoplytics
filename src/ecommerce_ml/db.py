import psycopg

from ecommerce_ml.configs.config import settings


def get_connection():
	return psycopg.connect(
		host = settings.database_host,
		port = settings.database_port,
		user = settings.database_user,
		dbname = settings.database_name,
		password = settings.database_password
	)