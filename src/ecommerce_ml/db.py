import psycopg
import streamlit as st


def get_connection():
	return psycopg.connect(
		st.secrets['DATABASE_URL']
	)