from dst.db import models
from dst.data import config
from sqlmodel import SQLModel, create_engine


###  Initialise engine   ###
POSTGRES_SQLALCHEMY_URI = f'postgresql://{config["PGUSER"]}:{config["PGPASSWORD"]}@{config["PGHOST"]}:5432/{config["PGDATABASE"]}'
engine = create_engine(POSTGRES_SQLALCHEMY_URI, pool_pre_ping=True)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
