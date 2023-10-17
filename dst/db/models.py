from sqlmodel import (
    SQLModel,
    create_engine,
    Field,
    Session,
    select,
    delete,
    update,
    and_,
    or_,
    exists,
)
from typing import Optional, TypeVar, Generic, List, Type
from datetime import datetime
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime
from sqlalchemy.future import Engine

ModelType = TypeVar("ModelType", bound=SQLModel)
EngineType = TypeVar("EngineType", bound=Engine)


class LLM_REQUEST(SQLModel):
    id: Optional[int] = Field(primary_key=True, index=True)
    table_id: str = Field(index=True)
    query_type: int = Field(index=True)
    prompt: str = Field(index=True)
    response: str
    prev_request_table: Optional[str] = Field(default="", index=True)
    prev_request_api: Optional[str] = Field(default="", index=True)
    n: Optional[int] = Field(default=0)
    created_at: Optional[datetime] = Field(
        sa_column=Column(DateTime, default=datetime.utcnow, nullable=False)
    )


class LLM_EN(LLM_REQUEST, table=True):
    embedding: Optional[list[float]] = Field(sa_column=Column(Vector(1024)))


class LLM_DA(LLM_REQUEST, table=True):
    embedding: Optional[list[float]] = Field(sa_column=Column(Vector(768)))


class Table(SQLModel):
    id: str = Field(primary_key=True, index=True)
    text: str
    description: str
    unit: str


class Table_EN(Table, table=True):
    embedding_1: Optional[list[float]] = Field(sa_column=Column(Vector(1024)))
    embedding_2: Optional[list[float]] = Field(sa_column=Column(Vector(4096)))


class Table_DA(Table, table=True):
    embedding: Optional[list[float]] = Field(sa_column=Column(Vector(768)))
