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
from typing import Optional, TypeVar, Generic, List, Type, Dict
from datetime import datetime
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, JSON
from sqlalchemy.future import Engine

ModelType = TypeVar("ModelType", bound=SQLModel)
EngineType = TypeVar("EngineType", bound=Engine)


class LLM_Request(SQLModel, table=True):
    id: Optional[int] = Field(primary_key=True, index=True)
    table_id: str = Field(index=True)
    query_type: int = Field(index=True)
    prompt: str = Field(index=True)
    lang: str = Field(index=True)
    response: str
    prev_request_table: Optional[str] = Field(default="", index=True)
    prev_request_api: Optional[str] = Field(default="", index=True)
    n: Optional[int] = Field(default=0)
    created_at: Optional[datetime] = Field(
        sa_column=Column(DateTime, default=datetime.utcnow, nullable=False)
    )


class Table(SQLModel):
    id: str = Field(primary_key=True, index=True)
    text: str
    description: str
    unit: str


class Table_EN(Table, table=True):
    embedding: Optional[list[float]] = Field(sa_column=Column(Vector(4096)))


class Table_DA(Table, table=True):
    embedding: Optional[list[float]] = Field(sa_column=Column(Vector(1024)))


class Table_info(SQLModel, table=True):
    id: str = Field(primary_key=True, index=True)
    lang: str = Field(index=True)
    info: Dict = Field(sa_column=Column(JSON))


class Table_emb(SQLModel, table=True):
    table_id: str = Field(primary_key=True, index=True)
    var_name: str = Field(primary_key=True, index=True)
    lang: str = Field(primary_key=True, index=True)
    var_val_id: str = Field(primary_key=True, index=True)
    embedding: Optional[list[float]] = Field(sa_column=Column(Vector(384)))
