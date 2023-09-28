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
import pandas as pd
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime
from sqlalchemy.future import Engine
from sqlalchemy.dialects.postgresql import insert
from data import config

ModelType = TypeVar("ModelType", bound=SQLModel)
EngineType = TypeVar("EngineType", bound=Engine)


###   Data tables   ###
class LLM_REQUEST(SQLModel):
    id: Optional[int] = Field(primary_key=True, index=True)
    table_id: str = Field(index=True)
    action_type: int = Field(index=True)
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


class API_Request(SQLModel, table=True):
    table_id: str = Field(primary_key=True, index=True)
    request_hash: str = Field(primary_key=True, index=True)
    lang: str = Field(primary_key=True, index=True)
    file_name: str


###   CRUD   ###
class CRUDBase(Generic[ModelType, EngineType]):
    def __init__(self, model: Type[ModelType], engine: Type[EngineType]):
        """
        CRUD object with default methods to Create, Read, Update, Delete (CRUD).

        **Parameters**

        * `model`: A SQLModel class
        * `engine`: A sqlalchemy engine
        """
        self.model = model
        self.engine = engine

    # def get(self, id) -> ModelType:
    #     with Session(self.engine) as session:
    #         return session.get(self.model, id)

    def get(self, model_obj: ModelType):
        with Session(self.engine) as session:
            stmt = select(self.model).where(
                and_(
                    *[
                        getattr(self.model, k) == v
                        for k, v in model_obj.dict(exclude_unset=True).items()
                        if v
                    ]
                )
            )
            result = session.exec(stmt).first()
        return result

    def get_table(self):
        with Session(self.engine) as session:
            stmt = select(self.model)
            result = session.exec(stmt).all()
        return pd.DataFrame([r.dict() for r in result])

    def create(self, model_obj: ModelType) -> ModelType:
        with Session(self.engine) as session, session.begin():
            stmt = insert(self.model).values(**model_obj.dict(exclude_unset=True))
            session.exec(stmt)

    def update(self, model_obj: ModelType) -> ModelType:
        model_update = model_obj.dict(exclude_unset=True)
        with Session(self.engine) as session, session.begin():
            stmt = update(self.model).where(self.model.id == model_obj.id)
            session.exec(stmt.values(**model_update))

    def delete(self, id) -> None:
        with Session(self.engine) as session, session.begin():
            stmt = delete(self.model).where(self.model.id == id)
            session.exec(stmt)

    def check_cache(
        self,
        table_id: dict = {},
        action: dict = {},
        start_date=None,
        end_date=None,
        cols=[],
        limit=False,
        similarity_query=False,
    ) -> List[ModelType]:
        with Session(self.engine) as session:
            stmt = (
                select(*[getattr(self.model, s) for s in cols])
                if cols
                else select(self.model)
            )
            if equals:
                stmt = stmt.where(
                    and_(*[getattr(self.model, k) == v for k, v in equals.items() if v])
                )
            if _in:
                stmt = stmt.where(
                    or_(*[getattr(self.model, k).in_(v) for k, v in _in.items() if v])
                )

            if start_date and end_date:
                stmt = stmt.where(self.model.timestamp.between(start_date, end_date))
            elif start_date:
                stmt = stmt.where(self.model.timestamp >= start_date)
            elif end_date:
                stmt = stmt.where(self.model.timestamp <= end_date)
            else:
                pass

            if similarity_query:
                query_emb = embed(similarity_query)[0]
                stmt = stmt.order_by(self.model.embedding.l2_distance(query_emb))
            else:
                if "timestamp" in cols:
                    stmt.order_by(self.model.timestamp.desc())
                else:
                    pass

            if limit:
                stmt = stmt.limit(limit)
            result = session.exec(stmt).all()

        result = session.exec(stmt).all()
        cols = cols if cols else self.model.__fields__.keys()
        return pd.DataFrame.from_records(
            result,
            columns=cols,
        )


class CRUD_LLM_EN(CRUDBase[LLM_EN, EngineType]):
    ...


class CRUD_LLM_DA(CRUDBase[LLM_DA, EngineType]):
    ...


class CRUD_API_Request(CRUDBase[API_Request, EngineType]):
    ...


###  Initialise engine   ###
POSTGRES_SQLALCHEMY_URI = f'postgresql://{config["PGUSER"]}:{config["PGPASSWORD"]}@{config["PGHOST"]}:5432/{config["PGDATABASE"]}'
engine = create_engine(POSTGRES_SQLALCHEMY_URI)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


models = {"en": LLM_EN, "da": LLM_DA, "api": API_Request}
crud = {
    "en": CRUD_LLM_EN(LLM_EN, engine),
    "da": CRUD_LLM_DA(LLM_DA, engine),
    "api": CRUD_API_Request(API_Request, engine),
}
