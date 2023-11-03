from typing import Optional, TypeVar, Generic, List, Type
from datetime import datetime
import pandas as pd
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime
from sqlalchemy.future import Engine
from sqlalchemy.dialects.postgresql import insert
from sqlmodel import SQLModel, Session, select, and_, update, delete, or_
from dst.db import models
from dst.db.db import engine

ModelType = TypeVar("ModelType", bound=SQLModel)
EngineType = TypeVar("EngineType", bound=Engine)


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

    def get_by_id(self, id) -> ModelType:
        with Session(self.engine) as session:
            stmt = select(self.model).where(self.model.id == id)
            result = session.exec(stmt).first()
        return result

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

    # def check_cache(self, model_obj: ModelType) -> List[ModelType]:
    #     with Session(self.engine) as session:
    #         stmt = (
    #             select(
    #                 self.model.prompt,
    #                 self.model.embedding.cosine_distance(model_obj.embedding),
    #             )
    #             # ).where(
    #             # and_(
    #             # self.model.table_id == model_obj.table_id,
    #             # self.model.action_type == model_obj.action_type,
    #             # self.model.prev_request_table == model_obj.prev_request_table,
    #             # self.model.prev_request_api == model_obj.prev_request_api,
    #             # )
    #             # )
    #             # .filter(
    #             #     self.model.embedding.cosine_distance(model_obj.embedding) < 0.12
    #             # )
    #             .order_by(self.model.embedding.cosine_distance(model_obj.embedding))
    #         )
    #         result = session.exec(stmt).all()  # first()
    #     return result


class CRUD_LLM_EN(CRUDBase[models.LLM_EN, EngineType]):
    ...


class CRUD_LLM_DA(CRUDBase[models.LLM_DA, EngineType]):
    ...


class CRUD_Table(CRUDBase[ModelType, EngineType]):
    def get_likely_table_ids_for_QA(
        self, prompt_embedding: list[float], top_k=5, subset_table_ids=[]
    ):
        with Session(self.engine) as session:
            stmt = select(
                self.model.id,
                self.model.description,
            )
            if subset_table_ids:
                stmt = stmt.where(self.model.id.in_(subset_table_ids))

            stmt = stmt.order_by(
                self.model.embedding_2.cosine_distance(prompt_embedding)
            ).limit(top_k)
            result = session.exec(stmt).all()  # first()
        return [{"id": r[0], "description": r[1]} for r in result]

    def get_descriptions(self, ids):
        with Session(self.engine) as session:
            stmt = select(self.model.id, self.model.description)
            if isinstance(ids, str):
                stmt = stmt.where(self.model.id == ids)
            else:
                stmt = stmt.where(self.model.id.in_(ids))
            result = session.exec(stmt).all()
        return result

    def get_table(self):
        with Session(self.engine) as session:
            stmt = select(self.model.id, self.model.text, self.model.description)
            result = session.exec(stmt).all()
        return pd.DataFrame(result, columns=["id", "text", "description"])


class CRUD_Table_EN(CRUD_Table[models.Table_EN, EngineType]):
    ...


class CRUD_Table_DA(CRUD_Table[models.Table_DA, EngineType]):
    ...


class CRUD_Table_info(CRUDBase[models.Table_info, EngineType]):
    def get(self, id: str, lang: str):
        with Session(self.engine) as session:
            stmt = select(self.model).where(
                and_(
                    self.model.id == id,
                    self.model.lang == lang,
                )
            )
            table_info = session.exec(stmt).first()
        return table_info.info


class CRUD_Table_emb(CRUD_Table[models.Table_emb, EngineType]):
    ...


llm_en = CRUD_LLM_EN(models.LLM_EN, engine)
llm_da = CRUD_LLM_DA(models.LLM_DA, engine)
table_en = CRUD_Table_EN(models.Table_EN, engine)
table_da = CRUD_Table_DA(models.Table_DA, engine)
table_info = CRUD_Table_info(models.Table_info, engine)
table_emb = CRUD_Table_emb(models.Table_emb, engine)
