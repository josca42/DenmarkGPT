from jinja2 import Template
from datetime import datetime
from typing import Union
from dotenv import load_dotenv
import json
import faiss
import numpy as np
import pandas as pd
import cohere
import os
import pickle
from pathlib import Path
import random
from llm import embed, gpt


load_dotenv()

LLM_cohere = cohere.Client(os.environ["COHERE_API_KEY"])

INDEX_DIR = Path(os.environ["DATA_DIR"]) / "indexes"
TABLE_INFO_DIR = Path(os.environ["DATA_DIR"]) / "tables_info"

df_table = (
    pd.read_parquet(INDEX_DIR / "table_info.parquet").set_index("id").sort_index()
)
table_ids = np.load(INDEX_DIR / "tables_id.npy", allow_pickle=True)
table_indexes = {}
table_indexes["en_text"] = faiss.read_index(str(INDEX_DIR / "table_text_en_emb.bin"))
table_indexes["en_description"] = faiss.read_index(
    str(INDEX_DIR / "table_description_en_emb.bin")
)


def get_table(query, st=None):
    table_id = find_table(query, k=5, rerank=True)
    table_specs, metadata = decide_table_specs(query, table_id, st=st)
    table_df = _get_table(table_id, table_specs)
    return table_df, metadata


def find_table(query, k=5, index="en_description", rerank=False):
    query_embedding = embed([query])
    D, I = table_indexes[index].search(np.array(query_embedding), k)
    ids = table_ids[I][0]

    if rerank:
        descriptions = df_table.loc[ids, "description"]
        results = LLM_cohere.rerank(
            query=query, documents=descriptions, top_n=1, model="rerank-english-v2.0"
        )
        table_id = ids[results[0].index]
    else:
        table_id = ids[0]

    return table_id


def decide_table_specs(query, table_id, st=None):
    metadata = dict()

    # Load table info and embeddings
    table_info = pickle.load(open(TABLE_INFO_DIR / table_id / "info.pkl", "rb"))
    vars_embs = pickle.load(open(TABLE_INFO_DIR / table_id / "vars_embs.pkl", "rb"))

    # Unpack table info to create LLM messages
    var_few, var_many, time_var = [], [], None
    for var in table_info["variables"]:
        if var["time"]:
            value_texts = [v["text"] for v in var["values"]]
            values_sample = (
                random.sample(value_texts, 10) if len(value_texts) > 10 else value_texts
            )
            time_var = {"id": var["id"], "text": var["text"], "values": values_sample}
        elif len(var["values"]) > 20:
            values_sample = random.sample([v["text"] for v in var["values"]], 10)
            var_many.append(
                {"id": var["id"], "text": var["text"], "values": values_sample}
            )
        else:
            var_few.append(
                {"id": var["id"], "text": var["text"], "values": var["values"]}
            )

        # For geo variables add useful info for visualization to metadata
        if "map" in var:
            metadata["geo"] = {
                "id2text_mapping": {val["text"]: val["id"] for val in var["values"]}
            }

    # Get GPT response
    msgs = [
        dict(role="system", content=TABLE_SPECS_SYS_MSG),
        dict(
            role="user",
            content=TABLE_SPECS_USER_MSG.render(
                query=query,
                description=table_info["description"],
                vars_few=var_few,
                vars_many=var_many,
                time_var=time_var,
            ).strip(),
        ),
    ]
    response_txt = gpt(messages=msgs, model="gpt-4", st=st)

    # Parse GPT response
    result = response_txt.split("Result: ")[1]
    result = json.loads(result)
    var_many = [var["id"] for var in var_many]
    for var, values in result.items():
        if var in var_many:
            if values != ["*"]:
                var_emb = vars_embs[var]
                result[var] = semantic_search(
                    queries=values,
                    ids=np.array(var_emb["ids"]),
                    embeddings=var_emb["embs"],
                    k=1,
                )
    return result, metadata


def _get_table(table_id, table_specs):
    r = requests.post(
        "https://api.statbank.dk/v1/data",
        json={
            "table": table_id,
            "format": "BULK",
            "lang": "en",
            "variables": [
                {"code": var, "values": values} for var, values in table_specs.items()
            ],
        },
    )
    df = pd.read_csv(StringIO(r.text), sep=";", decimal=",")
    return df


def semantic_search(queries, ids, embeddings, k=1):
    # Build faiss embedding index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Find k nearest neighbors of query embedding
    queries_emb = embed(queries)
    D, I = index.search(np.array(queries_emb), k)
    ids = ids[I.squeeze()]
    return list(ids)


###   Prompts   ###

TABLE_SPECS_SYS_MSG = """You are data analysis GPT. Your task is to filter a data table such that it shows the relevant information for answering a query.
In order to do so you choose the variables and their corresponding values that should be included. To make your decision you get information on the following form:

Query: "The query that should be answered by the table" 
Table description: "Description of the table"
Variables with few unique values:  [{"id": "the id of the variable", "text": "variable description", "values": [{"id": "the value id", "text": "value description"}], ... ]
Variables with many unique values: [{"id": "the id of the variable", "text": "variable description", "values": ["random sample of 10 unique value texts"]
Time variable: {"id": "the id of the time variable", "text": "time variable description", "values": ["random sample of 10 unique time period values"]}

Before answering spend a few sentences explaining background context, assumptions, and step-by-step thinking. 

For all variables you can choose all values by writing ["*"]. 
For variables with few unique values you can choose a subset of values in the form of a list with the value ids. 
For variables with many unique values you can choose a subset of values in the form of a list with the likely value texts. 
For the Time variable you can also write ["latest"] to choose the latest time period only.

Output the result on the following following form:
Result: {"variable id": []}

If you do not use a variable then do not include it in the Result."""

TABLE_SPECS_USER_MSG = Template(
    """Query: {{ query }}
Table description: {{ description }}
{% if vars_few -%}
Variables with few unique values: {{ vars_few }}
{% endif -%}
{% if vars_many -%}
Variables with many unique values: {{ vars_many }}
{% endif -%}
{% if time_var -%}
Time variable: {{ time_var }}
{% endif -%}"""
)

if __name__ == "__main__":
    query = "Accidents by police district?"
    table_id = find_table(query, k=5, rerank=True)
    a = decide_table_specs(query, table_id)
