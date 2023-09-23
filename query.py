from jinja2 import Template
from datetime import datetime
from typing import Union
import json
import faiss
import numpy as np
import pandas as pd
import cohere
import pickle
import random
from llm import embed, gpt
import streamlit as st
import requests
from io import StringIO
from data import (
    df_table,
    table_indexes,
    TABLE_IDS,
    TABLE_EMBS,
    TABLE_INFO_DIR,
    LLM_cohere,
    REGIONER_ID,
    KOMMUNER_ID,
)


# @st.cache_data(persist="/Users/josca/projects/dstGPT/data/streamlit_cache")
def get_table(query, subset_table_ids=None, st=None):
    table_id, ids = find_table(query, subset_table_ids, k=5, rerank=True)

    if st:
        with st.sidebar:
            with st.chat_message("assistant", avatar="👨‍🏫"):
                st.markdown(
                    TABLE_SELECTED.render(
                        table_id=table_id,
                        table_descr=df_table.loc[table_id, "description"],
                    )
                )

    metadata_df, response = decide_table_specs(query, table_id, st=st)
    if metadata_df is None:
        return None, {"tables_considered": ids}, response
    else:
        df = _get_table(table_id, metadata_df["specs"])
        df = postprocess_table(df)
        return df, metadata_df, response


def find_table(query, subset_table_ids=None, k=5, index="en_description", rerank=False):
    query_embedding = embed([query])

    if subset_table_ids:
        indices = np.where(np.isin(TABLE_IDS, table_subset))[0]
        embeddings = TABLE_EMBS[indices]
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

    else:
        index = table_indexes[index]

    D, I = index.search(np.array(query_embedding), k)
    ids = TABLE_IDS[I][0]

    if rerank:
        descriptions = df_table.loc[ids, "description"]
        results = LLM_cohere.rerank(
            query=query, documents=descriptions, top_n=1, model="rerank-english-v2.0"
        )
        table_id = ids[results[0].index]
    else:
        table_id = ids[0]

    return table_id, ids


def decide_table_specs(query, table_id, st=None):
    # Load table info and embeddings
    table_info = pickle.load(open(TABLE_INFO_DIR / table_id / "info.pkl", "rb"))
    vars_embs = pickle.load(open(TABLE_INFO_DIR / table_id / "vars_embs.pkl", "rb"))
    table_metadata = {"table_id": table_id, "description": table_info["description"]}

    # Unpack table info to create LLM messages and store various useful metadata
    n_obs = 1
    variables, var_few, var_many, time_var = [], [], [], None
    for var in table_info["variables"]:
        var["id"] = var["id"].upper()

        if var["time"]:
            values_sample = var["values"][-10:] if len(var["values"]) > 10 else values
            time_var = {
                "id": var["id"],
                "text": var["text"],
                "values": [v["id"] for v in values_sample],
            }
            time_latest = var["values"][-1]["id"]
        elif len(var["values"]) > 10:
            var_many.append(
                {"id": var["id"], "text": var["text"], "values": var["values"][:10]}
            )
        else:
            var_few.append(
                {"id": var["id"], "text": var["text"], "values": var["values"]}
            )

        # For geo variables add useful info for visualization to metadata
        if "map" in var:
            table_metadata["geo"] = {
                "id2text_mapping": {val["text"]: val["id"] for val in var["values"]}
            }
            table_metadata["geo"]["var"] = var["id"]
            table_metadata["geo"]["geo_types"] = detect_geo_types(
                [val["id"] for val in var["values"]]
            )

        table_metadata[var["id"]] = {
            "text": var["text"],
            "values": var["values"],
            "totals": [
                v["text"] for v in var["values"] if v["id"] in ["SUM", "TOT", "000"]
            ],
        }
        variables.append(var["id"])
        n_obs *= len(var["values"])

    # Get GPT response
    msgs = [
        dict(role="system", content=TABLE_SPECS_SYS_MSG.render(n_large=n_obs > 10_000)),
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
    response_txt = gpt(messages=msgs, model="gpt-4", st=st, temperature=0)

    # Parse GPT response
    result = response_txt.split("Result: ")
    if len(result) == 1:
        return None, response_txt
    else:
        result = result[1]
    result = json.loads(result)
    result.update({var: ["*"] for var in variables if var not in result})
    result = {var: values if values != [] else ["*"] for var, values in result.items()}
    var_many = [var["id"] for var in var_many]
    n_obs = 1
    for var, values in result.items():
        if var in var_many:
            if values != ["*"]:
                id2text = {v["id"]: v["text"] for v in table_metadata[var]["values"]}
                var_emb = vars_embs[var]
                result[var] = semantic_search(
                    queries=[id2text[v] for v in values],
                    ids=np.array(var_emb["ids"]),
                    embeddings=var_emb["embs"],
                    k=1,
                )
        elif var == time_var["id"]:
            if values == ["latest"]:
                result[var] = [time_latest]
        else:
            pass

        if values != ["*"]:
            n_obs *= len(result[var])
        else:
            n_obs *= len(table_metadata[var]["values"])

    table_metadata["specs"] = result
    table_metadata["n_obs"] = n_obs
    table_metadata["table_info"] = table_info
    return table_metadata, response_txt


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


def postprocess_table(df):
    df.rename(columns={"INDHOLD": "y"}, inplace=True)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    return df


def semantic_search(queries, ids, embeddings, k=1):
    # Build faiss embedding index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Find k nearest neighbors of query embedding
    queries_emb = embed(queries)
    D, I = index.search(np.array(queries_emb), k)
    ids = ids[I.squeeze()]
    if isinstance(ids, str):
        ids = [ids]
    else:
        ids = list(ids)
    return ids


def detect_geo_types(geo_ids):
    # Determine if data contains both regions and municipalities
    geo_types = []
    if np.any([_id in geo_ids for _id in KOMMUNER_ID]):
        geo_types.append("kommuner")
    if np.any([_id in geo_ids for _id in REGIONER_ID]):
        geo_types.append("regioner")
    return geo_types


###   Prompts   ###

TABLE_SPECS_SYS_MSG = Template(
    """You are data analysis GPT. Your task is to filter a data table such that it shows the relevant information for answering a query.
In order to do so you choose the variables and their corresponding values that should be included. To make your decision you get information on the following form:

Query: "The query that should be answered by the table" 
Table description: "Description of the table"
Variables with few unique values:  [{"id": "the id of the variable", "text": "variable description", "values": [{"id": "the value id", "text": "value description"}], ... ]
Variables with many unique values: [{"id": "the id of the variable", "text": "variable description", "values": ["random sample of 10 unique value texts"]
Time variable: {"id": "the id of the time variable", "text": "time variable description", "values": ["random sample of 10 unique time period values"]}

If the table cannot be used to answer the query then tell the user that Denmarks statistics does not have the data to answer the query.

For all variables you can choose all values by writing ["*"]. 
For variables with few unique values you can choose a subset of values in the form of a list with the value ids. 
For variables with many unique values you can choose a subset of values in the form of a list with the likely value texts. 
For the Time variable you can also write ["latest"] to choose the latest time period only.

{% if n_large -%}
Since the total dataset is large then choose specific values or totals for most variables. If you don't need to filter by a variable choose an aggregate number if possible.

{% endif -%}
Output the result on the following following form:
Result: {"variable id": []}

Before answering spend a few sentences explaining background context, assumptions, and step-by-step thinking. Keep it short and concise.
If you do not use a variable then do not include it in the Result."""
)

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

TABLE_SELECTED = Template(
    """The following table has been selected:
Name: {{ table_id }}
Description: {{ table_descr }}"""
)

if __name__ == "__main__":
    query = "Accidents by police district?"
    table_id = find_table(query, k=5, rerank=True)
    a = decide_table_specs(query, table_id)
