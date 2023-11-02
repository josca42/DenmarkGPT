from jinja2 import Template
from datetime import datetime
from typing import Union
import json
import faiss
import numpy as np
import pandas as pd
import pickle
import random
from dst import llm
from dst.db import crud, models
import streamlit as st
import requests
from io import StringIO
from dst.data import (
    TABLE_INFO_DA_DIR,
    TABLE_INFO_EN_DIR,
    LLM_cohere,
    REGIONER_ID,
    KOMMUNER_ID,
)


def determine_query_type(user_input, setting_info):
    setting_info["table_id"] = "QUERYTYPE"
    sys_msg = dict(
        role="system", content=ACTION_SYS_MSG.render(lang=setting_info["lang"])
    )
    example_msgs = ACTION_MSG_EN_EXAMPLE  # if lang == "en" else ACTION_MSG_DA_EXAMPLE
    user_msg = dict(
        role="user",
        content=ACTION_USER_MSG.render(
            table_description=setting_info["prev_table_descr"],
            api_request=setting_info["prev_api_request"],
            user_request=user_input,
        ),
    )
    response_txt = llm.gpt(
        messages=[sys_msg] + example_msgs + [user_msg],
        model="gpt-3.5-turbo",
        temperature=0,
        setting_info=setting_info,
    )
    result = response_txt.split("-")
    if len(result) == 1:
        query_table_descr = ""
    else:
        query_table_descr = result[1].strip()

    query_type = int(result[0].strip())
    return query_type, query_table_descr


def find_table_candidates(
    table_descr, lang, subset_table_ids=[], k=10, query="", rerank=False
):
    query_embedding = llm.embed([table_descr], lang=lang)[0]
    crud_table = crud.table_en if lang == "en" else crud.table_da
    tables = crud_table.get_likely_table_ids_for_QA(
        query_embedding, top_k=10, subset_table_ids=subset_table_ids
    )

    if rerank:
        model = "rerank-english-v2.0" if lang == "en" else "rerank-multilingual-v2.0"
        results = LLM_cohere.rerank(
            query=query + ". " + table_descr,
            documents=[t["description"] for t in tables],
            top_n=1,
            model=model,
        )
        table = tables[results[0].index]
    else:
        table = tables[0]

    return table, tables


def create_api_call(query, table_id, setting_info, update_request, st):
    lang = setting_info["lang"]
    table_metadata = load_and_process_table_info(table_id=table_id, lang=lang)
    api_call_txt = gpt_create_api_call(
        query, table_metadata, setting_info, update_request, st
    )
    api_call, table_metadata = parse_gpt_response(
        response_txt=api_call_txt, table_metadata=table_metadata, lang=lang
    )
    if api_call is None:
        return None, table_metadata, api_call_txt
    else:
        return api_call, table_metadata, api_call_txt


def get_table_from_api(table_id, api_call, lang):
    df = call_dst_api(table_id, api_call, lang)
    df = postprocess_table(df)
    return df


def load_and_process_table_info(table_id, lang):
    table_dir = TABLE_INFO_EN_DIR if lang == "en" else TABLE_INFO_DA_DIR

    # Load table info
    table_info = pickle.load(open(table_dir / table_id / "info.pkl", "rb"))
    table_metadata = {
        "table_id": table_id,
        "description": table_info["description"],
        "table_info": table_info,
    }

    # Unpack table info to create LLM messages and store various useful metadata
    n_obs = 1
    variables, var_few, var_many, time_var = [], [], [], None
    for var in table_info["variables"]:
        var["id"] = var["id"].upper()

        if var["time"]:
            values_sample = (
                [var["values"][0]] + var["values"][-10:]
                if len(var["values"]) > 10
                else var["values"]
            )
            time_var = {
                "id": var["id"],
                "text": var["text"],
                "values": [v["id"] for v in values_sample],
            }
            time_latest = var["values"][-1]["id"]
        elif len(var["values"]) > 10:
            var_many.append(
                {
                    "id": var["id"],
                    "text": var["text"],
                    "values": [v["text"] for v in var["values"][:10]],
                }
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

    table_metadata["variables"] = variables
    table_metadata["var_few"] = var_few
    table_metadata["var_many"] = var_many
    table_metadata["time_var"] = time_var
    table_metadata["time_latest"] = time_latest
    return table_metadata


def gpt_create_api_call(
    query, table_metadata, setting_info={}, update_request="", st=None
):
    # Use GPT to create api call to DST api
    if update_request:
        msgs = [
            dict(role="system", content=TABLE_SPECS_UPDATE_SYS_MSG),
            dict(
                role="user",
                content=TABLE_SPECS_UPDATE_USER_MSG.render(
                    request=update_request,
                    cmd=query,
                    description=table_metadata["description"],
                    vars_few=table_metadata["var_few"],
                    vars_many=table_metadata["var_many"],
                    time_var=table_metadata["time_var"],
                ).strip(),
            ),
        ]
        response_txt = llm.gpt(
            messages=msgs,
            model="gpt-4",
            st=st,
            temperature=0,
            setting_info=setting_info,
        )
    else:
        if setting_info:
            setting_info["prev_request_table"] = ""
            setting_info["prev_request_api"] = ""

        msgs = [
            dict(
                role="system",
                content=TABLE_SPECS_SYS_MSG.render(lang=setting_info["lang"]),
            ),
            dict(
                role="user",
                content=TABLE_SPECS_USER_MSG.render(
                    query=query,
                    description=table_metadata["description"],
                    vars_few=table_metadata["var_few"],
                    vars_many=table_metadata["var_many"],
                    time_var=table_metadata["time_var"],
                ).strip(),
            ),
        ]
        response_txt = llm.gpt(
            messages=msgs,
            model="gpt-4",
            st=st,
            temperature=0,
            setting_info=setting_info,
        )

    return response_txt


def parse_gpt_response(response_txt, table_metadata, lang):
    table_dir = TABLE_INFO_EN_DIR if lang == "en" else TABLE_INFO_DA_DIR
    vars_embs = pickle.load(
        open(table_dir / table_metadata["table_id"] / "vars_embs.pkl", "rb")
    )
    variables, var_few, var_many, time_var, time_latest = (
        table_metadata["variables"],
        table_metadata["var_few"],
        table_metadata["var_many"],
        table_metadata["time_var"],
        table_metadata["time_latest"],
    )

    # Parse GPT response
    if lang == "en":
        result = response_txt.split("Result: ")
    else:
        result = response_txt.split("Resultat: ")

    if len(result) == 1:
        try:
            result = json.loads(response_txt)
        except:
            return None, table_metadata
    else:
        result = json.loads(result[1])
        if result == {}:
            return None, table_metadata

    # Parse GPT response
    result.update({var: ["*"] for var in variables if var not in result})
    result = {var: values if values != [] else ["*"] for var, values in result.items()}
    var_many = [var["id"] for var in var_many]
    n_obs = 1
    for var, values in result.items():
        if var in var_many:
            if values != ["*"]:
                if any(
                    any(val == val["id"] for val in table_metadata[var]["values"])
                    for val in values
                ):
                    continue
                else:
                    var_emb = vars_embs[var]
                    result[var] = semantic_search(
                        queries=values,
                        ids=np.array(var_emb["ids"]),
                        embeddings=var_emb["embs"],
                        lang=lang,
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
    return result, table_metadata


def call_dst_api(table_id, table_specs, lang):
    r = requests.post(
        "https://api.statbank.dk/v1/data",
        json={
            "table": table_id,
            "format": "BULK",
            "lang": lang,
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
    # FIXME: Sum up, what appears to be duplicated category values. Unsure if this is a bug in the API. But should work for now
    if df[df.columns[:-1]].duplicated().any():
        df = df.groupby(list(df.columns[:-1])).sum().reset_index()
    return df


def semantic_search(queries, ids, embeddings, lang, k=1):
    # Build faiss embedding index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Find k nearest neighbors of query embedding
    queries_emb = llm.embed(queries, lang=lang, small=True)
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
ACTION_SYS_MSG = Template(
    """Your job is to decide the type of request made by the user. The requests are made in order to extract information from a database of tables.

The request can be one of 2 types. 

1 - Specific question or query that should be answered.
2 - Update or change to existing request.

To help you decide the type of request you are provided with the previous API request and the table that API request was made to. If no previous request has been made then the input is an empty string.

Output the request type number. If the request type is of type 1 then add a concise table description. A table description consists of the subject of interest and optionally a by statement with the minimum needed variables the subject should be grouped by in order to answer the question.

For variables about geographic places such as city or municipality use the variable region.
For variables about time such as year or date use the variable time.
{% if lang == "da" %}
Write the table description in danish.
{% endif %}"""
)

ACTION_MSG_EN_EXAMPLE = [
    dict(
        role="user",
        content="""Previous table description: 
Previous API request:
User request: Which municipalities have the highest number of traffic accidents""",
    ),
    dict(
        role="assistant",
        content="1 - traffic accidents by region",
    ),
]

ACTION_USER_MSG = Template(
    """Previous table description: {{ table_description }}
Previous API request: {{ api_request }}
User request: {{ user_request }}"""
)

TABLE_SPECS_SYS_MSG = Template(
    """You are data analysis GPT. Your task is to filter a data table such that it shows the relevant information for answering a query.
In order to do so you choose the variables and their corresponding values that should be included. To make your decision you get information on the following form:

Query: "The query that should be answered by the table" 
Table description: "Description of the table"
Variables with few unique values:  [{"id": "the id of the variable", "text": "variable description", "values": [{"id": "the value id", "text": "value description"}], ... ]
Variables with many unique values: [{"id": "the id of the variable", "text": "variable description", "values": ["sample of 10 unique value texts"]
Time variable: {"id": "the id of the time variable", "text": "time variable description", "values": ["first time period", "10 latest time periods"]}

For all variables you can choose all values by writing ["*"]. 
For variables with few unique values you can choose a subset of values in the form of a list with the value ids. 
For variables with many unique values you can choose a subset of values in the form of a list with the likely value texts. 
For the Time variable you can also write ["latest"] to choose the latest time period only.

Before answering spend a few sentences explaining background context, assumptions, and step-by-step thinking. Keep it short and concise. If a total value is available for a variable then mention it. Remember a total can be an aggregate such as All Denmark or Age, total.
If the query does not specify the use of variable then choose the total of that variable.

Output the result as a formatted array of strings that can be used in JSON.parse(). Write Result: {"variable id": [], ... }.
Do not write anything after the result json.

If the table cannot be used to answer the query then instead of Result write "The best table match cannot be used to answer the query. Maybe you can find a better table match in the tree graph to the right".

Result: {"variable id": [], ... }
{% if lang == "da" %}
Write your answer in danish.
{% endif %}"""
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


# FIXME: Table look into fixing this update message. Consider maybe to see of you can get gpt-3.5-turbo to work.
TABLE_SPECS_UPDATE_SYS_MSG = """Your task is to update an API request to a data table such that it satisfies the user command.

In order to do so you get information on the following form:

Request: "Current API request"
Cmd: "User command"  
Table description: "Description of the table"
Variables with few unique values:  [{"id": "the id of the variable", "text": "variable description", "values": [{"id": "the value id", "text": "value description"}], ... ]
Variables with many unique values: [{"id": "the id of the variable", "text": "variable description", "values": ["sample of 10 unique value texts"]
Time variable: {"id": "the id of the time variable", "text": "time variable description", "values": ["10 latest time periods", "first time period"]}

For all variables you can choose all values by writing ["*"]. 
For variables with few unique values you can choose a subset of values in the form of a list with the value ids. 
For variables with many unique values you can choose a subset of values in the form of a list with the likely value texts. 
For the Time variable you can also write ["latest"] to choose the latest time period only.

Return updated API request."""

TABLE_SPECS_UPDATE_USER_MSG = Template(
    """Request: {{ request }}
Cmd: {{ cmd }}
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
