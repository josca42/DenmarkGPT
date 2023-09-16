import openai
import os
from jinja2 import Template
from datetime import datetime
import numpy as np
from typing import Union

from dotenv import load_dotenv
import json
import faiss
import numpy as np
import pandas as pd
import cohere
import os

load_dotenv()

LLM_cohere = cohere.Client(os.environ["COHERE_API_KEY"])


def embed(texts: Union[list[str], str], model="cohere"):
    if isinstance(texts, str):
        texts = [texts]
    texts = [text.replace("\n", " ") for text in texts]

    response = LLM_cohere.embed(
        texts=texts,
        model="embed-english-v2.0",
    )
    embeddings = response.embeddings
    return embeddings


def match_port(port):
    query_embedding = embed([port])
    D, I = ports_index.search(np.array(query_embedding).astype("float32"), 1)
    return gdf_ports.iloc[I[0]]["name"].squeeze()
