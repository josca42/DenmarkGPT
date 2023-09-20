import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import geopandas as gpd
import numpy as np
import faiss
import networkx as nx
import cohere
import openai
import pickle

load_dotenv()

DATA_DIR = Path(os.environ["DATA_DIR"])


LLM_cohere = cohere.Client(os.environ["COHERE_API_KEY"])
openai.api_key = os.environ.get("OPENAI_API_KEY")

gdf = gpd.read_file(DATA_DIR / "maps/kommune_og_region.shp")
REGIONER_ID = [x for x in gdf["geo_id"] if x[0] == "0"]
KOMMUNER_ID = [x for x in gdf["geo_id"] if x[0] != "0"]


LLM_cohere = cohere.Client(os.environ["COHERE_API_KEY"])

INDEX_DIR = DATA_DIR / "indexes"
TABLE_INFO_DIR = DATA_DIR / "tables_info"

df_table = (
    pd.read_parquet(INDEX_DIR / "table_info.parquet").set_index("id").sort_index()
)
table_ids = np.load(INDEX_DIR / "tables_id.npy", allow_pickle=True)
table_indexes = {}
table_indexes["en_text"] = faiss.read_index(str(INDEX_DIR / "table_text_en_emb.bin"))
table_indexes["en_description"] = faiss.read_index(
    str(INDEX_DIR / "table_description_en_emb.bin")
)


G = nx.read_gml(DATA_DIR / "table_network" / "subjects_graph.gml")
G.remove_node("0")
table2node = pickle.load(open(DATA_DIR / "table_network" / "table2node.pkl", "rb"))
