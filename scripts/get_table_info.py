import requests
import pandas as pd
from io import StringIO
from collections import defaultdict
from time import sleep
from tqdm import tqdm
import pickle
import os

r = requests.get("https://api.statbank.dk/v1/tables", params={"lang": "da"}).json()
df_tables = pd.DataFrame(r)

table_metadata = []
variables = defaultdict(list)
for i, table in tqdm(df_tables.iterrows(), total=len(df_tables)):
    sleep(10)
    r = requests.get(
        "https://api.statbank.dk/v1" + "/tableinfo",
        params={"id": table["id"], "format": "JSON", "lang": "da"},
    ).json()
    # Create a directory with the table id as the name
    directory = "/Users/josca/projects/dstGPT/data/tables_info_da/" + table["id"]
    os.makedirs(directory, exist_ok=True)

    # Pickle dump the response to the directory with the name info.pkl
    with open(directory + "/info.pkl", "wb") as f:
        pickle.dump(r, f)
