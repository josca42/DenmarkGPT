# Start by importing necessary packages
import requests
import pandas as pd
from IPython.display import display
from io import StringIO
from collections import defaultdict
from time import sleep
from tqdm import tqdm
import pickle

# from dstapi import DstApi # The helper class
r = requests.get("https://api.statbank.dk/v1/tables", params={"lang": "en"}).json()

df_tables = pd.DataFrame(r)
table_metadata = []
variables = defaultdict(list)
for i, table in tqdm(df_tables.iterrows(), total=len(df_tables)):
    sleep(5)
    r = requests.get(
        "https://api.statbank.dk/v1" + "/tableinfo",
        params={"id": table["id"], "format": "JSON", "lang": "en"},
    ).json()
    r_variables = r.pop("variables")
    table_metadata.append(r.copy())
    for var in r_variables:
        if var["id"] not in variables:
            variables[var["id"]].append(
                {var["id"]: {"values": var["values"], "tables": [table["id"]]}}
            )
        else:
            for tables in variables[var["id"]]:
                tables = tables[var["id"]]
                vals = tables["values"]

                set_list1 = [frozenset(d.items()) for d in var["values"]]
                set_list2 = [frozenset(d.items()) for d in vals]

                if set(set_list1) == set(set_list2):
                    tables["tables"].append(table["id"])
                else:
                    variables[var["id"]].append(
                        {var["id"]: {"values": var["values"], "tables": [table["id"]]}}
                    )

pickle.dump(variables, open("./data/variables.p", "wb"))
pickle.dump(table_metadata, open("./data/table_metadata.p", "wb"))
