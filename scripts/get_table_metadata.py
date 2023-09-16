# Start by importing necessary packages
import requests
import pandas as pd
from IPython.display import display
from io import StringIO
from collections import defaultdict
from time import sleep
from tqdm import tqdm
import pickle
from pathlib import Path

table_metadata = []
variables = defaultdict(list)
fps = list(Path("/Users/josca/projects/dstGPT/data/tables_info").glob("*.pkl"))
for fp in tqdm(fps, total=len(fps)):
    r = pickle.load(open(fp, "rb"))
    r_variables = r.pop("variables")
    table_metadata.append(r.copy())
    for var in r_variables:
        if var["id"] not in variables:
            variables[var["id"]].append(
                {var["id"]: {"values": var["values"], "tables": [fp.stem]}}
            )
        else:
            in_tables = False
            for tables in variables[var["id"]]:
                if var["id"] == "Tid":
                    pass

                tables = tables[var["id"]]
                vals = tables["values"]

                set_list1 = [frozenset(d.items()) for d in var["values"]]
                set_list2 = [frozenset(d.items()) for d in vals]

                if set(set_list1) == set(set_list2):
                    tables["tables"].append(fp.stem)
                    in_tables = True
                    break
                else:
                    pass

            if not in_tables:
                variables[var["id"]].append(
                    {var["id"]: {"values": var["values"], "tables": [fp.stem]}}
                )


pickle.dump(variables, open("./data/variables.p", "wb"))
pickle.dump(table_metadata, open("./data/table_metadata.p", "wb"))
