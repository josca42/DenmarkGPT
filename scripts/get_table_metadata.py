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

# Extract all variable data into dictionary structure
# and all table data into pandas dataframe
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


# Create dictionary with all variable values
data_dict = {}
for variable, tables in variables.items():
    if tables:
        if variable not in data_dict:
            data_dict[variable] = {}

        for table in tables:
            values = table[variable]["values"]
            for value in values:
                data_dict[variable][value["text"]] = None


fps = list(Path("/Users/josca/projects/dstGPT/data/tables_info").glob("*.pkl"))
for fp in fps:
    new_dir = Path("/Users/josca/projects/dstGPT/data/tables_info") / fp.stem
    new_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(fp), str(new_dir / "info.pkl"))


i = 0
for var, v in data_dict.items():
    if var in ALREADY_DONE:
        continue

    print(i, var, len(v))

    values = list(v.keys())
    value_chunks = [values[i : i + 70] for i in range(0, len(values), 70)]
    embeddings = []
    for vals in value_chunks:
        embeddings += embed(vals)

    text2emb = {text: emb for text, emb in zip(values, embeddings)}
    pickle.dump(text2emb, open(f"./data/variable_emb/{var}.p", "wb"))


TABLES_DIR = Path(os.environ["DATA_DIR"]) / "tables_info"
VAR_DIR = Path(os.environ["DATA_DIR"]) / "variable_emb"


for table_dir in tqdm(TABLES_DIR.iterdir(), total=len(list(TABLES_DIR.iterdir()))):
    table_info = pickle.load(open(table_dir / "info.pkl", "rb"))

    vars_embs = {}
    for variable in table_info["variables"]:
        var_emb = pickle.load(open(VAR_DIR / f"{variable['id']}.p", "rb"))

        ids, embs = [], []
        for value in variable["values"]:
            ids.append(value["id"])
            embs.append(var_emb[value["text"]])

        embs = np.array(embs)
        vars_embs[variable["id"]] = {"ids": ids, "embs": embs}

    pickle.dump(vars_embs, open(table_dir / "vars_embs.pkl", "wb"))
