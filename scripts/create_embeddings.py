import pickle
import faiss
import numpy as np
import pandas as pd
from llm import embed

table_metadata = pickle.load(open("data/table_metadata.p", "rb"))
df_table = pd.DataFrame(table_metadata)

embeddings = embed(df_table["description"].to_list())

embeddings = np.array(embeddings)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(
    index, "/Users/josca/projects/dstGPT/data/indexes/table_description_en_emb.bin"
)
