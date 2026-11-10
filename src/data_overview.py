# Module to explore the data file

import pandas as pd

data = pd.read_json("data/sql_create_context_v4.json")

print(data.head())
data["n_tables"] = data.apply(lambda row: row["context"].count(";") + 1, axis=1)
print(data.groupby("n_tables").size())
