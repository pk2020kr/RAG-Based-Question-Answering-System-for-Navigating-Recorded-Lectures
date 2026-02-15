import requests
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding


# Only process actual .json files and skip non-JSON / non-UTF8 files like .DS_Store
jsons = [f for f in os.listdir("jsons") if f.lower().endswith(".json")]
my_dicts = []
chunk_id = 0

for json_file in jsons:
    path = os.path.join("jsons", json_file)
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f)
    except UnicodeDecodeError:
        print(f"Skipping {path}: cannot decode as UTF-8.")
        continue
    except json.JSONDecodeError:
        print(f"Skipping {path}: invalid JSON.")
        continue

    print(f"Creating Embeddings for {json_file}")
    chunks = content.get("chunks", [])
    if not chunks:
        print(f"Skipping {json_file}: no chunks found.")
        continue

    embeddings = create_embedding([c.get('text', '') for c in chunks])
    if not embeddings or len(embeddings) != len(chunks):
        print(f"Embedding failed or mismatch for {json_file}, skipping.")
        continue

    for i, chunk in enumerate(chunks):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)

df = pd.DataFrame.from_records(my_dicts)
# Save this dataframe
joblib.dump(df, 'embeddings.joblib')

