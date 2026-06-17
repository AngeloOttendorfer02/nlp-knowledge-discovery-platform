import json
import pandas as pd

df = pd.read_csv(
    "data/processed/processed_documents.csv",
    dtype={"doc_id": "string"},
)

available_columns = set(df.columns)

text_column = "text"

if text_column not in available_columns:
    if "abstract" in available_columns:
        text_column = "abstract"
    else:
        text_column = None

queries = []

for i, row in df.head(5).iterrows():

    if "title" in available_columns:
        query_text = str(row["title"])
    elif text_column:
        query_text = str(row[text_column])[:80]
    else:
        query_text = f"document {i+1}"

    queries.append(
        {
            "query_id": f"q{i+1}",
            "query": query_text,
            "relevant_doc_ids": [str(row["doc_id"])],
        }
    )

output_path = "data/evaluation/local_retrieval_queries.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(
        queries,
        file,
        indent=2,
        ensure_ascii=False,
    )

print(f"Created: {output_path}")
print(json.dumps(queries, indent=2, ensure_ascii=False))