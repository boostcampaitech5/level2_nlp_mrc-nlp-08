import json
import os

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value, load_from_disk
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer

from retrieval import SparseRetrieval
from utils_taemin import run_sparse_retrieval

model_name = 'klue/roberta-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenize_fn = tokenizer.tokenize

dataset_path = "./data/"
context_path = "wikipedia_documents.json"
with open(os.path.join(dataset_path, context_path), "r", encoding="utf-8") as f:
    wiki = json.load(f)

contexts = list(
dict.fromkeys([v["text"] for v in wiki.values()])
)


datasets = pd.read_csv("./csv_data/test_data.csv")

k = 1
if k == 1:
    f = Features(
        {
            "context": Value(dtype="string", id=None),
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
        }
    )

datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})


bm25 = BM25Okapi(contexts)

a = bm25.get_top_n(datasets, contexts, n=40)