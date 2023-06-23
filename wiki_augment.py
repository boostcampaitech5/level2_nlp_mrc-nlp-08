import builtins
import os

import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from retrieval import SparseRetrieval


def stop_print(*args, **kwargs):
    pass

def get_external_data(shuffle=True, seed=42, idx=0, count=10000):
    """
    HuggingFace에서 외부 데이터셋을 불러온 후, 지정한 개수만큼 반환하는 함수
        Args:
            shuffle (bool): 외부 데이터셋을 불러온 후 데이터의 순서를 섞을지 여부를 지정
            seed (int): 데이터셋 셔플에 사용할 랜덤 시드
            idx (int): 불러온 데이터의 어느 인덱스부터 반환할지를 지정
            count (int): 증강할 데이터의 개수를 지정
        Returns:
            aug_dataset (dict): 증강에 사용할 데이터셋
    """
    if shuffle:
        raw_dataset = load_dataset("graelo/wikipedia", "20230601.ko")["train"].shuffle(seed)
    else:
        raw_dataset = load_dataset("graelo/wikipedia", "20230601.ko")["train"]
    aug_dataset = raw_dataset[idx:idx + count]
    return aug_dataset

def setup_retrieval(model_name, context_path):
    tokenizer_fn = AutoTokenizer.from_pretrained(model_name).tokenize 
    retriever = SparseRetrieval(
        tokenize_fn=tokenizer_fn, data_path="./csv_data", context_path="wikipedia_documents.json"
    )

    retriever.get_sparse_embedding()
    return retriever

def filter_data(retriever, dataset, raw_data_length, threshold=0.7):
    texts = []
    titles = []
    document_ids = []
    document_id_start = raw_data_length + 1

    for i, data in enumerate(tqdm(dataset["text"])):
        score, doc = retriever.retrieve(data, topk=1)
        print(f"{dataset['title'][i]} appended. (Top Score: {score[0]})")
        if score[0] < threshold:
            texts.append(data)
            titles.append(dataset["title"][i])
            document_ids.append(document_id_start)
            document_id_start += 1
    assert len(texts) == len(titles) == len(document_ids), "Length mismatch."

    data_len = len(texts)

    filtered_data = pd.DataFrame({"text": texts,
                         "corpus_source": ["위키피디아"] * data_len,
                         "url": ["TODO"] * data_len,
                         "domain": [None] * data_len,
                         "title": titles,
                         "author": [None] * data_len,
                         "html": [None] * data_len,
                         "document_id": document_ids})
    return filtered_data

def concat_data(raw_data, aug_data):
    output_data = pd.concat([raw_data, aug_data])
    output_data.reset_index(inplace=True, drop=True)

    print(f"Raw Data Count: {len(raw_data)}")
    print(f"Output Data Count: {len(output_data)} (+{len(aug_data)} augmented)")
    return output_data

def export_data(output_data, aug_data=None, data_path="./csv_data"):
    output_json = output_data.transpose()
    output_json.to_json(os.path.join(data_path, "./wikipedia_documents_augmented.json"), force_ascii=False) 

    aug_json = aug_data.transpose()
    aug_json.to_json(os.path.join(data_path, "./wikipedia_documents_augmented_only.json"), force_ascii=False)

def main():
    raw_print = builtins.print
    model_name = "klue/roberta-large"
    
    retrieval = setup_retrieval(model_name=model_name, context_path="./csv_data/wikipedia_documents.json")
    builtins.print = stop_print

    raw_json = pd.read_json("./csv_data/wikipedia_documents.json")
    raw_data = raw_json.transpose() # dataframe에서 작업할 수 있도록 전치 수행

    raw_data_length = len(raw_data)
    aug_data = get_external_data(count=10)
    aug_data_filtered = filter_data(retriever=retrieval, dataset=aug_data, raw_data_length=raw_data_length, threshold=0.7)

    builtins.print = raw_print

    output_data = concat_data(raw_data, aug_data_filtered)
    export_data(output_data, aug_data=aug_data_filtered, data_path="./csv_data")

if __name__ == "__main__":
    main()