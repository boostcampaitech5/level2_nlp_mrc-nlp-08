import os
from typing import Callable, Dict, List, Tuple

import evaluate
import pandas as pd
import transformers
from datasets import (Dataset, DatasetDict, Features, Sequence, Value,
                      load_from_disk, metric)
from numpy import array
from transformers import (AutoTokenizer, DataCollatorWithPadding,
                          EvalPrediction, TrainingArguments)

from arguments import DataTrainingArguments
from data_preprocessing import Preprocess
from retrieval import SparseRetrieval
from retrieval_bm25 import SparseRetrievalBM25
from utils_qa import postprocess_qa_predictions


def data_collators(tokenizer):
    data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8
        )
    return data_collator
def compute_metrics(p: EvalPrediction):
    metric = evaluate.load("squad")
    return metric.compute(predictions=p.predictions, references=p.label_ids)

def post_processing_function(examples, features, predictions, training_args):
    # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
    # datasets = load_from_disk(os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/train_dataset/"))
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        max_answer_length=30,
        output_dir=training_args.output_dir,
    )
    # Metric을 구할 수 있도록 Format을 맞춰줍니다.
    formatted_predictions = [
        {"id": k, "prediction_text": v} for k, v in predictions.items()
    ]
    if training_args.do_predict:
        return formatted_predictions

    elif training_args.do_eval:
        references = [
            {"id": ex["id"], "answers": eval(ex['answers'])}
            # for ex in datasets["validation"]
            for _, ex in examples.iterrows()
        ]
        return EvalPrediction(
            predictions=formatted_predictions, label_ids=references
        )

def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: pd.DataFrame,
    data_path: str = os.path.join(os.path.abspath(os.path.dirname(__file__)), "csv_data"),
    context_path: str = "wikipedia_documents.json",
    bm25: bool = False,
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    if bm25:
        retriever = SparseRetrievalBM25(
            tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
        )
    else:
        retriever = SparseRetrieval(
            tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
        )    

    retriever.get_sparse_embedding()
    df = retriever.retrieve(datasets, topk=40)

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    k = 1
    if k==1:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif k==0:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets
