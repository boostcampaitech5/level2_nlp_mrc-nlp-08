import evaluate
import transformers
from transformers import DataCollatorWithPadding,AutoTokenizer,EvalPrediction
from datasets import load_from_disk, metric
from data_preprocessing import Preprocess
from retrieval import SparseRetrieval
from utils_qa import postprocess_qa_predictions
from typing import Callable, Dict, List, Tuple
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk
)
from datasets import DatasetDict, Features
from transformers import TrainingArguments

from arguments import DataTrainingArguments

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
    datasets = load_from_disk('./data/train_dataset')
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
            {"id": ex["id"], "answers": ex['answers']}
            for ex in datasets["validation"]
        ]
        return EvalPrediction(
            predictions=formatted_predictions, label_ids=references
        )

def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    data_path: str = "./data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = SparseRetrieval(
        tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
    )
    retriever.get_sparse_embedding()
    p = 1
    if p==0:
        retriever.build_faiss(num_clusters=64)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=40
        )
    else:
        df = retriever.retrieve(datasets["validation"], topk=40)

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