import os
import json
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pprint import pprint
from time import strftime
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


from transformers import (
    AutoConfig, AutoModelForQuestionAnswering,
    AutoTokenizer,DataCollatorWithPadding,
    BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    HfArgumentParser, Trainer,
    TrainingArguments,
)

from dev_dpr import set_seed
from dev_dpr import  DenseRetrieval
from dev_dpr import BertEncoder


from arguments import DataTrainingArguments, ModelArguments
from data_preprocessing import Preprocess
from dataset import Dataset

from datasets import (Dataset, DatasetDict, Features, Sequence, Value,
                      load_from_disk, metric)

from QA_trainer import QuestionAnsweringTrainer
from utils import config_parser
from utils_taemin import (compute_metrics, data_collators,
                          post_processing_function, run_sparse_retrieval)


if __name__ == "__main__":
    set_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 데이터셋과 모델은 아래와 같이 불러옵니다.
    data_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "csv_data")
    train_dataset = pd.read_csv(os.path.join(data_path, "squad_kor_v1.csv"))
    train_dataset = train_dataset

    output_dir=os.path.join(os.path.abspath(os.path.dirname(__file__)), "dense_retireval")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01
    )  

    model_checkpoint = '/opt/ml/input/code/level2_nlp_mrc-nlp-08/dense_retireval/2023.06.21 - 03:13:45/epochs:2/'
    # model_checkpoint = "klue/roberta-large"    
    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    # p_encoder = BertEncoder.from_pretrained(model_checkpoint+ 'p_encoder.pt').to(args.device)
    # q_encoder = BertEncoder.from_pretrained(model_checkpoint +'q_encoder.pt').to(args.device)
    p_encoder = torch.load(model_checkpoint + 'p_encoder.pt')
    q_encoder = torch.load(model_checkpoint + 'q_encoder.pt')
    retriever = DenseRetrieval(args=args, dataset=train_dataset[:-1], num_neg=3, tokenizer=tokenizer, p_encoder=p_encoder, q_encoder=q_encoder)


    query = pd.read_csv('/opt/ml/input/code/level2_nlp_mrc-nlp-08/csv_data/test_data.csv')
    # results = retriever.get_relevant_doc(query=query, k=1)
    
    # print(f"[Search Query] {query}\n")

    # indices = results.tolist()
    # for i, idx in enumerate(indices):
    #     print(f"Top-{i + 1}th Passage (Index {idx})")
    #     pprint(retriever.dataset['context'][idx])

    datasets = retriever.get_relevant_doc(query=query, k=40)

    examples = datasets["validation"].to_pandas()

    model_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), "checkpoint/checkpoint-2994")
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name,config=config)
    test_data = Preprocess(tokenizer=tokenizer,dataset=datasets['validation'],state='val').output_data
    data_collator = data_collators(tokenizer)
    

    args = TrainingArguments(
        output_dir=os.path.join(os.path.abspath(os.path.dirname(__file__)), "output"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.1,
        dataloader_num_workers=0,
        logging_steps=50,
        seed=42,
        group_by_length=True,
        do_eval=False,
        do_predict=True
    )

    trainer = QuestionAnsweringTrainer(
        model=model,
        args=args,
        train_dataset=None,
        eval_dataset=test_data,
        eval_examples=examples,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    predictions = trainer.predict(
        test_dataset=test_data, test_examples=examples
    )
