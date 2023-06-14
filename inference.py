#Code refactoringe by Tae min Kim
import os

import pandas as pd
from datasets import DatasetDict, load_from_disk
from transformers import (AutoConfig, AutoModelForQuestionAnswering,
                          AutoTokenizer, DataCollatorWithPadding,
                          HfArgumentParser, Trainer, TrainingArguments)

from arguments import DataTrainingArguments, ModelArguments
from data_preprocessing import Preprocess
from dataset import Dataset
from QA_trainer import QuestionAnsweringTrainer
from utils import config_parser
from utils_taemin import (compute_metrics, data_collators,
                          post_processing_function, run_sparse_retrieval)


def main():

    model_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), "checkpoint/checkpoint-2994")

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name,config=config)

    datasets = run_sparse_retrieval(
        tokenizer.tokenize, pd.read_csv(os.path.join(os.path.abspath(os.path.dirname(__file__)), "csv_data/test_data.csv")),
    )

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
        group_by_length=True
    )
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=args,
        train_dataset=None,
        eval_dataset=test_data,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    predictions = trainer.predict(
        test_dataset=test_data, test_examples=datasets["validation"]
    )

    print(1)
if __name__ == "__main__":
    main()
