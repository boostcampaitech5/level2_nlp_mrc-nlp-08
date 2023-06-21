import os

import pandas as pd

from transformers import (AutoConfig, AutoModelForQuestionAnswering,
                          AutoTokenizer, set_seed,
                          TrainingArguments)

from dataset import Dataset
from QA_trainer import QuestionAnsweringTrainer
from utils_taemin import (compute_metrics, data_collators,
                          post_processing_function)


def main(model_name, data_path):

    set_seed(42)

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name,config=config)

    train_data = Dataset(dataframe=pd.read_csv(os.path.join(data_path, "train_data.csv")), state="train", tokenizer=tokenizer)
    val_data = Dataset(dataframe=pd.read_csv(os.path.join(data_path, "validation_data.csv")), state="valid", tokenizer=tokenizer)

    data_collator = data_collators(tokenizer)

    args = TrainingArguments(
        output_dir=os.path.join(os.path.abspath(os.path.dirname(__file__)), "checkpoint"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        dataloader_num_workers=4,
        logging_steps=50,
        seed=42,
        group_by_length=True
    )
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        eval_examples=pd.read_csv(os.path.join(data_path, "validation_data.csv")),
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )
    trainer.train()

if __name__ == "__main__":
    model_name = 'klue/roberta-large'
    data_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "csv_data")
    main(model_name=model_name, data_path=data_path)
