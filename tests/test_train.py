import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pandas as pd
from dataset import Dataset
from datasets import load_from_disk
from QA_trainer import QuestionAnsweringTrainer
from transformers import (AutoConfig, AutoModelForQuestionAnswering,
                          AutoTokenizer, TrainingArguments)
from utils_taemin import (compute_metrics, data_collators,
                          post_processing_function)


def test_train():
    os.environ["WANDB_DISABLED"] = "true"

    model_name = 'klue/roberta-small'

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name,config=config)

    # train_data = Preprocess(tokenizer=tokenizer,dataset=datasets['train'],state='train').output_data
    # val_data = Preprocess(tokenizer=tokenizer,dataset=datasets['validation'],state='val').output_data

    train_data = Dataset(dataframe=pd.read_csv(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../dummy_data/train_data.csv"))[:16], state="train", tokenizer=tokenizer)
    val_data = Dataset(dataframe=pd.read_csv(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../dummy_data/validation_data.csv"))[:4], state="valid", tokenizer=tokenizer)

    data_collator = data_collators(tokenizer)

    args = TrainingArguments(
        output_dir=os.path.join(os.path.abspath(os.path.dirname(__file__)), "../checkpoint"),
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
        eval_examples=pd.read_csv(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../dummy_data/validation_data.csv"))[:4],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )
    trainer.train()

def test_checkpoint():
    assert os.path.isdir("../checkpoint")
