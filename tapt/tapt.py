import os
import sys

import torch
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling,
                          T5ForConditionalGeneration, Trainer,
                          TrainingArguments)

from dataset import TaptDataSet


def main():
    MODEL_NAME = "paust/pko-t5-large"

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    BASE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({"mask_token": '[MASK]'})
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    dataset = TaptDataSet(
        [os.path.join(BASE_DIR, "csv_data/train_data.csv"), os.path.join(BASE_DIR, "csv_data/validation_data.csv"), os.path.join(BASE_DIR, "csv_data/test_data.csv")], tokenizer=tokenizer
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=f"./{MODEL_NAME}_TAPT",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=1e-6,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(f"./{MODEL_NAME}_TAPT_output")


if __name__ == "__main__":
    main()
