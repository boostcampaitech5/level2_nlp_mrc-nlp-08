import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, T5ForConditionalGeneration,
                          T5Tokenizer)

from t5_dataset import T5Dataset


def main():
    # Load pre-trained model and tokenizer
    model_name = "lcw99/t5-large-korean-text-summary"  # Example model, you can choose a different one
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'csv_data')

    train_data = pd.read_csv(os.path.join(DATA_DIR, 'train_data.csv'))
    validation_data = pd.read_csv(os.path.join(DATA_DIR, 'validation_data.csv'))
    #data2 = pd.read_csv(os.path.join(DATA_DIR, 'only_clean_val.csv'))

    #data = pd.concat([data,data2])
    # dataset_train, dataset_valid = train_test_split(data, test_size=0.1, random_state=42)

    #dataset_valid.to_csv('test_t5.csv',index=False)

    # Prepare your dataset as input-target pairs
    train_dataset = T5Dataset(dataframe=train_data, state="train", tokenizer=tokenizer)  # Your training dataset
    validation_dataset = T5Dataset(dataframe=validation_data, state="valid", tokenizer=tokenizer) # Your evaluation dataset

    #tokenized_inputs = [tokenizer.encode(input_seq) for input_seq in dataset_train['text']]
    #tokenized_targets = [tokenizer.encode(target_seq) for target_seq in dataset_train['text']]

    #val_tokenized_inputs = [tokenizer.encode(input_seq) for input_seq in dataset_valid['text']]
    #val_tokenized_targets = [tokenizer.encode(target_seq) for target_seq in dataset_valid['text']]

    #input_target_pairs = list(zip(tokenized_inputs, tokenized_targets))
    #val_input_target_pairs = list(zip(val_tokenized_inputs, val_tokenized_targets))

    label_pad_token_id = tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id
    )

    #train_dataset = CustomDataset(input_target_pairs)
    #val_dataset = CustomDataset(val_input_target_pairs)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(BASE_DIR, "checkpoint"),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        fp16=True,
        weight_decay=0.01,
        logging_steps = 50,
        dataloader_num_workers=0,
    )

    # Define the fine-tuning trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator = data_collator,
        # compute_metrics=compute_metrics
    )

    # Fine-tune the model
    trainer.train()

if __name__ == "__main__":
    main()
