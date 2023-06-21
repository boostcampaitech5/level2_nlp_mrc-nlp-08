import os

import nltk
import pandas as pd
import torch
from evaluate import load
from numpy import array
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, EvalPrediction,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          T5ForConditionalGeneration, T5Tokenizer)

from t5_dataset import T5Dataset


def main():
    nltk.download('punkt')
    # Load pre-trained model and tokenizer
    # model_name = "lcw99/t5-large-korean-text-summary"  # Example model, you can choose a different one
    model_name = "paust/pko-t5-large"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
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

    def postprocess_text(preds, labels):

        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        metric = load("squad")
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # decoded_labels은 rouge metric을 위한 것이며, f1/em을 구할 때 사용되지 않음
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 간단한 post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        formatted_predictions = [{"id": ex["id"], "prediction_text": decoded_preds[i]} for i, ex in validation_data.iterrows()]
        references = [{"id": ex["id"], "answers": eval(ex["answers"])} for _, ex in validation_data.iterrows()]

        result = metric.compute(predictions=formatted_predictions, references=references)
        return result

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
        predict_with_generate=True,
        weight_decay=0.01,
        logging_steps = 50,
        dataloader_num_workers=0,
        save_total_limit=1
    )

    # Define the fine-tuning trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Fine-tune the model
    trainer.train()

if __name__ == "__main__":
    main()

# {'predictions': {'id': Value(dtype='string', id=None), 'prediction_text': Value(dtype='string', id=None)}, 
# 'references': {'id': Value(dtype='string', id=None), 'answers': Sequence(feature={'text': Value(dtype='string', id=None), 'answer_start': Value(dtype='int32', id=None)}, length=-1, id=None)}}
