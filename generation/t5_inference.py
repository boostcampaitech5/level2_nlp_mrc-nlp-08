import json
import os
import sys
from collections import defaultdict

import numpy as np
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq)
from utils_taemin import run_sparse_retrieval

from t5_dataset import T5Dataset


def length_penalty(length, alpha=1.2, min_length=5):
    return ((min_length + length) / (min_length + 1)) ** alpha

def compute_cumulative_prob(sequence, score):
    # total_prob = 1
    total_prob = 0
    len = 0
    for seq, sc in zip(sequence[1:], score):
        if seq == 1:
            break
        len += 1
        # total_prob *= np.exp(sc.cpu().numpy())
        total_prob += sc.cpu().numpy()
    return np.exp(total_prob / length_penalty(len))

def main():

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'csv_data')
    model_name = os.path.join(BASE_DIR, "checkpoint/checkpoint-4077")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    datasets = run_sparse_retrieval(
        tokenize_fn=tokenizer.tokenize, data_path=DATA_DIR, datasets=pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"))
    )

    test_data = datasets["validation"].to_pandas()

    test_dataset = T5Dataset(dataframe=test_data, state="test", tokenizer=tokenizer)

    # label_pad_token_id = tokenizer.pad_token_id
    # data_collator = DataCollatorForSeq2Seq(
    #     tokenizer,
    #     model=model,
    #     label_pad_token_id=label_pad_token_id
    # )

    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # prediction = defaultdict(lambda: ("", 0))
    prediction = dict()
    # prediction = defaultdict(list)

    for sample in tqdm(test_dataloader):
        example_id = sample.pop("example_id")
        sample["input_ids"] = sample["input_ids"].to(device)
        sample["attention_mask"] = sample["attention_mask"].to(device)
        with torch.no_grad():
            output = model.generate(
                **sample,
                num_beams=4,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                return_dict_in_generate=True,
                output_scores=True
            ) # output.sequences, output.scores
            output_texts = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
            # print(output_texts)

            transition_scores = model.compute_transition_scores(output.sequences, output.scores, output.beam_indices, normalize_logits=True)
            result = list(zip(output_texts, [compute_cumulative_prob(output.sequences[i], transition_scores[i]) for i in range(len(transition_scores))]))
        for id, res in zip(example_id, result):
            # if prediction[id][1] < res[1]:
            #     prediction[id] = res

            if id not in prediction:
                prediction[id] = res
            else:
                if prediction[id][1] < res[1]:
                    prediction[id] = res

            # prediction[id].append(res)

    for key, value in prediction.items():
        prediction[key] = value[0]

    os.makedirs("./output", exist_ok=True)
    with open("./output/prediction.json", "w", encoding="utf-8") as f:
        json.dump(prediction, f, ensure_ascii=False)

    # args = TrainingArguments(
    #     output_dir=os.path.join(BASE_DIR, "output"),
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     learning_rate=2e-5,
    #     per_device_train_batch_size=8,
    #     per_device_eval_batch_size=8,
    #     num_train_epochs=3,
    #     weight_decay=0.1,
    #     dataloader_num_workers=0,
    #     logging_steps=50,
    #     seed=42,
    #     group_by_length=True,
    #     do_eval=False,
    #     do_predict=True
    # )
    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     args=args,
    #     train_dataset=None,
    #     eval_dataset=test_dataset,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     post_process_function=post_processing_function,
    #     # compute_metrics=compute_metrics,
    # )

    # trainer.predict(test_dataset=test_dataset)

if __name__ == "__main__":
    main()
