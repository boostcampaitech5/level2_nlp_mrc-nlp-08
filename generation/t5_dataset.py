import random
import string

import pandas as pd
import torch
from numpy import array
from torch.utils.data import Dataset


class T5Dataset(Dataset):
    def __init__(
        self, 
        dataframe,
        state,
        tokenizer
    ):
        dataframe.question = "질문: " + dataframe.question
        dataframe.context = " 본문: " + dataframe.context
        self.dataframe = dataframe
        self.state = state
        self.tokenizer = tokenizer
        self.pad_on_right = tokenizer.padding_side == "right"

        column_names = self.dataframe.columns.tolist()

        self.question_column_name = "question" if "question" in column_names else column_names[0]
        self.context_column_name = "context" if "context" in column_names else column_names[1]

        if self.state == "test":
            self.dataset = self.test_preprocessing(self.dataframe)
        else:
            self.dataset = self.train_preprocessing(self.dataframe)

    def __getitem__(self, idx):
        if self.state == "test":
            return {
                "input_ids": torch.tensor(self.dataset["input_ids"][idx]),
                "attention_mask": torch.tensor(self.dataset["attention_mask"][idx]),
                "example_id": self.dataset["example_id"][idx]
            }
        else:
            return {
                'input_ids': torch.tensor(self.dataset['input_ids'][idx]),
                'attention_mask': torch.tensor(self.dataset['attention_mask'][idx]),
                'labels': torch.tensor(self.dataset["labels"][idx]),
                "example_id": self.dataset["example_id"][idx]
            }

    def __len__(self):
        return len(self.dataset["input_ids"])

    def train_preprocessing(self, dataframe: pd.DataFrame):
        tokenized_examples = self.tokenizer(
            dataframe[self.question_column_name if self.pad_on_right else self.context_column_name].tolist(),
            dataframe[self.context_column_name if self.pad_on_right else self.question_column_name].tolist(),
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=384,
            stride=128,
            return_overflowing_tokens=True,
            return_token_type_ids=False,
            padding="max_length"
        )

        labels = self.tokenizer(
            [eval(answer)["text"][0] for answer in dataframe["answers"]],
            max_length=128,
            padding="max_length",
            truncation=True
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        tokenized_examples["labels"] = []
        tokenized_examples["example_id"] = []

        for sample_index in sample_mapping:
            tokenized_examples["labels"].append(labels["input_ids"][sample_index])
            tokenized_examples["example_id"].append(dataframe["id"][sample_index])

        return tokenized_examples



    def test_preprocessing(self,dataframe):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = self.tokenizer(
            dataframe[self.question_column_name if self.pad_on_right else self.context_column_name].tolist(),
            dataframe[self.context_column_name if self.pad_on_right else self.question_column_name].tolist(),
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=384,
            stride=128,
            return_overflowing_tokens=True,
            return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length"
        )

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
        # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(dataframe["id"][sample_index])
            
        return tokenized_examples
