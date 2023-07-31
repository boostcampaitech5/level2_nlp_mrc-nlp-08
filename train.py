import os
import pandas as pd
from transformers import (AutoConfig, AutoModelForQuestionAnswering,
                          AutoTokenizer, TrainingArguments, set_seed)
from dataset import Dataset
from QA_trainer import QuestionAnsweringTrainer
from utils_taemin import (compute_metrics, data_collators,
                          post_processing_function, run_sparse_retrieval)
from model import Custom_RobertaForQuestionAnswering

def main(model_name, data_path):
    
    eval_as_test = False
    skip_train = False
    base_model = AutoModelForQuestionAnswering # Custom_RobertaForQuestionAnswering
    bm25 = None # None(TF-IDF), "Okapi", "L", "plus"
    
    seed = 42
    set_seed(seed)

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = base_model.from_pretrained(model_name,config=config)
    train_data = Dataset(dataframe=pd.read_csv(os.path.join(data_path, "train_data.csv")), state="train", tokenizer=tokenizer)
    eval_data_df = pd.read_csv(os.path.join(data_path, "validation_data.csv"))

    if eval_as_test:
        datasets = run_sparse_retrieval(
            tokenize_fn=tokenizer.tokenize, data_path=data_path, datasets=eval_data_df.drop(["context"], axis=1), 
            bm25=bm25, context_path="wikipedia_documents.json"
        )
        extracted_context = datasets["validation"]["context"]
        eval_examples = eval_data_df.assign(context=extracted_context)
        eval_data = Dataset(dataframe=eval_examples, state="valid", tokenizer=tokenizer)

    else:
        eval_examples = eval_data_df
        eval_data = Dataset(dataframe=eval_examples, state="valid", tokenizer=tokenizer)

    data_collator = data_collators(tokenizer)

    args = TrainingArguments(
        output_dir=os.path.join(os.path.abspath(os.path.dirname(__file__)), "checkpoint"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=9e-6,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        dataloader_num_workers=4,
        logging_steps=50,
        seed=seed,
        group_by_length=True
    )
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        eval_examples=eval_examples,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    if skip_train:
        trainer.evaluate()
    else:
        trainer.train()

if __name__ == "__main__":
    model_name = 'klue/roberta-large'
    data_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "csv_data")
    main(model_name=model_name, data_path=data_path)