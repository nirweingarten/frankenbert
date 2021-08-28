import argparse
import os
from datetime import datetime
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
import math
import sys
import utils
import torch
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import AutoModelWithLMHead, AutoTokenizer, top_k_top_p_filtering
import transformers
from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML
import torch.nn.functional as F


def train(model_name, task, dataset_name):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    datasets = load_dataset(*dataset_name.split(','))
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenized_datasets = datasets.map(lambda examples: tokenizer(examples["text"]),
                                      batched=True, num_proc=2, remove_columns=["text"])
    lm_datasets = tokenized_datasets.map(
                        utils.group_texts,
                        batched=True,
                        batch_size=1000,
                        num_proc=2,
                        )
    training_args = TrainingArguments(
        "test-clm",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
    )
    if task == 'causal':
        model = AutoModelForCausalLM.from_pretrained(model_name)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_datasets["train"],
            eval_dataset=lm_datasets["validation"],
        )
    elif task == 'MLM':
        model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_datasets["train"],
            eval_dataset=lm_datasets["validation"],
            data_collator=data_collator
        )
    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    return model


def main(raw_args):
    parser = argparse.ArgumentParser(description='Train a model and pickle it!\n'
                                             'List of MLM models: https://huggingface.co/models?filter=masked-lm\n'
                                             'List of causal models: https://huggingface.co/models?filter=causal-lm\n'
                                             'List of datasets: https://huggingface.co/datasets')
    parser.add_argument(
        '--model_name', '-m', type=str, nargs='?', help='Type of pre trained model', default='bert-base-uncased')
    parser.add_argument(
        '--task', '-t', type=str, nargs='?', help='Type of task. either MLM or causal', default='MLM')
    parser.add_argument(
        '--dataset', '-d', type=str, nargs='?', help='Dataset to use. can be more than one word for *args,\n'
                                                     'for example: \'wikitext,wikitext-2-raw-v1\' will be parsed as\n'
                                                     '[\'wikitext\',\'wikitext-2-raw-v1\']')
    parser.add_argument('--save_dir', '-s', type=str, nargs='?', help='Path of dir to save model in')

    args = parser.parse_args(raw_args)
    assert os.path.isdir(args.save_dir)
    timestamp = datetime.now().strftime('%y%m%d%H%m')
    save_path = os.path.join(args.save_dir, '{0}_{1}_{2}.pkl'.format(args.model_name,
                                                                     args.dataset.split(',')[0], timestamp))
    model = train(args.model_name, args.task, args.dataset)
    torch.save(model, save_path)
    print('Saved model to {}'.format(save_path))

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

