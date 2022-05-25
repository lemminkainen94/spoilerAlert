import argparse
import math
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForMaskedLM, AdamW, TrainingArguments, Trainer


def group_texts(examples):
    # Concatenate all texts.
    print(examples)
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def parse_args():
    parser = argparse.ArgumentParser("GoodreadsMLM")
    parser.add_argument('data', type=str, help='path to data file')
    parser.add_argument('--model_name', type=str, default='bert-base-cased',
                        help='Huggingface Transformers pretrained model. bert-base-cased by default' +
                        'Can also try google/electra-small-discriminator or distilbert-base-(un)cased')
    parser.add_argument('--out_model', default=None, help="Name of the trained model (will be saved with that name). Used for training only")
    parser.add_argument("--block_size", default=128, type=int,
                        help="The input input text will be divided into blocks of block_size length after concatenation.")
    parser.add_argument('--acc_steps',
                        type=int,
                        default=1,
                        help="gradient accumulation steps")
    parser.add_argument('--epochs',
                        type=int,
                        default=5,
                        help="number of training epochs")
    parser.add_argument('--lr',
                        type = float,
                        default = 0.003,
                        help = "Adam (for LSTM) or AdamW (for transformer-based) learning rate. Default is 0.003.")

    return parser.parse_args()


def main():
    args = parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = pd.read_pickle(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_text = df.sentence.apply(
        lambda x: tokenizer.encode(x, add_special_tokens=False)
    )

    lm_data = [x for list_ in tokenized_text.values for x in list_]    
    total_len = (len(lm_data) // args.block_size) * args.block_size
    lm_blocks = [lm_data[i : i + args.block_size] for i in range(0, total_len, args.block_size)]
    lm_data = [{'input_ids': x, 'labels': x, 'attention_mask': [1 for _ in x]} for x in lm_blocks]
    print(tokenizer.decode(lm_data[0]['input_ids']))

    lm_train, lm_eval = train_test_split(lm_data, test_size=0.05)

    model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    args.model_name = args.model_name.split("/")[-1]
    training_args = TrainingArguments(
        f"models/{args.model_name}-goodreads",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        weight_decay=0.01
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_train,
        eval_dataset=lm_eval,
        data_collator=data_collator,
    )

    eval_results = trainer.evaluate()
    print(f"Perplexity before: {math.exp(eval_results['eval_loss']):.2f}")

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Perplexity after: {math.exp(eval_results['eval_loss']):.2f}")

if __name__ == "__main__":
    main()
