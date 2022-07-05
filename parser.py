from bs4 import BeautifulSoup
import json
import re
import pandas as pd
import numpy as np
import torch

from nltk import tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from flask import Flask, request
from flask_cors import CORS
from types import SimpleNamespace
from transformers import AutoTokenizer

from src.transformer_model import SpoilerClassifier
from src.data_loader import create_data_loader


detokenizer = TreebankWordDetokenizer()

with open('app/electra_ep5.json')as f:
    config = json.load(f)

args = SimpleNamespace()
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.model_name = config['model_name']
args.final_dropout = config['final_dropout']
args.max_len = config['max_len']
args.batch_size = config['batch_size']
args.transformer_weights = config['transformer_weights'] if 'transformer_weights' in config else None
args.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
args.in_model = config['in_model']
args.arch = config['arch']
args.threshold = config['threshold'] if 'threshold' in config else 0.5
args.vocab_size = args.tokenizer.vocab_size + 1

args.model = None

if args.arch == 'transformer':
    args.model = SpoilerClassifier(args)
elif args.arch  == 'LSTM':
    args.embedding_dim = 32
    args.model = LSTMSpoilerClassifier(args)

if args.in_model:
    args.model.load_state_dict(torch.load(args.in_model))
args.model.to(args.device)

class_names = ['no_spoilers', 'has_spoilers']

def prepare_df(soup, pattern):
    rev_sents = []
    spoils = []
    rev_ids = []

    for rev in soup.find_all(id=re.compile("^review_\d+$")):
        short_rev = rev.find_all(id=re.compile(pattern))
        if len(short_rev) == 0:
            continue
        short_rev = short_rev[0]
        tok_sents = tokenize.sent_tokenize(short_rev.get_text())
        rev_sents += tok_sents
        spoils += [0] * len(tok_sents)
        rev_ids += [rev.get('id')] * len(tok_sents)

    return pd.DataFrame(list(zip(rev_sents, rev_ids, spoils)), 
                        columns =['sentence', 'rev_id', 'has_spoiler'])


def get_predictions(args, dl):
    predictions = []
    prediction_probs = []

    with torch.no_grad():
        for d in dl:
            input_ids = d["input_ids"].to(args.device)
            attention_mask = d["attention_mask"].to(args.device)
            
            if args.arch == 'LSTM':
                outputs = args.model(input_ids)
            else:
                attention_mask = d["attention_mask"].to(args.device)
                outputs = args.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            preds = torch.zeros_like(outputs)
            ones = torch.ones_like(preds)
            preds = torch.where(torch.sigmoid(outputs) < args.threshold, preds, ones)

            predictions.extend(preds)
            prediction_probs.extend(outputs)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    return predictions, prediction_probs


def mark_spoilers(df):
    print(df.shape)
    data_loader = create_data_loader(df, args)        
    y_pred, y_pred_probs = get_predictions(args, data_loader)
    print(f'found {y_pred.sum()} spoilers!!!')
    df.has_spoiler = y_pred
    df.sentence = df.apply(lambda x: '<!!!!!!!!!SPOILER!!!!!!!!!!!!>' + x.sentence + '</!!!!!!!!!SPOILER!!!!!!!!!!!!>' if x.has_spoiler == 1 else x.sentence, axis=1)

    return df


def update_html(soup, df, pattern):
    for rev in soup.find_all(id=re.compile("^review_\d+$")):
        rev_id = rev.get('id')
        rev = rev.find_all(id=re.compile(pattern))
        if len(rev) == 0:
            continue
        rev = rev[0]
        rev.string = detokenizer.detokenize(list(df[df.rev_id == rev_id].sentence))


def hide_spoilers_pattern(soup, pattern):
    df = prepare_df(soup, pattern)
    mark_spoilers(df)
    print(df[df.has_spoiler==1].sentence)
    update_html(soup, df, pattern)


app = Flask('spoiler-alert')
CORS(app)


@app.route('/spoilerAlert', methods=['POST'])
def hide_spoilers():
    html_doc = request.get_data(as_text=True)

    soup = BeautifulSoup(html_doc, 'html.parser')

    hide_spoilers_pattern(soup, "freeTextContainer\d+")
    hide_spoilers_pattern(soup, "freeText\d+")

    return str(soup)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
