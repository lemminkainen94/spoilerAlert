from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
from nltk import tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from flask import Flask, request


detokenizer = TreebankWordDetokenizer()

def prepare_df(soup, pattern):
    #sprint(soup[:100])
    rev_sents = []
    spoils = []
    rev_ids = []

    for rev in soup.find_all(id=re.compile("^review_\d+$")):
        short_rev = rev.find_all(id=re.compile(pattern))
        if len(short_rev) == 0:
            continue
        short_rev = short_rev[0]
        tok_sents = []
        if short_rev.string:
            tok_sents = tokenize.sent_tokenize(short_rev.string)
        if tok_sents == []:
            tok_sents = tokenize.sent_tokenize(str(short_rev))
        rev_sents += tok_sents
        spoils += [0] * len(tok_sents)
        rev_ids += [rev.get('id')] * len(tok_sents)

    return pd.DataFrame(list(zip(rev_sents, rev_ids, spoils)), 
                        columns =['sentence', 'rev_id', 'has_spoiler'])


def mark_spoilers(df):
    #data_loader = create_data_loader(df, args)        
    #_, y_pred, y_pred_probs, _ = get_predictions(args.model, data_loader)
    y_pred = [1] * len(df)
    df.has_spoiler = y_pred
    df.sentence = df.apply(lambda x: '<spoiler>' + x.sentence + '</spoiler>' if x.has_spoiler == 1 else x.sentence, axis=1)

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
    update_html(soup, df, pattern)


app = Flask('spoiler-alert')


@app.route('/spoilerAlert', methods=['POST'])
def hide_spoilers():
    html_doc = request.get_data(as_text=True)

    soup = BeautifulSoup(html_doc, 'html.parser')

    hide_spoilers_pattern(soup, "freeTextContainer\d+")
    hide_spoilers_pattern(soup, "freeText\d+")

    return str(soup)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)



"""for rev in soup.find_all(id=re.compile("^review_\d+$")):
    short_rev = rev.find_all(id=re.compile("freeTextContainer\d+"))
    if len(short_rev) == 0:
        continue
    short_rev = short_rev[0]
    short_rev.string = ''

    
    long_rev = rev.find_all(id=re.compile("freeText\d+"))
    if len(long_rev) == 0:
        continue
    long_rev = long_rev[0]
    long_rev.string = model_output


for rev in soup.find_all(id=re.compile("^review_\d+$")):
    short_rev = rev.find_all(id=re.compile("freeTextContainer\d+"))
    if len(short_rev) == 0:
        continue
    short_rev = short_rev[0]
    print(short_rev)
    
    long_rev = rev.find_all(id=re.compile("freeText\d+"))
    if len(long_rev) == 0:
        continue
    long_rev = long_rev[0]
    print(long_rev)
"""