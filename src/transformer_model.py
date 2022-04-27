import transformers
import torch
import torch.nn as nn
from transformers import BertModel


class SpoilerClassifier(nn.Module):

    def __init__(self, args):
        super(SpoilerClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(args.model_name)
        self.drop = nn.Dropout(p=args.final_dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, 1) 

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        ).to_tuple()
        output = self.drop(pooled_output)
        return self.out(output)


if __name__ == '__main__':
    pass