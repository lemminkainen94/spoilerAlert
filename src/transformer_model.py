import transformers
import torch
import torch.nn as nn
from transformers import AutoModel


class SpoilerClassifier(nn.Module):

    def __init__(self, args):
        super(SpoilerClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained(args.model_name, return_dict=False)
        self.drop = nn.Dropout(p=args.final_dropout)
        self.out = nn.Linear(self.transformer.config.hidden_size, 1) 

    def forward(self, input_ids, attention_mask):
        pooled_output = self.transformer(
          input_ids=input_ids,
          attention_mask=attention_mask,
        )[0][:, 0, :]
        output = self.drop(pooled_output)
        return self.out(output)


if __name__ == '__main__':
    pass