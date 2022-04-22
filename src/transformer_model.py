import transformers
import torch
from transformers import BertModel


class SpoilerClassifier(nn.Module):

  	def __init__(self, n_classes, model_name, dropout=0.3):
    	super(SpoilerClassifier, self).__init__()
    	self.bert = BertModel.from_pretrained(model_name)
    	self.drop = nn.Dropout(p=dropout)
    	self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

	def forward(self, input_ids, attention_mask):
	    _, pooled_output = self.bert(
	      input_ids=input_ids,
	      attention_mask=attention_mask
	    ).to_tuple()
	    output = self.drop(pooled_output)
	    return self.out(output)
