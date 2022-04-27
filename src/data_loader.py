import pandas as pd
import transformers
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader


BATCH_SIZE = 32
MAX_LEN = 128
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

class ReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len=128):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        target = self.targets[idx]
        encoding = self.tokenizer.encode_plus(
          review,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          pad_to_max_length=True,
          return_attention_mask=True,
          return_tensors='pt',
        )
        return {
          'review_text': review,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
        }

def create_data_loader(df, args):
    ds = ReviewDataset(
        reviews=df.sentence.to_numpy(),
        targets=df.has_spoiler.to_numpy(),
        tokenizer=args.tokenizer,
        max_len=args.max_len
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True
    )


if __name__ == '__main__':
    pass