import pandas as pd
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors, pad_tensor

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
























"""
def get_transformer_item(self, review, target):
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

    def get_lstm_item(self, review, target):
        encoding = pad_tensor(self.tokenizer.encode(review), length=self.max_len)
        return {
          'review_text': review,
          'input_ids': encoding.flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
        }
"""