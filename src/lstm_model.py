import torch
import torch.nn as nn


class LSTMSpoilerClassifier(nn.Module):

    def __init__(self, args):
        super(LSTMSpoilerClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.dropout1 = nn.Dropout2d(0.25)

        self.lstm = nn.LSTM(args.embedding_dim, args.hidden_dim, dropout=0.1, batch_first=True, num_layers=2)
        self.dropout2 = nn.Dropout(args.final_dropout)

        self.linear = nn.Linear(args.hidden_dim, 1)

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        embeds = embeds.unsqueeze(2)
        embeds = embeds.permute(0, 3, 2, 1)
        embeds = self.dropout1(embeds)
        embeds = embeds.permute(0, 3, 2, 1)
        embeds = embeds.squeeze(2)
        
        out, _ = self.lstm(embeds)
        out = self.dropout2(out)

        return self.linear(out[:,-1,:])


if __name__ == '__main__':
    pass