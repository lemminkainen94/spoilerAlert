import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMSpoilerClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMSpoilerClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout1 = nn.Dropout2d(0.25)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=0.1, batch_first=True, num_layers=2)
        self.dropout2 = nn.Dropout(0.4)

        self.linear = nn.Linear(hidden_dim, 1)

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
