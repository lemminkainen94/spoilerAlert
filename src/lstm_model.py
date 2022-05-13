import torch
import torch.nn as nn


class LSTMSpoilerClassifier(nn.Module):

    def __init__(self, args):
        super(LSTMSpoilerClassifier, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.lstm_layers = args.lstm_layers
        self.device = args.device

        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.dropout1 = nn.Dropout2d(0.25)

        self.lstm = nn.LSTM(args.embedding_dim, args.hidden_dim, dropout=0.1, batch_first=True, num_layers=args.lstm_layers)
        self.dropout2 = nn.Dropout(args.final_dropout)

        self.linear = nn.Linear(args.hidden_dim, 1)

    def forward(self, sentence):
        h = torch.zeros((self.lstm_layers, sentence.size(0), self.hidden_dim)).to(self.device)
        c = torch.zeros((self.lstm_layers, sentence.size(0), self.hidden_dim)).to(self.device)

        nn.init.xavier_normal_(h)
        nn.init.xavier_normal_(c)

        embeds = self.embedding(sentence)
        embeds = embeds.unsqueeze(2)
        embeds = embeds.permute(0, 3, 2, 1)
        embeds = self.dropout1(embeds)
        embeds = embeds.permute(0, 3, 2, 1)
        embeds = embeds.squeeze(2)
        
        out, _ = self.lstm(embeds, (h,c))
        out = self.dropout2(out)

        return self.linear(out[:,-1,:])


if __name__ == '__main__':
    pass