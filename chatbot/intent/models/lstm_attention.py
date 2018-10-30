# -*- coding: utf-8 -*-
# @Time    : 5/27/18 13:51
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import torch
from torch import nn
import torch.nn.functional as F


# 实验结果不理想
class LSTMAttention(nn.Module):
    def __init__(self, args):
        self.args = args
        super(LSTMAttention, self).__init__()
        self.hidden_dim = args["rnn_dim"]

        self.word_embeddings = nn.Embedding(args["vocab_size"], args["embed_dim"])
        # self.bidirectional = True
        self.dropout = nn.Dropout(0.2)
        self.bilstm = nn.LSTM(args["embed_dim"], self.hidden_dim, batch_first=True, num_layers=1, bidirectional=False)
        self.hidden2label = nn.Linear(self.hidden_dim, args["class_num"])

    def init_hidden(self, batch_size):
        h0 = torch.zeros(1, batch_size, self.hidden_dim)
        c0 = torch.zeros(1, batch_size, self.hidden_dim)
        return h0, c0

    def attention(self, rnn_out, state):
        merged_state = torch.cat([s for s in state], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        # (batch, seq_len, cell_size) * (batch, cell_size, 1) = (batch, seq_len, 1)
        weights = torch.bmm(rnn_out, merged_state)
        weights = F.softmax(weights.squeeze(2), 1).unsqueeze(2)
        # (batch, cell_size, seq_len) * (batch, seq_len, 1) = (batch, cell_size, 1)
        return torch.bmm(torch.transpose(rnn_out, 1, 2), weights).squeeze(2)

    def forward(self, X):
        embedded = self.word_embeddings(X)
        hidden = self.init_hidden(X.size()[0])
        rnn_out, hidden = self.bilstm(embedded, hidden)
        h_n, c_n = hidden
        attn_out = self.attention(rnn_out, h_n)
        logits = self.hidden2label(attn_out)
        return logits

    def get_attention(self, x):
        embedded = self.word_embeddings(x)
        hidden = self.init_hidden(x.size()[0])
        rnn_out, hidden = self.bilstm(embedded, hidden)
        merged_state = torch.cat([s for s in hidden[0]], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        # (batch, seq_len, cell_size) * (batch, cell_size, 1) = (batch, seq_len, 1)
        weights = torch.bmm(rnn_out, merged_state)
        weights = F.softmax(weights.squeeze(2), 1).unsqueeze(2)
        return weights


if __name__ == "__main__":
    param = {
        "vocab_size": 1000,
        "embed_dim": 60,
        "rnn_dim":30,
        "dropout":0.5,
        "class_num":10,
    }
    rnn = LSTMAttention(param)
    test_input = torch.arange(0, 200).view(10, 20).long()
    out = rnn(test_input)