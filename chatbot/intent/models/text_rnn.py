# -*- coding: utf-8 -*-
# @Time    : 5/26/18 13:57
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com

import torch
from torch import nn
import torch.nn.functional as F


class TextRNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        vocab_size = args["vocab_size"]
        embed_dim = args["embed_dim"]
        rnn_dim = args["rnn_dim"]
        dropout = args["dropout"]
        class_num = args['class_num']
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, rnn_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_dim, class_num)

    def forward(self, x):
        # x: (batch, sentences)
        x_embed = self.embed(x)
        print("x_embed shape:", x_embed.size())
        # x_embed: (batch, sentences, embed_dim)
        x_embed = self.dropout(x_embed)
        hidden = self.init_hidden(x_embed.size()[0])
        print("h_0 shape:", hidden[0].size())
        print("c_0 shape:", hidden[1].size())
        rnn_out, _ = self.rnn(x_embed, hidden)
        print("rnn_out size", rnn_out.size(), torch.sum(rnn_out[-1]))
        print("h_n size", _[0].size(), torch.sum(_[0]))
        print("c_n size", _[1].size(), torch.sum(_[1]))
        # rnn_out: (batch, sentences, encoding_dim)
        logit = self.fc(rnn_out[:, -1, :].squeeze())
        return logit

    def init_hidden(self, batch_size):
        h0 = torch.zeros(1, batch_size, self.args["rnn_dim"])
        c0 = torch.zeros(1, batch_size, self.args["rnn_dim"])
        return h0, c0


if __name__ == "__main__":
    param = {
        "vocab_size": 1000,
        "embed_dim": 60,
        "rnn_dim": 30,
        "dropout": 0.5,
        "class_num": 10
    }
    rnn = TextRNN(param)
    test_input = torch.arange(0, 40).view(2, 20).long()
    out = rnn(test_input)

