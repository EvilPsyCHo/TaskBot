# -*- coding: utf-8 -*-
# @Time    : 5/27/18 20:48
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import uniform


class SelfAttention(nn.Module):
    def __init__(self, attention_size, non_linear):
        super().__init__()
        if non_linear == "relu":
            self.non_linear = nn.ReLU()
        else:
            self.non_linear = nn.Tanh()
        self.fc = nn.Linear(attention_size, attention_size)
        uniform(self.fc.weight.data, -0.005, 0.005)

    def forward(self, x, lengths=None):
        # x(batch, sentence, embed_dim)
        score = self.non_linear(self.fc(x))

        score = F.softmax(score, dim=-1)
        weights = torch.mul(x, score.unsqueeze(-1).expand_as(x))


class CNNAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        vocab_size = args["vocab_size"]
        class_num = args["class_num"]
        kernel_num = args["kernel_num"]
        kernel_size = args["kernel_size"]
        embed_dim = args["embed_dim"]
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv_11 = nn.Conv2d(1, kernel_num,
                                 (kernel_size[0], embed_dim), padding=((kernel_size[0] - 1) / 2, 0))
        self.conv_12 = nn.Conv2d(1, kernel_num,
                                 (kernel_size[1], embed_dim), padding=((kernel_size[1] - 1) / 2, 0))
        self.conv_13 = nn.Conv2d(1, kernel_num,
                                 (kernel_size[1], embed_dim), padding=((kernel_size[2] - 1) / 2, 0))
        self.att_1 = nn.Linear()

    def init_parameter(self):
        pass
