# coding:utf8
# @Time    : 18-9-14 下午5:12
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import math

import torch
from torch import nn
import torch.nn.functional as F


class Attn(nn.Module):
    def __init__(self, hidden_size, method="concat"):
        """

        Args:
            hidden_size: <int>, hidden size
                         previous hidden state size of decoder
            method: <str>, {"concat"}
                        Attention method

        Notes:
            we use the GRU outputs instead of using encoder t-step
            hidden sates for attention, because the pytorch-GRU hidden_n only
            contains the last time step information.
        """
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, decoder_hidden, encoder_outputs):
        """

        Args:
            decoder_hidden: <torch.FloatTensor>, shape(B,H)
                    previous hidden state of the last layer in decoder
            encoder_outputs: <torch.FloatTensor>, shape(T,B,H)
                    encoder outputs

        Returns:
            normalized attention weights: <torch.FloatTensor>, shape(B,T)
        """
        max_len = encoder_outputs.size(0)
        H = decoder_hidden.repeat(max_len, 1, 1).transpose(0, 1)  # (B,T,H)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # (B,T,H)
        attn_energies = self.score(H, encoder_outputs)  # (B,T)
        return F.softmax(attn_energies).unsqueeze(1)  # (B,1,T)

    def score(self, hidden, encoder_outputs):
        # hidden & encoder_outputs both shape is (B,T,H)
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))  # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2, 1)  # (B,H,T)
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # (B,1,T)
        return energy.squeeze(1)  # (B,T)
