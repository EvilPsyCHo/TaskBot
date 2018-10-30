# coding:utf8
# @Time    : 18-9-14 下午5:12
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import numpy as np

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.3):
        """initialize encoder

        Args:
            input_size: <int>, encoder vocab size
            embed_size: <int>, encoder embed size
            hidden_size: <int>, GRU hidden state size
            n_layers: <int>, GRU layers
            dropout: <float>, dropout rate

        Notes:
            default batch_first, bidirectional=True
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = self.dropout
        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, bidirectional=True, dropout=dropout)

    def forward(self, input_seqs, input_lens, init_hidden=None):
        """

        Args:
            input_seqs: <numpy.ndArray>, shape(B,T), padded sentences
            input_lens: <numpy.ndArray>, shape(B), sentences length
            init_hidden: <torch.FloatTensor>

        Returns:
            outputs: <torch.FloatTensor>, shape(T,B,H)
            hidden: <torch.FloatTensor>, shape(NL*2,B,H)
        """
        batch_size = input_seqs.shape[1]
        # sort inputs
        input_seqs, input_lens = self._sort_inputs(input_seqs, input_lens)
        input_seqs = torch.LongTensor(input_seqs)
        input_lens = torch.LongTensor(input_lens)
        # embedding
        embedded = self.embedding(input_seqs)  # (T,B,E)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lens)
        # encoder
        if init_hidden is None:
            init_hidden = self._init_hidden(batch_size)  # (NL*2,B,H)
        outputs, hidden = self.gru(packed, init_hidden)  # outputs (T,B,2*H) hidden (NL*2,B,H)
        outputs, outputs_lens = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # outputs (T,B,H)
        return outputs, hidden

    @staticmethod
    def _sort_inputs(input_seqs, input_lens):
        """sort input_seqs decreasingly by input_lens

        Args:
            input_seqs: <numpy.ndArray>, shape(T,B)
            input_lens: <numpy.ndArray>, shape(B)

        Returns:
            sorted_input_seqs: <numpy.ndArray>, shape(T,B)
            sorted_input_lens: <numpy.ndArray>, shape(B)
        """
        sort_idx = np.argsort(-input_lens)
        input_seqs = input_seqs.transpose(1, 0)[sort_idx].transpose(1, 0)
        input_lens = input_lens[sort_idx]
        return input_seqs, input_lens

    def _init_hidden(self, batch_size):
        return torch.zeros(self.n_layers * 2, batch_size, self.hidden_size)