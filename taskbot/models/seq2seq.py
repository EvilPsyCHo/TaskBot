# coding:utf8
# @Time    : 18-9-5 下午5:13
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import numpy as np
import math

import torch
from torch import nn
import torch.nn.functional as F

CUDA = torch.has_cudnn


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.5):
        """initialize encoder

        Args:
            input_size: <int>, vocab_size
            embed_size: <int>, encoder embed size
            hidden_size: <int>, RNN hidden state size
            n_layers: <int>, RNN layers
            dropout: <float>, dropout rate

        Notes:
            default batch_first
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
        if CUDA:
            input_seqs = input_seqs.cuda()
            input_lens = input_seqs.cuda()
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
            sorted_input_seqs:
            sorted_input_lens:
        """
        sort_idx = np.argsort(-input_lens)
        input_seqs = input_seqs.transpose(1, 0)[sort_idx].transpose(1, 0)
        input_lens = input_lens[sort_idx]
        return input_seqs, input_lens

    def _init_hidden(self, batch_size):
        return torch.zeros(self.n_layers * 2, batch_size, self.hidden_size)


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
            decoder_hidden: <torch.FloatTensor>, shape(B, H)
                    previous last layer hidden state of decoder
            encoder_outputs: <torch.FloatTensor>, shape(T,B,H)
                    encoder outputs

        Returns:
            normalized attention weights: <torch.FloatTensor>, shape(B,1,T)
        """
        max_len = encoder_outputs.size(0)
        H = decoder_hidden.repeat(max_len, 1, 1).transpose(0, 1)  # (B,T,H)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # (B,T,H)
        attn_energies = self.score(H, encoder_outputs)  # (B,T)
        return F.softmax(attn_energies).unsqueeze(1)  # (B,1,T)

    def score(self, hidden, encoder_outputs):
        # hidden & encoder_outputs, (B,T,H)
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))  # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2, 1)  # (B,H,T)
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, embed_size, output_size,
                 n_layers=1, dropout=0.1):
        super(AttnDecoder, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        # Define layers
        self.embedding = nn.Embedding(output_size, embed_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.attn = Attn('concat', hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, word_input, encoder_outputs, last_hidden=None):
        """

        Args:
            word_input:  <torch.LongTensor>, shape(B), word input for current time step
            last_hidden:  <torch.FloatTensor>, shape(NL,B,H), last hidden stat of the decoder
            encoder_outputs: <torch.FloatTensor>, shape(T*B*H),

        Returns:
            outputs: <torch.FloatTensor>
            hidden: <torch.FloatTensor>, shape(NL,B,H)

        Notes:
            ...
        """
        if last_hidden is None:
            last_hidden = self._init_hidden(word_input.shape[0])
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, word_input.size(0), -1)  # (1,B,E)
        word_embedded = self.dropout_layer(word_embedded)  # (1,B,E)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)   # (B,1,T)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,T)*(B,T,H)->(B,1,H)
        context = context.transpose(0, 1)  # (1,B,H)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)  # (1,B,E) cat (1,B,H) -> (1,B,E+H)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,V)->(B,V)
        output = F.log_softmax(self.out(output))
        return output, hidden, attn_weights

    def _init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing=0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.forcing = teacher_forcing
        self.vocab_size = self.decoder.output_size

    def forward(self, src_seqs, src_lengths, trg_seqs):
        """

        Args:
            src_seqs:
            src_lengths:
            trg_seqs:

        Returns:

        """
        batch_size = src_seqs.shape[0]
        max_len = trg_seqs.shape[0]

        outputs = torch.zeros(max_len, batch_size, self.vocab_size)
        encoder_outputs, encoder_hidden = self.encoder(src_seqs, src_lengths)
        # first input to the decoder is the <sos> tokens
        output = trg_seqs[0, :]
        hidden = None
        for t in range(1, max_len):
            output, hidden, attn = self.decoder(output, encoder_outputs, hidden)
            outputs[t] = output
            teacher_force = np.random.random() < self.forcing
            output = (trg_seqs[t] if teacher_force else output.max(1)[1])
        return outputs

    def predict(self, src_seqs, src_lengths, max_trg_len=20, start_idx=1):
        max_src_len = src_seqs.shape[0]
        batch_size = src_seqs.shape[1]
        outputs = torch.zeros(max_trg_len, batch_size, self.vocab_size)
        encoder_outputs, hidden = self.encoder(src_seqs, src_lengths)
        output = torch.LongTensor([start_idx] * batch_size)
        attn_weights = torch.zeros((max_trg_len, batch_size, max_src_len))
        for t in range(1, max_trg_len):
            output, hidden, attn_weight = self.decoder(output, encoder_outputs)
            outputs[t] = output
            output = output.max(1)[1]
            attn_weights[t] = attn_weight
        return outputs, attn_weights
