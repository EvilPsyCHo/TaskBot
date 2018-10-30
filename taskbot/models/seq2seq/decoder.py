# coding:utf8
# @Time    : 18-9-14 下午5:12
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import torch
from torch import nn
import torch.nn.functional as F

from .attention import Attn


class AttnDecoder(nn.Module):
    """seq2seq decoder with attention mechanism"""
    def __init__(self, hidden_size, embed_size, output_size,
                 n_layers=1, dropout=0.1):
        """

        Args:
            hidden_size: GRU hidden_size
            embed_size:  embedding size
            output_size:  outputs vocab size
            n_layers:  GRU layers
            dropout:  dropout ratio,
        """
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