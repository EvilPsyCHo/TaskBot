# coding:utf8
# @Time    : 18-9-13 上午11:49
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import numpy as np

from taskbot.models.seq2seq import Encoder


vocab_size = 100
embed_size = 60
hidden_size = 30
n_layer = 2
n_batch = 32
max_len = 10


class TestSeq2Seq:

    def test_encoder(self):
        seq = np.random.randint(0, vocab_size, size=(max_len, n_batch))
        seq_lens = np.random.randint(5, max_len+1, size=n_batch)
        self.encoder = Encoder(vocab_size, embed_size, hidden_size, n_layer)
        outputs, hidden = self.encoder(seq, seq_lens)
        print("outputs: ", outputs.size(), "hidden:", hidden.size())

