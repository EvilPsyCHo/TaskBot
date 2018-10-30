<<<<<<< HEAD
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

=======
# coding: utf-8
# Author: Kevin Zhou
# Mail  : evilpsycho42@gmail.com
# Time  : 9/30/18


from taskbot.models.seq2seq import DynamicEncoder
from taskbot.trash.vocabulary import BasicVocabulary


encoder_vocab_size = 10000
encoder_embed_size = 60
encoder_state_size = 30
encoder_rnn_layers = 4
encoder = DynamicEncoder(
    encoder_vocab_size, encoder_embed_size, encoder_state_size, encoder_rnn_layers
)

docs = [
    'haha 昨天    是不是 , ！ 没来 哦',
    '那 有过 呢'
]


encoder_dictionary = BasicVocabulary()
encoder_dictionary.fit(docs)
encoder_dictionary.seq2doc(encoder_dictionary.doc2seq(docs))
encoder_dictionary.doc2seq(docs)


class TestSeq2Seq:
    def test_encoder(self):
        self.encoder = encoder
>>>>>>> 029e56027ae8cff8cb6253de70649bcbca944284
