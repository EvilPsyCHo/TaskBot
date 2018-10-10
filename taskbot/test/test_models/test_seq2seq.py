# coding: utf-8
# Author: Kevin Zhou
# Mail  : evilpsycho42@gmail.com
# Time  : 9/30/18


from taskbot.models.seq2seq import DynamicEncoder
from taskbot.vocabulary import BasicVocabulary


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
