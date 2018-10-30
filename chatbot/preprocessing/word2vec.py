# -*- coding: utf-8 -*-
# @Time    : 5/14/18 22:07
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


import gensim
import numpy as np

from chatbot.utils.log import get_logger
from chatbot.utils.wrapper import time_counter
from chatbot.config.constant import PAD, PAD_IDX, UNK, UNK_IDX
from chatbot.cparse.vocabulary import Vocabulary

logger = get_logger("Word2vec")


class Word2vecExt(object):
    def __init__(self):
        self.model = None

    def _input2sentences(self, inputs, **kwargs):
        if isinstance(inputs, str):
            path = Path(inputs).resolve()
            if path.is_dir():
                sentences = gensim.models.word2vec.PathLineSentences(inputs, **kwargs)
            else:
                sentences = gensim.models.word2vec.LineSentence(inputs, **kwargs)
        elif isinstance(inputs, list):
            # [["我", "喜欢", "你"], ...]
            sentences = inputs
        else:
            raise TypeError
        return sentences

    @time_counter
    def build(self, texts, **kwargs):
        logger.info("Start build word2vec model")
        sentences = self._input2sentences(texts, **kwargs)
        logger.info("Finish get sentences")
        self.model = gensim.models.Word2Vec(sentences, **kwargs)
        logger.info("Finish build word2vec model")

    @time_counter
    def load(self, path):
        self.model = gensim.models.Word2Vec.load(path)

    @time_counter
    def save(self, path):
        self.model.save(path)
        logger.info("Success save model in %s" % path)

    @time_counter
    def update(self, texts, update_new_words=True, **kwargs):
        sentences = self._input2sentences(texts)
        if not isinstance(self.model, gensim.models.Word2Vec):
            logger.warning("Can't update word2vec model, "
                           "Please build or load model first.")
            return
        if update_new_words:
            self.model.build_vocab(sentences, update=True)
        self.model.train(sentences, **kwargs)

    def transform(self, x):
        if isinstance(x, list):
            if len(x) == 0:
                logger.warning("Input sentence length is 0")
                return None
            rst = []
            for word in x:
                try:
                    rst.append(self.model.wv[word])
                except KeyError:
                    pass
            return np.stack(rst, 0)
        elif isinstance(x, str):
            try:
                return self.model.wv[x]
            except KeyError:
                return None
        else:
            raise ValueError

def get_word2vec_matrix(path):
    word2vecExt_obj = Word2vecExt()
    word2vecExt_obj.load(path)
    wv = word2vecExt_obj.model.wv
    vocab = Vocabulary()
    vocab.fit(wv.index2word)
    matrix = np.concatenate((np.zeros((2, wv.vectors.shape[1])),
                             wv.vectors
                             ),
                            axis=0)
    return vocab, matrix