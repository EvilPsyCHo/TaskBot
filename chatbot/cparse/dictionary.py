# coding:utf8
# @Time    : 18-5-21 下午2:11
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import pickle

from chatbot.utils.log import get_logger
from chatbot.core.serializable import Serializable
from chatbot.core.trainable import Trainable
from chatbot.core.transformer import Transformer

logger = get_logger(__name__)


class Dictionary(Serializable, Transformer, Trainable):
    def __init__(self):
        super(Dictionary, self).__init__()
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.training = True

    def _add_one(self, word):
        if self.word2idx.get(word) is None:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def size(self):
        return self.idx

    def fit(self, x):
        raise NotImplementedError

    def transform(self, *args, **kwargs):
        raise NotImplementedError

    def reverse(self, *args, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return self.idx

    def __str__(self):
        return "{}(size = {})".format(self.__class__.__name__, self.idx)

    def __repr__(self):
        return self.__str__()

    def save(self, path):
        with open(path, 'wb') as f:
            return pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)


if __name__ == "__main__":
    c = Dictionary()
    c.save("test")
