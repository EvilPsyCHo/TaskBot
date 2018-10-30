# coding:utf8
# @Time    : 18-5-21 上午11:46
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from chatbot.config.constant import *
from chatbot.utils.log import get_logger
from chatbot.cparse.dictionary import Dictionary
logger = get_logger(__name__)


class Vocabulary(Dictionary):
    def __init__(self, init_vocablary=[PAD, UNK]):
        super(Vocabulary, self).__init__()
        for i in init_vocablary:
            self._add_one(i)

    def fit_word(self, x):
        assert isinstance(x, str), ValueError
        self._add_one(x)

    def fit_sentence(self, x):
        assert isinstance(x, list)
        assert isinstance(x[0], str)
        for w in x:
            self.fit_word(w)

    def fit_sentences(self, x):
        for s in x:
            self.fit_sentence(s)

    def fit(self, x):
        """

        :param x: string, list of string, list of list of string
        :return: id or list of id, or list of list of id
        """
        if self.training:
            if isinstance(x, str):
                self.fit_word(x)
            elif isinstance(x, list):
                if isinstance(x[0], str):
                    self.fit_sentence(x)
                elif isinstance(x[0], list):
                    self.fit_sentences(x)
                else:
                    raise ValueError
            else:
                raise ValueError("input error")
        else:
            logger.info("{} can't training now".format(self.__class__.__name__))

    def _transform_input_error(self):
        raise ValueError("required list of string or list of list of string")

    def transform(self, x, max_length=None):
        """文本转换成id

        :param x: list of list of string, a string means a word
        :param max_length: 固定返回的每个句子固定长度
        :return: <list of list> 最小元素为单词的idx
        """
        assert isinstance(x, list), ValueError
        assert isinstance(x[0], list), ValueError
        assert isinstance(x[0][0], str), ValueError
        rst = []
        for sentence in x:
            rst.append(self.transform_sentence(sentence, max_length))
        return rst

    def transform_sentence(self, x, max_length=None):
        """

        :param x: list of string
        :param max_length:
        :return:
        """
        rst_s = [self.transform_word(word) for word in x]
        if max_length is not None:
            rst_s = rst_s[:max_length]
            rst_s = [PAD_IDX] * (max_length - len(rst_s)) + rst_s
        return rst_s

    def transform_word(self, x):
        return self.word2idx.get(x, UNK_IDX)

    def reverse_sentence(self, x, join=False, join_chat=""):
        """

        :param x: list of int, int is word idx
        :param join: 反回的文本是否join成字符串
        :param join_chat: 返回的文本join的字符
        :return:
        """
        rst = []
        for idx in x:
            rst.append(self.reverse_word(idx))
        if join:
            return join_chat.join(rst)
        else:
            return rst

    def reverse_word(self, x):
        return self.idx2word.get(x, UNK)

    def reverse(self, x, join=False, join_chat=""):
        """

        :param x: list of list of int, int is word idx
        :param join: 反回的文本是否join成字符串
        :param join_chat: 返回的文本join的字符
        :return:
        """
        rst = []
        for sentence in x:
            rst.append(self.reverse_sentence(sentence, join, join_chat))
        return rst


if __name__ == "__main__":
    sentences1 = [["我", "我 喜欢 你"]]
    sentences2 = [["他"]]
    vocab = Vocabulary()
    vocab.fit(sentences1)
    print(vocab.transform_sentence(sentences1[0], max_length=10))
    print(vocab.transform_word(sentences2[0][0]))
    print(vocab.reverse_word(15454))
    vocab.training = False
    vocab.fit(sentences2)
    print(vocab.transform(sentences2, max_length=10))
    print(vocab.reverse(vocab.transform(sentences2), False))
