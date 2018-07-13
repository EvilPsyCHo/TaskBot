# -*- coding: utf-8 -*-
# @Time    : 7/7/18 16:06
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from taskbot.base import MetaTransformer, MetaSerializable, MetaTrainable
from taskbot.config import UNKNOWN, UNKNOWN_IDX, PAD, PAD_IDX

from gensim.corpora.dictionary import Dictionary as GensimDictionary
from gensim import utils


__all__ = ["Dictionary"]


class Dictionary(GensimDictionary, MetaTransformer, MetaSerializable, MetaTrainable):

    def __init__(self, documents=None, prune_at=2000000):
        super().__init__(documents=documents, prune_at=prune_at)
        self.token2id = {UNKNOWN: UNKNOWN_IDX, PAD: PAD_IDX}

    def transform(self, documents, max_len=None, unknown_word_index=UNKNOWN_IDX):
        """

        Args:
            documents: <List of List of String>
            需要转换的文档， e.x. [["我", "很好"]]
            max_len: <Int or None> 需要pad和截断的目标长度
            unknown_word_index: <Int> 未登陆词默认idx

        Returns:
            <List of List of Int>
        """
        rst = []
        for document in documents:
            rst_i = self.doc2idx(document=document, unknown_word_index=unknown_word_index)
            if max_len is None:
                pass
            else:
                if len(rst_i) >= max_len:
                    rst_i = rst_i[:max_len]
                else:
                    rst_i = [PAD_IDX] * (max_len - len(rst_i)) + rst_i
            rst.append(rst_i)
        return rst

    def reverse(self, documents_idx, unknown_word="UNKNOWN"):
        if len(self.id2token) != len(self.token2id):
            self.id2token = utils.revdict(self.token2id)
        rst = []
        for doc_idx in documents_idx:
            rst_i = []
            for idx in doc_idx:
                rst_i.append(self.id2token.get(idx, unknown_word))
            rst.append(rst_i)
        return rst

    def fit(self, documents, prune_at=2000000):
        self.add_documents(documents, prune_at=2000000)
        return self

    def __str__(self):
        print(super(Dictionary))
    #     # return "{} contains {} words".format(self.__class__.__name__, len(self.token2id))
    #
    # def __repr__(self):
    #     return self.__str__()


if __name__ == "__main__":
    path = "/home/zhouzr/project/test"
    d = Dictionary()
    s = [["1", "2"], ["23", "1"]*16]
    d.add_documents(s)
    d.transform(s, max_len=10)
    d.reverse(d.transform(s))
    d.reverse(d.transform(s, max_len=10))
    d.save(path)
    restore = Dictionary.load(path)