# -*- coding: utf-8 -*-
# @Time    : 7/7/18 16:06
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from taskbot.base import MetaTransformer, MetaSerializable, MetaTrainable

from gensim.corpora.dictionary import Dictionary as GensimDictionary
from gensim import utils


__all__ = ["Dictionary"]


class Dictionary(GensimDictionary, MetaTransformer, MetaSerializable, MetaTrainable):

    def transform(self, documents, unknown_word_index=-1):
        rst = []
        for document in documents:
            rst.append(self.doc2idx(document=document, unknown_word_index=unknown_word_index))
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
        return "{} contains {} words".format(self.__class__.__name__, len(self.token2id))

    def __repr__(self):
        return self.__str__()


# if __name__ == "__main__":
#     path = "/Users/zhouzhirui/Desktop/test"
#     d = Dictionary()
#     s = [["1", "2"]]
#     d.add_documents(s)
#     d.transform(s)
#     d.reverse(d.transform(s))
#     d.save(path)
#     restore = Dictionary.load(path)