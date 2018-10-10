# coding:utf8
# @Time    : 18-9-5 下午4:10
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import os

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

from taskbot.corpus import BasicDictionary


class TestCorpus:

    def test_BasicDictionary(self):
        docs = ["哈哈 哈哈 我 说过 这个 吗", "哈哈 是不是 哦", "好吧", "哈哈 不错 哦"]
        d = BasicDictionary(ngram_range=[1, 2], max_df=1.0, min_df=1)
        d.fit(docs)
        d.doc2bow(docs)
        d.doc2seq(docs)
        d.seq2doc(d.doc2seq(docs))
        d.save("./test_dictionary")
        d = BasicDictionary.load("./test_dictionary")
        os.remove("./test_dictionary")
        tfidf = TfidfTransformer()
        tfidf.fit(d.doc2mat(docs))
        m = d.doc2mat(docs)
        tfidf.transform(d.doc2mat(docs))