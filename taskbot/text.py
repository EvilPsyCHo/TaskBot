# -*- coding: utf-8 -*-
# @Time    : 9/4/18 14:20
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from sklearn.feature_extraction.text import CountVectorizer
corpus = ["好呀 123", "嗯那"]
cv = CountVectorizer(ngram_range=[1,2]).fit(corpus)



class Dictionary:
    pass