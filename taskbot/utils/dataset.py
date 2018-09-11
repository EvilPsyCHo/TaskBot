# coding:utf8
# @Time    : 18-9-7 上午9:23
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com


def chat2qa(path, mode="QAQAQ"):
    import json
    def _chat2QA(sess, mode):
        waiter_first = 1 if sess["is_waiter"][0]=="1" else 0
        waiter_end = 1 if sess["is_waiter"][-1]=="1" else 0
        if len(sess["content"]) < len(mode) + 1 + waiter_first + (1-waiter_end):
            return None
        pairs = []
        start = 1 if waiter_first else 0
        end = len(sess["content"]) - len(mode) if waiter_end else len(sess["content"]) - len(mode) -1
        for i in range(start, end, 2):
            pairs.append(["<s>".join(sess["content"][i: i+len(mode)]), sess["content"][i+len(mode)]])
        return pairs
    chat = json.load(open(path, 'rb'))
    sess_id = chat.keys()
    q, a = [], []
    for sess in sess_id:
        res_i = _chat2QA(chat[sess], mode)
        if res_i:
            for i in res_i:
                q.append(i[0])
                a.append(i[1])
    return q, a


if __name__ == "__main__":
    from pathlib import Path
    path = str(Path(".").resolve().parent.parent/"externel"/"chat_train.json")
    path = str(Path(".").resolve() / "externel" / "chat_train.json")
    q, a=chat2qa(path)
    print(q[2])
    print(a[2])
    from taskbot.text import Dictionary
    import jieba
    q = [" ".join(jieba.lcut(i)) for i in q[:3]]
    a = [" ".join(jieba.lcut(i)) for i in a[:3]]
    corpus = q + a
    dictionary = Dictionary()
    dictionary.fit(q)
    dictionary.doc2bow(q)
    q_c = dictionary.doc2mat(q)
    from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer

    from gensim.models.tfidfmodel import TfidfModel
    from gensim.corpora.dictionary import Dictionary
    c = CountVectorizer()
    c.fit_transform(q)
    sk_tfidf = TfidfTransformer().fit(dictionary.doc2mat(q))
    sk_tfidf = TfidfTransformer().fit(c.fit_transform(q))
    gs_tfidf = TfidfModel(dictionary.doc2bow(q))