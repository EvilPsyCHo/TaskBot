# coding:utf8
# @Time    : 18-7-9 上午10:29
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import pysparnn.cluster_index as ci
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals.joblib import load, dump

from chatbot.utils.log import get_logger


logger = get_logger("TfidfQA")


class TfidfQA(object):
    '''
    词频-逆文本频率
    '''
    def __init__(self, path=None):
        '''

        :param path: 文件路径
        '''
        if path is None:
            self._tfidf = TfidfVectorizer()
            self._ci = None
        else:
            self._tfidf = load(path+".tfidf")
            self._ci = load(path+".cluster_index")

    def fit(self, queries, queries_idx):
        """

        Args:
            queries:  list of string, string is a cut query
            queries_idx:

        Returns:
            self
        """
        self._tfidf.fit(queries)
        self._ci = ci.MultiClusterIndex(self._tfidf.transform(queries), queries_idx)
        return self

    def infer(self, query_cut, top_n=5):
        ''' 获取搜索排名前n结果

        :param query_cut:
        :param top_n: 要选取的top个数
        :return: search_result
        '''
        q_tfidf = self._tfidf.transform(query_cut)
        # q_tfidf = self._tfidf.transform([query_cut]).toarray()
        search_result = self._ci.search(q_tfidf, top_n)[0]
        return search_result

    def save(self, path):
        ''' 保存结果到文件

        :param path:
        :return:
        '''
        dump(self._tfidf, path+".tfidf")
        dump(self._ci, path+".cluster_index")
        logger.info("Successful saving TfidfQA in {}".format(path))

    @classmethod
    def load(cls, path):
        ''' 加载文件

        :param path: 加载文件的路径
        :return:
        '''
        logger.info("Loading TfidfQA in {}".format(path))
        return TfidfQA(path)


if __name__ == "__main__":
    import pandas as pd
    from chatbot.preprocessing import text
    from chatbot.utils.path import ROOT_PATH
    file_path = str(ROOT_PATH.parent / "corpus" / "policy_filev3.utf8.csv")
    df = pd.read_csv(file_path)
    df["content_cut"] = df["abstract"].apply(text.cut).apply(lambda x: " ".join(x))

    qa = TfidfQA()
    qa.fit(df.content_cut, [int(i) for i in df.index.tolist()])
    # qa.fit(df.content_cut, df["content"])
    print(qa.infer("四川 电价"))
    qa.save(str(ROOT_PATH.parent/ "corpus"/"test"))
    qa = TfidfQA.load(str(ROOT_PATH.parent/ "corpus"/"test"))