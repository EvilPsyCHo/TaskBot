# coding:utf8
# @Time    : 18-6-6 上午10:02
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import sys
sys.path.append("/home/zhouzr/project/Task-Oriented-Chatbot")

from sklearn.externals.joblib import load
import pandas as pd

from chatbot.core.skill import BaseSkill
from chatbot.core.entity import TimeInterval, Location
from chatbot.QA import TfidfQA


class FileRetrievalExt(BaseSkill):
    '''
    文件检索系统类
    功能：对用户文件检索的回复，可选择最匹配的文件数量
    '''
    def __init__(self, QAmodel_path, df_path, k=5):
        '''

        :param QAmodel_path: <string> 模型保存的路径
        :param df_path: <string> 保存了所有政策文件的文件路径
        :param k: <string> 选取的匹配个数
        '''
        super().__init__()
        self.QAmodel = TfidfQA.load(QAmodel_path)
        self.df = self._file = pd.read_csv(df_path, parse_dates=[1])
        self.k = k

    def __call__(self, context):
        """ 如果满足调用条件，返回回复；否则，返回对应的问题

        :param context:<dict> 当前对话的上下文语境
        :return: context非空时，返回self._act(context)；否则，返回None
        """
        is_satisfied, question = self._check_satisfied(context)
        if is_satisfied:
            return self._act(context)
        else:
            return question

    def init_slots(self):
        '''

        :return: <string> 初始化检索的地区和时间区间
        '''
        return {
            Location.name(): Location(),
            TimeInterval.name(): TimeInterval()
        }

    def contain_slots(self, entities):
        """ 判断是否包含skill所需词槽，如果包含，返回True；否则，返回False

        :param entities: <dict of list>, key: entity name, values: list of entity
        :return: <bool>
        """
        for k, v in entities.items():
            if k in self.init_slots().keys():
                return True
        return False

    def _check_satisfied(self, context):
        return True, None

    def _act(self, context):
        search_result = self.QAmodel.infer(context["query_cut"], top_n=self.k*200)
        slot = context["slots"][context["intent"]]
        if len(search_result) == 0:
            return self._not_find
        result = []
        scores = []
        idx_tfidf_scores = []
        for score, idx in search_result:
            scores.append(score)
            idx_tfidf_scores.append(idx)
        #时间区间限制
        idx_time_limit = self._file[(self._file.publish_time >= slot["TimeInterval"]["start"]) & \
                               (self._file.publish_time <= slot["TimeInterval"]["end"])].index.tolist()
        #地区限制
        idx_location_limit = self._file.index.tolist() if slot["Location"]["province"] is None \
            else self._file[(self._file.area==slot["Location"]["province"])].index.tolist()
        #
        idx_limit = set(idx_location_limit) & set(idx_time_limit)
        #生成k个文件检索结果
        count = 0
        for (d, i) in search_result:
            if i in idx_limit:
                result.append(self._response_template(province=self._file.loc[i, "area"],
                                                      time=self._file.loc[i, "publish_time"].date(),
                                                      name=self._file.loc[i, "title"],
                                                      url=self._file.loc[i, "url"],
                                                      distance=d
                                                      ))
                count += 1
                if count >= self.k:
                    break
        if len(result) == 0:
            return self._not_find
        #返回最终查询结果
        else:
            head = self._response_head(slot["Location"]["province"],
                                       startdate=slot["TimeInterval"]["start"],
                                       enddate=slot["TimeInterval"]["end"]
                                       )
            return head + "\n".join(result)

    @property
    def _not_find(self):
        return "抱歉，您所查找的政策文件不存在，小益已经上报，可能明天就有了哦～"

    @staticmethod
    def _response_template(province, time, name, url, distance):
        return "{} {}：{}，{}，相似度:{:.1f}%".format(province, time, name,
                                             url, (1-distance)*100)

    @staticmethod
    def _response_head(location=None, startdate=None, enddate=None):
        '''

        :param location: <string> 地区
        :param startdate: <string> 起始日期
        :param enddate: <string> 结束日期
        :return: <string> 回复语句头部
        '''
        l = "不限" if location is None else location
        s = "不限" if startdate is None else startdate
        e = "不限" if enddate is None else enddate
        return "您查询的地区“{}”在{}至{}相关文件如下: \n".format(l, s, e)

class FileRetrieval(BaseSkill):
    def __init__(self, tfidf_path, cluster_index_path, file_path, k=5, limit_distance=0.99):
        super().__init__()
        self._tfidf = load(tfidf_path)
        self._ci = load(cluster_index_path)
        self._file = pd.read_csv(file_path, parse_dates=[0])
        self._k = k
        self._limit_distance = limit_distance

    def init_slots(self):
        return {
            Location.name(): Location(),
            TimeInterval.name(): TimeInterval()
        }

    def __call__(self, context):
        """ 如果满足调用条件，返回回复，否则，返回对应的问题

        :param context:
        :return:
        """
        is_satisfied, question = self._check_satisfied(context)
        if is_satisfied:
            return self._act(context)
        else:
            return question

    def contain_slots(self, entities):
        """

        :param entities: <dict of list>, key: entity name, values: list of entity
        :return:
        """
        for k, v in entities.items():
            if k in self.init_slots().keys():
                return True
        return False

    def _check_satisfied(self, context):
        return True, None

    def _act(self, context):
        q_tfidf = self._tfidf.transform([" ".join(context["query"])]).toarray()
        search_result = self._ci.search(q_tfidf, k=self._k)[0]
        slot = context["slots"][context["intent"]]
        if len(search_result) == 0:
            return self._not_find
        result = []
        idx_time_limit = self._file[(self._file.publish_time >= slot["TimeInterval"]["start"]) & \
                               (self._file.publish_time <= slot["TimeInterval"]["end"])].index.tolist()
        idx_location_limit = self._file.index.tolist() if slot["Location"]["province"] is None \
            else self._file[(self._file.area==slot["Location"]["province"])].index.tolist()
        idx_limit = set(idx_location_limit) & set(idx_time_limit)
        for (d, i) in search_result:
            if (d <= self._limit_distance) and (i in idx_limit):
                result.append(self._response_template(province=self._file.loc[i, "area"],
                                                      time=self._file.loc[i, "publish_time"].date(),
                                                      name=self._file.loc[i, "title"],
                                                      url=self._file.loc[i, "url"],
                                                      distance=d
                                                      ))
        if len(result) == 0:
            return self._not_find
        else:
            head = self._response_head(slot["Location"]["province"],
                                       startdate=slot["TimeInterval"]["start"],
                                       enddate=slot["TimeInterval"]["end"]
                                       )
            return head + "\n".join(result)

    @staticmethod
    def _response_template(province, time, name, url, distance):
        return "{} {}：{}，{}，相似度:{:.1f}%".format(province, time, name,
                                             url, (1-distance)*100)

    @staticmethod
    def _response_head(location=None, startdate=None, enddate=None):
        l = "不限" if location is None else location
        s = "不限" if enddate is None else startdate
        e = "不限" if enddate is None else enddate
        return "您查询的地区“{}”在{}至{}相关文件如下: \n".format(l, s, e)

    @property
    def _not_find(self):
        return "抱歉，您所查找的政策文件不存在，小益已经上报，可能明天就有了哦～"


if __name__ == "__main__":
    from chatbot.utils.path import MODEL_PATH, ROOT_PATH
    skill = FileRetrievalExt(str(ROOT_PATH.parent/"corpus"/"test"),
                          str(ROOT_PATH.parent/"corpus"/"policy_filev3.utf8.csv"))

    from chatbot.preprocessing.text import cut

    context = {
        "query_cut": " ".join(cut("月度竞价")),
        "slots":{"文件检索":skill.init_slots()},
        "intent": "文件检索"
               }

    context["slots"]["文件检索"]["TimeInterval"]["end"]="2018-01-01"
    context["slots"]["文件检索"]["TimeInterval"]["start"] = "2018-07-02"
    context["slots"]["文件检索"]["Location"]["province"] = "四川"
    print(skill(context))
