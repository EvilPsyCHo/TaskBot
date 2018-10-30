# coding:utf8
# @Time    : 18-7-4 下午5:03
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import pandas as pd

from chatbot.utils.path import ROOT_PATH
from chatbot.utils.edit_distance import FuzzyMatch
from chatbot.core.entity import Company


class CompanyNer(object):
    def __init__(self):
        df = pd.read_excel(
            str(ROOT_PATH/"config"/"项目编号对照表.xlsx")
        )
        self.alias2id = dict(df[["alias", "id"]].values)
        keywords = df.alias.tolist()
        threshold = [1 if len(w) >= 3 else 0 for w in keywords]
        self.fuzzy_match = FuzzyMatch(keywords, threshold)

    def extract(self, context):
        entities = self.fuzzy_match.match(context["query"])
        if len(entities) == 0:
            return None
        else:
            return entities

    def transform(self, context):
        entities = context["entities"]
        rst = []
        if Company.name() in entities.keys():
            for alias in entities[Company.name()]:
                c = Company()
                c["alias"] = alias
                c["id"] = self.alias2id[alias]
                rst.append(c)
            if len(rst) == 0:
                return None
            else:
                return rst[0]
        else:
            return None
