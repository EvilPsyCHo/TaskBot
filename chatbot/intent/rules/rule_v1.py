# coding:utf8
# @Time    : 18-6-7 上午10:59
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from chatbot.core.estimator import Estimator
from chatbot.cparse.label import IntentLabel


class IntentRuleV1(Estimator):
    def __init__(self):
        super().__init__()

    def infer(self, query):
        """

        :param query: 没有经过转化的原始输入
        :return: <tuple> (class, prob)
        """
        if query.startswith("留言"):
            return "留言", 1.0
        else:
            return None


if __name__ == "__main__":
    rule = IntentRuleV1()
    print(rule.infer("意图 我"))