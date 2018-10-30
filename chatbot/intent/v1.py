# coding:utf8
# @Time    : 18-6-11 下午3:31
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from chatbot.core.estimator import Estimator


class IntentFlow(Estimator):
    def __init__(self, model, rule):
        super().__init__()
        self.model = model
        self.rule = rule

    def infer(self, context):
        rule_rst = self.rule.infer(context["qeury"])
        model_rst = self.model.infer(context["qeury_idx"])
        if rule_rst is not None:
            return rule_rst
        intent = rule_rst[0]
        confidence = rule_rst[1]



    @staticmethod
    def _get_last_intent(context):
        if len(context["history_intent"]) == 0:
            return None
        else:
            return context["history_intent"]

