# coding:utf8
# @Time    : 18-6-25 下午2:36
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from chatbot.core.skill import BaseSkill


class CompanyInfo(BaseSkill):
    def __call__(self, context):
        return "华宇万益是..."

    def contain_slots(self, entities):
        return False

    def init_slots(self):
        return {}


class CompanyServe(BaseSkill):
    def __call__(self, context):
        return "产品服务介绍，我还没背会呢。。"

    def contain_slots(self, entities):
        return False

    def init_slots(self):
        return {}
