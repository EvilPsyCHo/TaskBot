# coding:utf8
# @Time    : 18-6-25 下午2:36
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from chatbot.core.skill import BaseSkill
import random


_response = ["人家是大力水手小益", "人家是萌萌小益拉，刚刚2个月大"]


class BotQA(BaseSkill):
    def __call__(self, context):
        return random.choice(_response)

    def contain_slots(self, entities):
        return False

    def init_slots(self):
        return {}
