# coding:utf8
# @Time    : 18-6-25 下午2:36
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from chatbot.core.skill import BaseSkill


class Help(BaseSkill):
    def __call__(self, context):
        return "人家可以帮你查用电数据(电量|峰平谷电量或电费)，政策文件等等哦"

    def contain_slots(self, entities):
        return False

    def init_slots(self):
        return {}
