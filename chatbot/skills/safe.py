# coding:utf8
# @Time    : 18-6-25 下午2:36
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from chatbot.core.skill import BaseSkill


class SafeResponse(BaseSkill):
    def __call__(self, context):
        return "这个小益还没学会哦，请试试政策文件检索和电量数据查询功能吧 ：）"

    def contain_slots(self, entities):
        return False

    def init_slots(self):
        return {}
