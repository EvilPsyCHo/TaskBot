# coding:utf8
# @Time    : 18-6-8 上午10:04
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from chatbot.core.skill import BaseSkill


class CorpusCollect(BaseSkill):
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        pass

    def init_slots(self):
        return dict()

    def contain_slots(self, entities):
        return False


if __name__ == "__main__":
    act1 = CorpusCollect()
    print(act1)