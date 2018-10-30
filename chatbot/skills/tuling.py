# coding:utf8
# @Time    : 18-6-8 上午10:04
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
# coding:utf8
# @Time    : 18-6-2 下午6:14
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import requests
import random

from wxpy.utils.misc import enhance_connection

from chatbot.config.constant import TULING_KEY, TULING_LOC, TULING_URL, TULING_USERID
from chatbot.core.skill import BaseSkill


class Tuling(BaseSkill):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = requests.session()
        enhance_connection(self.session)

    def __call__(self, context):
        """

        :param context: <class Context>
        :return: resp: <class Response>
        """
        try:
            send = dict(
                key=TULING_KEY,
                userid=TULING_USERID,
                info=context["query"],
                loc=TULING_LOC,
            )
            r = self.sesstion.post(TULING_URL, send)
            answer = r.json()
            answer_text = answer["text"]
        except:
            answer_text = random.choice([
                "人家累了要睡觉拉，不许调戏人家",
                "请投币后再调戏-.-",
                "再见，{}".format(context["user"])
                ])
        return answer_text

    def init_slots(self):
        return dict()

    def contain_slots(self, entities):
        return False


if __name__ == "__main__":
    tuling = Tuling()
    context = {"query": "哈哈", "user": "周"}
    tuling(context)
