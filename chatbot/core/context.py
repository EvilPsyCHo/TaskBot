# coding:utf8
# @Time    : 18-6-2 下午4:48
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import datetime as dt

from chatbot.config.constant import TIMEOUT


class Context(dict):
    def __init__(self, user, app, intent2slots, context_id, right=None, timeout=TIMEOUT):
        self._timeout = timeout
        now = dt.datetime.now()
        super().__init__(
            user=user,
            app=app,
            right=right if right else [],
            history_query=[],
            history_resp=[],
            history_intent=[],
            query=None,
            query_cut=None,
            query_idx=None,
            intent=None,
            entities=None,
            slots=intent2slots,
            last_query_time=now,
            context_id=context_id,
            current_slots=None
        )

    @property
    def is_timeout(self):
        """是否超时

        如果超时，上下文将被从DM中删除
        """
        if (dt.datetime.now() - self["last_query_time"]).seconds > self._timeout:
            return True
        else:
            return False


if __name__ == "__main__":
    c = Context(user="zhouzr", app="web2.0", intent2slots=dict(), context_id="sad")
    import pickle
    with open("log", 'wb') as f:
        pickle.dump(c, f)
    with open("log", "rb") as f:
        x = pickle.load(f)