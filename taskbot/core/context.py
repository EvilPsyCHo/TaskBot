# coding:utf8
# @Time    : 18-7-17 下午4:06
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import datetime as dt


class Context(dict):
    def __init__(self, user, app, cid, timeout=60, **kwargs):
        self._timeout = timeout
        now = dt.datetime.now()
        super().__init__(
            user=user,
            app=app,
            cid=cid,
            history_query=[],
            history_resp=[],
            history_intent=[],
            query=[],
            query_cut=[],
            query_idx=[],
            intent=[],
            entity=None,
            last_query_time=now,
        )

    def recive(self):

    def next_round(self):
        self._record()
        self._refresh()

    def _record(self):
        pass

    def _refresh(self):
        pass
