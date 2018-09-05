# coding:utf8
# @Time    : 18-9-5 上午11:42
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from taskbot.utils.common import (reverse_dict)


class TestUtilsCommon:

    def test_revers_dict(self):
        o = {2: "a", "c": 3}
        t = {"a": 2, 3: "c"}
        assert reverse_dict(o) == t
