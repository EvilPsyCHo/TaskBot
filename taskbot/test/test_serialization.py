# coding:utf8
# @Time    : 18-9-5 上午11:00
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import os

from taskbot.utils.serialization import SaveLoad


class TestSaveLoad:

    def test_saveload(self):
        sl = SaveLoad()
        path = "./test_saveload"
        sl.save(path)
        assert os.path.exists("./test_saveload")
        sl = sl.load(path)
        assert sl is not None
        os.remove(path)
