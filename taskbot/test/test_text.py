# coding:utf8
# @Time    : 18-9-5 下午4:10
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import os

from taskbot.text import Dictionary


class TestText:

    def test_Dictionary(self):
        dictionary = Dictionary()
        dictionary.save("./test_dictionary")
        dictionary = Dictionary.load("./test_dictionary")
        os.remove("./test_dictionary")