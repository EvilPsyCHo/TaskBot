# coding:utf8
# @Time    : 18-6-8 下午2:13
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import abc


class Transformer(object):

    def transform(self, *args, **kwargs):
        raise NotImplementedError

    def reverse(self, *args, **kwargs):
        raise NotImplementedError
