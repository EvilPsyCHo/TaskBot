# coding:utf8
# @Time    : 18-6-8 下午2:19
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import abc


class Serializable(object):

    def save(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractclassmethod
    def load(cls, *args, **kwargs):
        raise NotImplementedError
