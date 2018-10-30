# coding:utf8
# @Time    : 18-6-8 下午3:40
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import abc


class Extractor(object):

    def extract(self, context):
        raise NotImplementedError

    def transform(self, context):
        raise NotImplementedError