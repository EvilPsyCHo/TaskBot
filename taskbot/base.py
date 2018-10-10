# -*- coding: utf-8 -*-
# @Time    : 9/4/18 14:19
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import pickle


__all__ = ["SaveLoad"]


class SaveLoad(object):
    def save(self, path):
        with open(path, 'wb') as f:
            return pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)



