# coding:utf8
# @Time    : 18-9-5 上午10:39
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import logging
import pickle

logger = logging.getLogger("SaveLoad")


class SaveLoad(object):
    def save(self, path):
        with open(path, 'wb') as f:
            return pickle.dump(self, f)
        logger.info(f'save {self.__class__.__name__}')

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
        logger.info(f'load {path}')