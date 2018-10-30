# coding:utf8
# @Time    : 18-5-21 下午3:55
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import numpy as np


def accuracy(y_true, y_pred):
    return np.equal(y_pred, y_true).sum() / y_true.shape[0]