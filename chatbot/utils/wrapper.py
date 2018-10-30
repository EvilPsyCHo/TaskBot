# -*- coding: utf-8 -*-
# @Time    : 5/12/18 14:17
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import time
from functools import wraps

from chatbot.utils.log import get_logger


time_counter_logger = get_logger("Time Counter")


def time_counter(func):
    @wraps(func)
    def time_it(*args, **kwargs):
        s = time.time()
        rst = func(*args, **kwargs)
        e = time.time()
        t = (e - s) / 60
        time_counter_logger.info(func.__name__ + " running %.2f min" % t)
        return rst
    return time_it
