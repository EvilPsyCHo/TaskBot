# coding:utf8
# @Time    : 18-7-13 上午9:16
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from pathlib import Path


__ALL__ = [
    "PAD", "PAD_IDX", "UNKNOWN", "UNKNOWN_IDX",
    "PATH"
]


# Word
PAD = "<pad>"
PAD_IDX = 1
UNKNOWN = "<unknown>"
UNKNOWN_IDX = 0


# Path
class TaskBotPath(object):
    def __init__(self):
        self.root = Path(__file__).resolve().parent


PATH = TaskBotPath()
