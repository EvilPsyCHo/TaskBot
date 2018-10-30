# -*- coding: utf-8 -*-
# @Time    : 2018/7/4 15:05
# @Author  : Lin
# @File    : chatRecord.py
import os
from chatbot.utils.path import LOG_PATH
import pandas as pd
import codecs


def chat_record(context):
    """
    聊天记录存储到txt
    :param context: context
    :return: None
    """
    path_txt = str(LOG_PATH / "chat_record")
    with codecs.open(path_txt, "a+", "utf-8") as f:
        f.write(context['context_id'] + '\t' + '\t' + context['app'] + '\t' + '\t' + str(context['last_query_time']) + '\t'
                + '\t' + context['intent'] + '\t' + '\t' + str(context['slots']) + '\t' + '\t' + context['user'] + '\t'
                + '\t' + context['query'] + '\t' + '\t' + context['history_resp'][-1] + '\n' * 4)


def record2df(path):
    """
    解析txt转化为Dataframe
    :param path: 文件路径
    :return: Dataframe
    """
    res = []
    with codecs.open(path, "r", "utf-8") as f:
        text = [line.split('\t'+'\t') for line in f.read().split("\n" * 4)]
        frame = pd.DataFrame(text).dropna()
    return frame


if __name__ == "__main__":
    # test = {'context_id': '1', 'app': '益电宝', 'user': 'abc', 'query': '请你们公司负责市场的同事跟我联系一下，电话14541234654',
    #         'last_query_time': '2018-07-04 15:16', 'intent': '留言',
    #         'slots': "{'TimeInterval': {'start': '1454-12-34', 'end': '1454-12-34'}}",
    #         'history_resp': ['你好', '小益已经帮你记下来']}
    # chat_record(test)
    paths = str(LOG_PATH / "chat_record")
    import os
    os.path.exists(paths)
    a = record2df(paths)
    print(a)
