# coding:utf8
# @Time    : 18-6-11 下午2:41
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import random
import codecs

import numpy as np
import pandas as pd

from chatbot.core.skill import BaseSkill
from chatbot.utils.path import ROOT_PATH
from chatbot.utils.log import get_logger

from common.database import get_pjt_pwr
from common.database import query_powerfactor
from common.database import query_charge
from common.database import query_load


logger = get_logger("simple skill")


def read_txt(path):
        """read txt

        :return: <list>
        """        
#        path = "D:\\Users\\tanmx\\chatbot\\Task-Oriented-Chatbot\\corpus\\skill\\GoodBye_response.txt"
        with open(path, "r", encoding='UTF-8') as f:
            txts = f.readlines()
        # remove chomp, blank
        sents = [item.strip().split(' ')[-1] for item in txts if len(item) > 1]
        return sents


class LeaveMessage(BaseSkill):
    """LeaveMessage存储及回复封装
      :param context: context
      :return: <String> 回复信息,context{user:,query} to txt
      """
    def __init__(self, path=None):
        if path is None:
            self.path = str(ROOT_PATH / "log" / "message")
        else:
            self.path = path
        logger.debug("leave message save in %s" % self.path)

    def __call__(self, context):
        with codecs.open(self.path, "a+", "utf-8") as f:
            f.write(context['context_id'] + '\t' + '\t' + context['app'] + '\t' + '\t' + str(context[
                'last_query_time']) + '\t' + '\t' + context['user'] + '\t' + '\t' + context['query'] + '\n' + '\n')
        f.close()
        return "您的反馈小益已经记下来了哦，回去就告诉他们：）"

    def contain_slots(self, entities):
        return False

    def init_slots(self):
        return {}


class SayHi(BaseSkill):
    """SayHi回复逻辑封装

    满足skill需求给出回复，不满足需求，给出反问
    
    :param context: <class Context> 当前会话上下文
    :return: <String> 回复信息
    """    
    
    def __init__(self):
        """initialize
        
        get the corpus' path
        get the corpus
        """
        print("loading response for SayHi...")
        path = str(ROOT_PATH.parent / "corpus"/"intent"/"skill"/"SayHi_response.txt")
        self.reps = read_txt(path)

    def __call__(self, context):
        """skill回复逻辑封装
        
        满足skill需求给出回复
        :param context: <class Context>
        :return: <String> 回复信息
        """
        answer_text = random.choice(self.reps)
        answer_text = answer_text + context["user"]
        return answer_text

    def init_slots(self):
        """skill name

        :return: <dict> slots dict
        """
        return {}

    def contain_slots(self, entities):
        """判断是否包含skill所需词槽

        :param entities: <list of dict>
        :return: <bool> True包含，False不包含
        """
        return False


class GoodBye(BaseSkill):
    """GoodBye回复逻辑封装

    满足skill需求给出回复，不满足需求，给出反问
    
    :param context: <class Context> 当前会话上下文
    :return: <String> 回复信息
    """    
    
    def __init__(self):
        """initialize
        
        get the corpus' path
        get the corpus
        """
        print("loading response for GoodBye...")
        path = str(ROOT_PATH.parent / "corpus"/"intent"/"skill"/"GoodBye_response.txt")
        self.reps = read_txt(path)

    def __call__(self, context):
        """skill回复逻辑封装
        
        满足skill需求给出回复
        :param context: <class Context>
        :return: <String> 回复信息
        """
        answer_text = random.choice(self.reps)
        answer_text = answer_text + context["user"]
        return answer_text

    def init_slots(self):
        """skill name

        :return: <dict> slots dict
        """
        return {}

    def contain_slots(self, entities):
        """判断是否包含skill所需词槽

        :param entities: <list of dict>
        :return: <bool> True包含，False不包含
        """
        return False

        
class Thanks(BaseSkill):
    """Thanks回复逻辑封装

    满足skill需求给出回复，不满足需求，给出反问
    
    :param context: <class Context> 当前会话上下文
    :return: <String> 回复信息
    """    
    
    def __init__(self):
        """initialize
        
        get the corpus' path
        get the corpus
        """
        print("loading response for Thanks...")
        path = str(ROOT_PATH.parent / "corpus"/"intent"/"skill"/"Thanks_response.txt")
        self.reps = read_txt(path)

    def __call__(self, context):
        """skill回复逻辑封装
        
        满足skill需求给出回复
        :param context: <class Context>
        :return: <String> 回复信息
        """
        answer_text = random.choice(self.reps)
        answer_text = answer_text + context["user"]
        return answer_text

    def init_slots(self):
        """skill name

        :return: <dict> slots dict
        """
        return {}

    def contain_slots(self, entities):
        """判断是否包含skill所需词槽

        :param entities: <list of dict>
        :return: <bool> True包含，False不包含
        """
        return False


class Praise(BaseSkill):
    """Praise回复逻辑封装

    满足skill需求给出回复，不满足需求，给出反问
    
    :param context: <class Context> 当前会话上下文
    :return: <String> 回复信息
    """    
    
    def __init__(self):
        """initialize
        
        get the corpus' path
        get the corpus
        """
        print("loading response for Praise...")
        path = str(ROOT_PATH.parent / "corpus"/"intent"/"skill"/"Praise_response.txt")
        self.reps = read_txt(path)

    def __call__(self, context):
        """skill回复逻辑封装
        
        满足skill需求给出回复
        :param context: <class Context>
        :return: <String> 回复信息
        """
        answer_text = random.choice(self.reps)
        answer_text = answer_text + context["user"]
        return answer_text

    def init_slots(self):
        """skill name

        :return: <dict> slots dict
        """
        return {}

    def contain_slots(self, entities):
        """判断是否包含skill所需词槽

        :param entities: <list of dict>
        :return: <bool> True包含，False不包含
        """
        return False


class Criticize(BaseSkill):
    """Criticize回复逻辑封装

    满足skill需求给出回复，不满足需求，给出反问
    
    :param context: <class Context> 当前会话上下文
    :return: <String> 回复信息
    """    
    
    def __init__(self):
        """initialize
        
        get the corpus' path
        get the corpus
        """
        print("loading response for Criticize...")
        path = str(ROOT_PATH.parent / "corpus"/"intent"/"skill"/"Criticize_response.txt")
        self.reps = read_txt(path)

    def __call__(self, context):
        """skill回复逻辑封装
        
        满足skill需求给出回复
        :param context: <class Context>
        :return: <String> 回复信息
        """
        answer_text = random.choice(self.reps)
        answer_text = answer_text + context["user"]
        return answer_text
    
    def init_slots(self):
        """skill name

        :return: <dict> slots dict
        """
        return {}

    def contain_slots(self, entities):
        """判断是否包含skill所需词槽

        :param entities: <list of dict>
        :return: <bool> True包含，False不包含
        """
        return False


class CompanyInfo(BaseSkill):
    """Criticize回复逻辑封装

    满足skill需求给出回复，不满足需求，给出反问

    :param context: <class Context> 当前会话上下文
    :return: <String> 回复信息
    """

    def __init__(self):
        """initialize

        get the corpus' path
        get the corpus
        """

        print("loading company information...")
        path = str(ROOT_PATH.parent / "corpus" / "intent" / "skill" / "company_info.csv")
        company_info = pd.read_csv(path)

        # save company information into a dictionary
        self.company_info_dict = {}
        self.company_info_dict.update({
            company_info['item'][i]: company_info['content'][i] for i in range(len(company_info['item']))
        })
        #   company_info_dict.get('ce',company_info_dict['ceo'])

        # sort the company information's keys according to weight
        company_keys = company_info['item'].values
        company_keys_weight = company_info['weight'].values
        self.company_keys = np.array([
            x for _, x in sorted(zip(company_keys_weight, company_keys), key=lambda pair: pair[0])
        ])

    def __call__(self, context, answer_single=1, debug=0):
        """skill回复逻辑封装

        满足skill需求给出回复
        :param
            context: <class Context>
            answer_single: <int>  1: generate 1 answer by the most priority key word;
                                  0: generate multiple answers by all the key words
            debug: <int> 1: the answer will show the key words;
                         0: the answer will not show the key words
        :return: <String> 回复信息
        """
        # extract the key words, remove the duplication
        customer_keys = np.array(list(set(context["query_cut"]))).reshape(-1, 1)

        # match the customer's keys with the company info keys
        match_idx = customer_keys == self.company_keys
        match_idx = match_idx.sum(axis=0)

        # find the answer from the company information
        key_position = np.where(match_idx == 1)[0]
        if len(key_position) > 0:  # successfully matched the key words

            # generate 1 answer according to the most priority key
            if answer_single == 1:
                key_position = key_position[0]
                key_priority = self.company_keys[key_position]
                company_answer = self.company_info_dict[key_priority]
                if debug == 1:
                    answer_text = key_priority + ": " + company_answer
                else:
                    answer_text = company_answer
            else:  # generate multiple answers if possible
                key_priority = self.company_keys[key_position]

                # if not an array then convert it
                if not isinstance(key_priority, np.ndarray):
                    key_priority = np.array([key_priority])
                company_answer = np.array([self.company_info_dict[key] for key in key_priority])

                # collect all the possible answer
                if debug == 1:
                    answer_list = [key_priority[num] + ": " + company_answer[num] for num in range(len(key_priority))]
                else:
                    answer_list = [company_answer[num] for num in range(len(key_priority))]
                answer_text = "\n".join(answer_list)
        else:  # fail to match the key words then use default answer
            answer_text = self.company_info_dict["公司简介"]
        return answer_text

    def init_slots(self):
        """skill name

        :return: <dict> slots dict
        """
        return {}

    def contain_slots(self, entities):
        """判断是否包含skill所需词槽

        :param entities: <list of dict>
        :return: <bool> True包含，False不包含
        """
        return False


class BusinessInfo(BaseSkill):
    """Criticize回复逻辑封装

    满足skill需求给出回复，不满足需求，给出反问

    :param context: <class Context> 当前会话上下文
    :return: <String> 回复信息
    """

    def __init__(self):
        """initialize

        get the corpus' path
        get the corpus
        """
        print("loading response for Criticize...")
        self.reps = "万益能源的主要业务有：\n1）购电支持服务：" + \
                    "为用电客户提供购电相关的各项服务，" + \
                    "从负荷预测、偏差风险分析、竞价策略设计、" + \
                    "偏差监控及管理等各个环节提供服务及解决方案。" + \
                    "\n2)数字化能源服务：面向多样化客户、提供智能报表，为管理、" + \
                    "节能提供自动化预警及智能化决策。"

    def __call__(self, context):
        """skill回复逻辑封装

        满足skill需求给出回复
        :param context: <class Context>
        :return: <String> 回复信息
        """
        answer_text = self.reps
        return answer_text

    def init_slots(self):
        """skill name

        :return: <dict> slots dict
        """
        return {}

    def contain_slots(self, entities):
        """判断是否包含skill所需词槽

        :param entities: <list of dict>
        :return: <bool> True包含，False不包含
        """
        return False

















