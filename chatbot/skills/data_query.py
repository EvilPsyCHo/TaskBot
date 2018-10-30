# coding:utf8
# @Time    : 18-6-25 下午2:36
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import datetime as dt
import numpy as np
import pandas as pd

from chatbot.core.skill import BaseSkill
from chatbot.core.entity import TimeInterval, Company, Tag
from chatbot.core.skill import BaseSkill
from chatbot.utils.path import ROOT_PATH
from chatbot.utils.log import get_logger

from common.database import get_pjt_pwr
from common.database import query_powerfactor
from common.database import query_charge
from common.database import query_load

class TestDataQuery(BaseSkill):
    def __call__(self, context):
        t = context["slots"][context["intent"]][TimeInterval.name()]
        p = context["slots"][context["intent"]][Company.name()]
        return "您在查询{}至{}的，{}用电数据".format(
            t["start"],
            t["end"],
            p["alias"]
        )

    def contain_slots(self, entities):
        """

        :param entities: <dict of list>, key: entity name, values: list of entity
        :return:
        """
        for k, v in entities.items():
            if k in self.init_slots().keys():
                return True
        return False

    def init_slots(self):
        return {TimeInterval.name(): TimeInterval(),
                Tag.name(): Tag(),
                Company.name(): Company()}


#  get_pjt_pwr(project, start, end, freq='day', transformer=None):
class _PwrInquiry(BaseSkill):
    """total pwr inquiry回复逻辑封装

    满足skill需求给出回复，不满足需求，给出反问

    :param context: <class Context> 当前会话上下文
    :return: <String> 回复信息
    """

    def __init__(self):
        """initialize

        get the corpus' path
        get the corpus
        """
        #        self.freq_list = ['30min', 'day', 'hour']
        pass

    def __call__(self, context, debug=0):
        """skill回复逻辑封装

        满足skill需求给出回复
        :param context: <class Context>
        :return: <String> 回复信息
        """
        idx_bool, resp = self._check_satisfied(context)
        if debug == 1:  # debug is active
            print('result from check_satisfied: ')
            print(resp)
            print('\n')
        else:  # debug is not active
            pass
        #        idx_bool, resp = _check_satisfied(context)
        if idx_bool is True:  # all the needed slots are filled
            demand_start = str(resp['start'])
            demand_end = str(resp['end'])
            interface_resp = get_pjt_pwr(resp['id'], demand_start, demand_end, freq='day')  # query the data
            if debug == 1:  # debug for interface
                print('result from get_pjt_pwr: ')
                print(interface_resp)
                print('\n')
            if interface_resp['state'] == '100':  # successfully inquiry the data
                data_query = interface_resp['value'].values  # extract the values
                date_query = list(interface_resp['value'].index)
                data_sum = data_query.sum(axis=0).astype('float32')[0]  # sum of the values
                if data_query.shape[0] >= 2:  # query more than 1 days, calculate the max, min and mean
                    # mean method
                    data_mean = data_query.mean().astype('float32')
                    # max method
                    data_max = data_query.max().astype('float32')
                    p_max = np.where(data_max == data_query)[0]
                    date_max = [str(date_query[item])[:10] for item in p_max]
                    date_max = '\n'.join(date_max)
                    # min method
                    data_min = data_query.min().astype('float32')
                    p_min = np.where(data_min == data_query)[0]
                    date_min = [str(date_query[item])[:10] for item in p_min]
                    date_min = '\n'.join(date_min)
                    answer_text = '总电量：%.2f度。\n日平均电量：%.2f度。\n日最大用电量：%.2f度 %s\
                    \n日最小用电量：%.2f度 %s' % (data_sum, data_mean, data_max, date_max, data_min, date_min)
                    # check if the demanded data range is covered
                    valid_start = str(date_query[0].date())  # get the start time of the valid data
                    valid_end = str(date_query[-1].date())
                    # demanded date is not equal to valid date
                    if valid_start != demand_start or valid_end != demand_end:
                        pre_text = '温馨提示：您查询的%s的%s只包含从%s到%s的数据。' % (resp['alias'], resp['tag'], valid_start, valid_end)
                        answer_text = pre_text + '\n' + answer_text
                    else:  # demanded date is equal to valid date
                        pre_text = '%s从%s到%s的%s为：' % (resp['alias'], valid_start, valid_end, resp['tag'])
                        answer_text = pre_text + '\n' + answer_text
                #                    print(answer_text)
                else:  # query only 1 day
                    answer_text = '%s %s的%s为：%.2f度。' % (str(date_query[0])[:10], resp['alias'], resp['tag'], data_sum)
            elif interface_resp['state'] == '200':  # have None value
                answer_text = "未查询到您要的数据。"
            elif interface_resp['state'] == '300':  # project does not exist
                answer_text = "项目不存在或没有权限。"
            else:
                answer_text = "查询数据时出错。"
        else:
            answer_text = resp
        return answer_text

    def init_slots(self):
        """skill name

        :return: <dict> slots dict
        """
        return {
            Tag.name(): Tag(),
            TimeInterval.name(): TimeInterval(),
            Company.name(): Company()
        }

    def contain_slots(self, entities):
        """判断是否包含skill所需词槽

        :param entities: <list of dict>
        :return: <bool> True包含，False不包含
        """
        for k, v in entities.items():
            if k in self.init_slots().keys():
                return True
        return False

    def _check_satisfied(self, context):
        """ check if all the necessary slots are filled

        :param context: <dict>
        :return <bool> True full, False not full
        """
        # get the customer's dict
        slots_received = context['slots'][context['intent']]

        # check the company id
        idx = slots_received['Company']['id']
        if idx is not None:  # company id exist
            # get alias
            alias = slots_received['Company']['alias']
            # convert the string to datetime
            start_time = slots_received['TimeInterval']['start']
            end_time = slots_received['TimeInterval']['end']
            if start_time is not None and end_time is not None:  # start time and end time can't be None
                try:
                    # start_time = dt.datetime.strptime(start_time, '%Y-%m-%d')
                    # end_time = dt.datetime.strptime(end_time, '%Y-%m-%d')
                    start_time = dt.datetime.strptime(start_time, '%Y-%m-%d').date()
                    end_time = dt.datetime.strptime(end_time, '%Y-%m-%d').date()
                except BaseException:
                    return False, "请完整填写起止时间：yyyy-mm-dd"
                if start_time > end_time:  # start time must be small than end time
                    return False, "起始时间应该小于终止时间。"
                else:  # calculate how many days need to be queried
                    if start_time != end_time:
                        valid_day = end_time - start_time
                        valid_day = str(valid_day).split(' ')[0]
                        valid_day = int(valid_day) + 1
                    else:
                        valid_day = 1
                    return True, {
                        'id': idx,
                        'alias': alias,
                        'start': start_time,
                        'end': end_time,
                        'demand_day': valid_day,
                        'tag': slots_received['Tag']['name']
                    }

            #                    # check for frequence
            #                    freq = slots_received['freq']
            #                    if freq not in self.freq_list:  # frequence is not legal
            #                        return False, "频率只能为%s中的一个" % ','.join(self.freq_list)
            #                    else:
            #                        return True, {
            #                                'id':idx,
            #                                'start':start_time,
            #                                'end':end_time,
            #                                'demand_day':valid_day,
            #                                'freq':freq}
            else:
                return False, "请完整填写起止时间：yyyy-mm-dd"
        else:
            return False, "请输入您要查询的公司名称。"


# query_charge(project, start,end,type='pwr'):
class _PFVInquiry(BaseSkill):
    """peak, flat, valley, charge and power inquiry回复逻辑封装

    满足skill需求给出回复，不满足需求，给出反问

    :param context: <class Context> 当前会话上下文
    :return: <String> 回复信息
    """

    def __init__(self):
        """initialize

        get the corpus' path
        get the corpus
        """
        # define the legal type list
        #        self.type_list = ['pwr', 'charge']
        pass

    def __call__(self, context, debug=0):
        """skill回复逻辑封装

        满足skill需求给出回复
        :param context: <class Context>
        :return: <String> 回复信息
        """
        idx_bool, resp = self._check_satisfied(context)
        #        idx_bool, resp = _check_satisfied(context)
        if debug == 1:  # debug is active
            print('result from check_satisfied: ')
            print(resp)
            print('\n')
        else:  # debug is not active
            pass
        if idx_bool is True:  # all the needed slots are filled
            demand_start = str(resp['start'])
            demand_end = str(resp['end'])
            demand_type = resp['type']
            interface_resp = query_charge(resp['id'], demand_start, demand_end, demand_type)  # inquiry the data
            if debug == 1:  # debug for interface
                print('result from query_charge: ')
                print(interface_resp)
                print('\n')
            if interface_resp['state'] == '100':  # successfully get the data\
                # data_column = list(interface_resp['value'].columns)  #  get column name
                data_query = interface_resp['value'].values  # extract the values
                date_query = list(interface_resp['value'].index)
                data_sum = data_query.sum(axis=0).astype('float32')  # sum of the values
                data_sum_all = data_sum.sum(axis=0)

                # get the peak flat valley total pwr or charge
                peak_value = data_sum[1]
                flat_value = data_sum[0]
                valley_value = data_sum[2]

                # calculate the ratio
                ratio_all = np.divide(data_sum, data_sum_all)
                peak_ratio = np.multiply(ratio_all[1], 100)
                flat_ratio = np.multiply(ratio_all[0], 100)
                valley_ratio = np.multiply(ratio_all[2], 100)

                if demand_type == 'pwr':  # answer for the pwr
                    answer_text = "峰期总电量: %.2f度; 占总电量的%.2f%%\n" \
                                  "平期总电量: %.2f度; 占总电量的%.2f%%\n" \
                                  "谷期总电量: %.2f度; 占总电量的%.2f%%\n" % \
                                  (peak_value, peak_ratio, flat_value, flat_ratio, valley_value, valley_ratio)
                else:  # answer for the charge
                    answer_text = "峰期总电费: %.2f元; 占总电费的%.2f%%\n" \
                                  "平期总电费: %.2f元; 占总电费的%.2f%%\n" \
                                  "谷期总电费: %.2f元; 占总电费的%.2f%%\n" % \
                                  (peak_value, peak_ratio, flat_value, flat_ratio, valley_value, valley_ratio)
                    # check if the demanded data range is covered
                valid_start = str(date_query[0].date())  # get the start time of the valid data
                valid_end = str(date_query[-1].date())
                if valid_start != demand_start or valid_end != demand_end:  # demanded date is not equal to valid date
                    pre_text = '温馨提示：您查询的%s的%s的有效范围是从%s到%s。' % (resp['alias'], resp['tag'], valid_start, valid_end)
                    answer_text = pre_text + '\n' + answer_text
                else:
                    pre_text = '%s到%s\n%s的%s为:' % (valid_start, valid_end, resp['alias'], resp['tag'])
                    answer_text = pre_text + '\n' + answer_text
            elif interface_resp['state'] == '200':  # None occur
                answer_text = "未查询到您要的数据。"
            elif interface_resp['state'] == '300':  # project does not exist
                answer_text = "项目不存在或没有权限。"
            else:
                answer_text = "查询数据时出错。"
        else:
            answer_text = resp
        return answer_text

    def init_slots(self):
        """skill name

        :return: <dict> slots dict
        """
        return {
            Tag.name(): Tag(),
            TimeInterval.name(): TimeInterval(),
            Company.name(): Company()
        }

    def contain_slots(self, entities):
        """判断是否包含skill所需词槽

        :param entities: <list of dict>
        :return: <bool> True包含，False不包含
        """
        for k, v in entities.items():
            if k in self.init_slots().keys():
                return True
        return False

    def _check_satisfied(self, context):
        """ check if all the necessary slots are filled

        :param context: <dict>
        :return <bool> True full, False not full
        """
        # get the customer's dict
        slots_received = context['slots'][context['intent']]

        # check the company id
        idx = slots_received['Company']['id']
        if idx is not None:  # company id exist
            # get the alias
            alias = slots_received['Company']['alias']
            # convert the string to datetime
            start_time = slots_received['TimeInterval']['start']
            end_time = slots_received['TimeInterval']['end']
            if start_time is not None and end_time is not None:  # start time and end time can't be None
                try:
                    # start_time = dt.datetime.strptime(start_time, '%Y-%m-%d')
                    # end_time = dt.datetime.strptime(end_time, '%Y-%m-%d')
                    start_time = dt.datetime.strptime(start_time, '%Y-%m-%d').date()
                    end_time = dt.datetime.strptime(end_time, '%Y-%m-%d').date()
                except BaseException:
                    return False, "请完整填写起止时间：yyyy-mm-dd"
                if start_time > end_time:  # start time must be small than end time
                    return False, "起始时间应该小于终止时间。"
                else:  # calculate how many days need to be queried
                    if start_time != end_time:
                        valid_day = end_time - start_time
                        valid_day = str(valid_day).split(' ')[0]
                        valid_day = int(valid_day) + 1
                    else:
                        valid_day = 1
                    tag_demanded = slots_received['Tag']['name']
                    if tag_demanded == '峰平谷电量':
                        type_demanded = 'pwr'
                        return True, {
                            'id': idx,
                            'alias': alias,
                            'start': start_time,
                            'end': end_time,
                            'demand_day': valid_day,
                            'type': type_demanded,
                            'tag': tag_demanded
                        }
                    elif tag_demanded == '峰平谷电费':
                        type_demanded = 'charge'
                        return True, {
                            'id': idx,
                            'alias': alias,
                            'start': start_time,
                            'end': end_time,
                            'demand_day': valid_day,
                            'type': type_demanded,
                            'tag': tag_demanded
                        }
                    else:
                        return False, "只能查询峰平谷电量或峰平谷电费"
            else:
                return False, "请完整填写起止时间：yyyy-mm-dd"
        else:
            return False, "请输入您要查询的公司名称。"


# query_powerfactor(project, start, end)
class _PwrFactorInquiry(BaseSkill):
    """power factor inquiry回复逻辑封装

    满足skill需求给出回复，不满足需求，给出反问

    :param context: <class Context> 当前会话上下文
    :return: <String> 回复信息
    """

    def __init__(self):
        """initialize

        get the corpus' path
        get the corpus
        """
        pass

    def __call__(self, context, debug=0):
        """skill回复逻辑封装

        满足skill需求给出回复
        :param context: <class Context>
        :return: <String> 回复信息
        """
        idx_bool, resp = self._check_satisfied(context)
        #        idx_bool, resp = _check_satisfied(context)
        if debug == 1:  # debug is active
            print('result from check_satisfied: ')
            print(resp)
            print('\n')
        else:  # debug is not active
            pass
        if idx_bool is True:  # all the needed slots are filled
            demand_start = str(resp['start'])
            demand_end = str(resp['end'])
            interface_resp = query_powerfactor(resp['id'], demand_start, demand_end)  # inquiry the data
            #            tmp = query_load(resp['id'], resp['start'])
            if debug == 1:  # debug for interface
                print('result from query_powerfactor: ')
                print(interface_resp)
                print('\n')
            if interface_resp['state'] == '100':  # successfully get the data\
                # data_column = list(interface_resp['value'].columns)  #  get column name
                data_query = interface_resp['value'].values[0].astype('float32')  # extract the values
                date_query = list(interface_resp['value'].index)
                date_query = list(map(str, date_query))
                # data_query = np.array([0.3,0.5])
                # date_query = ['2018-07-01 00:00:00','2018-07-02 00:00:00']
                answer_text = ['%.4f, %s\n' %
                               (data_query[i], date_query[i][:7]) for i in range(len(data_query))]
                answer_text = ''.join(answer_text)
                pre_text = "为%s查询到的%s为：\n" % (resp['alias'], resp['tag'])
                answer_text = pre_text + answer_text
            elif interface_resp['state'] == '200':  # None occur
                answer_text = "未查询到您要的数据。"
            elif interface_resp['state'] == '300':  # project does not exist
                answer_text = "项目不存在或没有权限。"
            else:
                answer_text = "查询数据时出错。"
        else:
            answer_text = resp
        return answer_text

    def init_slots(self):
        """skill name

        :return: <dict> slots dict
        """
        return {
            Tag.name(): Tag(),
            TimeInterval.name(): TimeInterval(),
            Company.name(): Company()
        }

    def contain_slots(self, entities):
        """判断是否包含skill所需词槽

        :param entities: <list of dict>
        :return: <bool> True包含，False不包含
        """
        for k, v in entities.items():
            if k in self.init_slots().keys():
                return True
        return False

    def _check_satisfied(self, context):
        """ check if all the necessary slots are filled

        :param context: <dict>
        :return <bool> True full, False not full
        """
        # get the customer's dict
        slots_received = context['slots'][context['intent']]

        # check the company id
        idx = slots_received['Company']['id']
        if idx is not None:  # company id exist
            # get the alias
            alias = slots_received['Company']['alias']
            # convert the string to datetime
            start_time = slots_received['TimeInterval']['start']
            end_time = slots_received['TimeInterval']['end']
            if start_time is not None and end_time is not None:  # start time and end time can't be None
                try:
                    # start_time = dt.datetime.strptime(start_time,'%Y-%m-%d')
                    # end_time = dt.datetime.strptime(end_time,'%Y-%m-%d')
                    start_time = dt.datetime.strptime(start_time, '%Y-%m-%d').date()
                    end_time = dt.datetime.strptime(end_time, '%Y-%m-%d').date()
                except BaseException:
                    return False, "请完整填写起止时间：yyyy-mm-dd。"
                if start_time > end_time:  # start time must be small than end time
                    return False, "起始时间应该小于终止时间。"
                else:  # calculate how many days need to be queried
                    if start_time != end_time:
                        valid_day = end_time - start_time
                        valid_day = str(valid_day).split(' ')[0]
                        valid_day = int(valid_day) + 1
                    else:
                        valid_day = 1

                    # get the tag
                    tag = slots_received['Tag']['name']
                    return True, {
                        'id': idx,
                        'alias': alias,
                        'start': start_time,
                        'end': end_time,
                        'demand_day': valid_day,
                        'tag': tag
                    }
            else:
                return False, "请完整填写起止时间：yyyy-mm-dd。"
        else:
            return False, "请输入您要查询的公司名称。"


class DataInquiry(BaseSkill):
    """data inquiry回复逻辑封装

    满足skill需求给出回复，不满足需求，给出反问

    :param context: <class Context> 当前会话上下文
    :return: <String> 回复信息
    """

    def __init__(self):
        """initialize

        get the corpus' path
        get the corpus
        """
        # initializing the instances
        self.pwrinquiry = _PwrInquiry()
        self.pfvinquiry = _PFVInquiry()
        self.pwrfactorinquiry = _PwrFactorInquiry()

    def init_slots(self):
        """skill name

        :return: <dict> slots dict
        """
        return {
            Tag.name(): Tag(),
            TimeInterval.name(): TimeInterval(),
            Company.name(): Company()
        }

    def contain_slots(self, entities):
        """判断是否包含skill所需词槽

        :param entities: <list of dict>
        :return: <bool> True包含，False不包含
        """
        for k, v in entities.items():
            if k in self.init_slots().keys():
                return True
        return False

    def __call__(self, context, debug=1):
        """skill回复逻辑封装

        满足skill需求给出回复
        :param context: <class Context>
        :return answer_text: <String> 回复信息
        """
        # get the concrete data inquiry intention
        tag = context['slots'][context['intent']]['Tag']['name']
        if tag == '电量':
            answer_text = self.pwrinquiry(context, debug=debug)
        elif tag == '峰平谷电量' or tag == '峰平谷电费':
            answer_text = self.pfvinquiry(context, debug=debug)
        elif tag == '功率因数':
            answer_text = self.pwrfactorinquiry(context, debug=debug)
        else:
            raise ValueError("unknown data inquiry intention")
        return answer_text
