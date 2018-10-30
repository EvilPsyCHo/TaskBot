# coding:utf8
# @Time    : 18-7-20 上午9:59
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import calendar
import re
import datetime
from chatbot.core.entity import TimeInterval


class TimeNer(object):

    def extract(self, context):
        """

        Args:
            context:

        Returns:
            List or None
        """
        pattern = [
            r'(\d{2,4}-\d{1,2}-\d{1,2})|(\d{8})|(\d{6})|(\d{2,4}/\d{1,2}/\d{1,2})|(\d{2,4}\.\d{1,2}\.\d{1,2})|'
            r'(\d{2,4}\.\d{1,2})|(\d{1,2}\.\d{1,2})|(\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2})|(\d{2,4}年\d{1,2}月)|'
            r'(\d{2,4}年\d{1,2}月\d{1,2}[日|号])|'
            r'(.[去年|明年|后年|前年|今年][\d|一|二|三|四|五|六|七|八|九|十|两][月].\d[日|号])|(.[去年|明年|后年|前年|今年]\d{1,2}[月])|'
            r'([去年|明年|后年|前年|今年](.*)季度)|([今|明|昨|去|前|后][年][今|明|昨|前|后][天|日])|([今|明|昨|去|前|后][天|年|日])|'
            r'(\d{2,4}[年][\d|一|二|三|四][季度].)|(\d{4}[年][第].[季度].)|([第].[季度].)|'
            r'([\d|一|二|三|四][季度].)|([本|上|下][月].*[日|号])|'
            r'.*?((?:下个月|上个月|这个月).*?[日|号])|([上|下|本][月|年])|((?:下个|上个|这个)[月])|.*?((?:过去|未来|最近).*?[年|月|天|日])|'
            r'(\d{2,4}年\d{1,2}月)|(\d{1,2}月\d{1,2}[日|号])|([一|二|三|四|五|六|七|八|九|十|十一|十二]{1,2}月\d{1,2}[日|号])|'
            r'(\d{1,2}[月])|([一|二|三|四|五|六|七|八|九|十|十一|十二]{1,2}[月])|(\d{1,2}[号|日])|([一|二|三|四|五|六|七|八|九|十][号|日])|'
            r'(\d{2,4}[年])|([上|下|这|本][周][一|二|三|四|五|六|天|\d])|([上|下|这|本][星期|礼拜].. *?)|'
            r'((\d{1}|一|二|三|四|五|六|七|八|九|两)(周|天|年)(前|后))|'
            r'((上|前|后)(\d{1}|一|二|三|四|五|六|七|八|九|两)(周|天|年))|'
            r'([上|下|这|本][周])|([周][\d|一|二|三|四|五])|((?:下个|上个|这个)(星期|礼拜)[\d|一|二三|四|五|六|七|天])|'
            r'(((?:下个|上个|这个)(星期|礼拜)))|([本|下|上][季度].)|((?:下个|上个|这个)(季度))|(最新|最近|近期|现在)']

        # pattern = [r'((\d{1}|一|二|三|四|五|六|七|八|九|两)(周|天|年)(前|后))']
        result_origin = []
        for i in range(len(pattern)):
            res = re.findall(pattern[i], context['query'], re.S)
            if len(res) != 0:
                result_origin.append(res)
            else:
                continue
        extract_result = []
        for i in range(len(result_origin)):
            if len(result_origin) > 1:
                if isinstance(result_origin[i][0], str):
                    for k in range(len(result_origin[i])):
                        extract_result.append(result_origin[i][k])
                if isinstance(result_origin[i][0], tuple):
                    for j in range(len(result_origin[i])):
                        res_a = list(result_origin[i][j])
                        while '' in res_a:
                            res_a.remove('')
                        extract_result.append(res_a[0])
            else:
                if isinstance(result_origin[0][0], str):
                    extract_result.append(result_origin[0][0])
                if isinstance(result_origin[0][0], tuple):
                    for q in range(len(result_origin[0])):
                        res_deal = list(result_origin[0][q])
                        while '' in res_deal:
                            res_deal.remove('')
                        extract_result.append(res_deal[0])
        if len(extract_result) == 0:
            return None
        else:
            return extract_result

    def transform(self, context):
        """
        :param context:
        :return: <list of time entity>
        """
        res = self.extract(context)
        res1 = self._infer_time_standard(context)
        res2 = self._infer_time_anglicize(context)
        res3 = self._infer_time_mixed(context)
        res4 = self._infer_time_contact(context)
        if res is not None:
            if len(res4) == 0:
                res = res1 + res2 + res3
            elif len(res4) == 1 and len(res2) == 2:
                res = res4
            elif len(res4) == 1 and len(res3) == 2:
                res = res4
            else:
                res = res4 + res2 + res3
        trans_time = res
        time_output_result = []
        if trans_time is not None:
            for i in range(len(trans_time)):
                transform = TimeInterval()
                if '~' in trans_time[i]:
                    transform['end'] = trans_time[i].split('~')[1]
                    transform['start'] = trans_time[i].split('~')[0]
                else:
                    transform['end'] = trans_time[i]
                    transform['start'] = trans_time[i]
                time_output_result.append(transform)
        if len(time_output_result) == 0:
            return None
        else:
            return time_output_result[0]


    @staticmethod
    def _has_numbers(string):
        return any(char.isdigit() for char in string)

    @staticmethod
    def _num_transform(i):
        """中文大写数字转换阿拉伯数字"""
        if ('一' in i or '二' in i or '两' in i or '三' in i or '四' in i or '五' in i or '六' in i or
                '七' in i or '八' in i or '九' in i or '十' in i or '天' in i):
            try:
                i = i.replace('十一', '11')
                i = i.replace('十二', '12')
                i = i.replace('一', '1')
                i = i.replace('二', '2')
                i = i.replace('两', '2')
                i = i.replace('三', '3')
                i = i.replace('四', '4')
                i = i.replace('五', '5')
                i = i.replace('六', '6')
                i = i.replace('七', '7')
                if '星期' in i or '礼拜' in i or '周' in i:
                    i = i.replace('天', '7')
                i = i.replace('八', '8')
                i = i.replace('九', '9')
                i = i.replace('十', '10')
            except Exception as e:
                print(e)
        return i

    def _infer_time_standard(self, context):
        """
        标准型时间映射
        :param context:
        :return:
        """
        result = self.extract(context)
        years = str(datetime.datetime.now().year)
        months = str(datetime.datetime.now().year)
        if result is not None:
            convert_result1 = []
            # user_param = TEST()
            for i in result:
                i = self._num_transform(i)
                if self._has_numbers(i) is True and '周' not in i and '星期' not in i and '礼拜' not in i and '季' not in i \
                        and '去年' not in i and '今年' not in i and '前年' not in i and '后年' not in i and '明年' not in i \
                        and '上' not in i and '下' not in i and '本' not in i and '这' not in i and '过去' not in i \
                        and '未来' not in i and '最近' not in i and '前' not in i and '后' not in i:
                    days = ''
                    time_convert = ''
                    '''时间格式的映射'''
                    if '.' in i or '-' in i or '/' in i:
                        if '.' in i:
                            if len(i.split('.')) == 3:
                                if len(i.split('.')[0]) == 4:
                                    years = i.split('.')[0]
                                if len(i.split('.')[0]) == 2:
                                    years = str(2000 + int(i.split('.')[0]))
                                if int(
                                        i.split('.')[1]) < 10 and len(
                                        i.split('.')[1]) == 1:
                                    months = '0' + (i.split('.')[1])
                                if len(i.split('.')[1]) == 2:
                                    months = i.split('.')[1]
                                if int(i.split('.')[2]) < 10 and len(i.split('.')[2]) == 1:
                                    days = '0' + (i.split('.')[2])
                                if len(i.split('.')[2]) == 2:
                                    days = i.split('.')[2]
                            elif len(i.split('.')) == 2:
                                if len(i.split('.')[0]) == 4:
                                    years = i.split('.')[0]
                                    months = i.split('.')[1]
                                    if int(months) < 10 and len(months) == 1:
                                        months = '0' + months
                                    else:
                                        months = months
                                    month_range = calendar.monthrange(
                                        int(years), int(months))
                                    days = ['01', str(month_range[1])]
                                if len(i.split('.')[0]) == 2:
                                    years = str(datetime.datetime.now().year)
                                    months = i.split('.')[0]
                                    if int(months) < 10 and len(months) == 1:
                                        months = '0' + months
                                    else:
                                        months = months
                                    if int(i.split('.')[1]) < 10 and len(i.split('.')[1]) == 1:
                                        days = '0' + (i.split('.')[1])
                                    else:
                                        days = i.split('.')[1]
                        if '/' in i:
                            if len(i.split('/')[0]) == 4:
                                years = i.split('-')[0]
                            if len(i.split('/')[1]) == 2:
                                years = str(2000 + int(i.split('/')[0]))
                            if int(i.split('/')
                                   [1]) < 10 and len(i.split('/')[1]) == 1:
                                months = '0' + (i.split('/')[1])
                            if len(i.split('/')[1]) == 2:
                                months = i.split('/')[1]
                            if int(i.split('/')
                                   [2]) < 10 and len(i.split('/')[2]) == 1:
                                days = '0' + (i.split('/')[2])
                            if len(i.split('/')[2]) == 2:
                                days = i.split('/')[2]
                        if '-' in i:
                            if len(i.split('-')[0]) == 4:
                                years = i.split('-')[0]
                            if len(i.split('-')[0]) == 2:
                                years = str(2000 + int(i.split('-')[0]))
                            if int(i.split('-')
                                   [1]) < 10 and len(i.split('-')[1]) == 1:
                                months = '0' + (i.split('-')[1])
                            if len(i.split('-')[1]) == 2:
                                months = i.split('-')[1]
                            if int(i.split('-')
                                   [2]) < 10 and len(i.split('-')[2]) == 1:
                                days = '0' + (i.split('-')[2])
                            if len(i.split('-')[2]) == 2:
                                days = i.split('-')[2]
                        if isinstance(days, str):
                            time_convert = years + '-' + months + '-' + days
                        else:
                            res = [years + '-' + months + '-' + day for day in days]
                            time_convert = res[0] + '~' + res[1]
                    if i.isdigit() is True and len(i) == 8:
                        years = i[0:4]
                        months = i[4:6]
                        days = i[6:8]
                        time_convert = years + '-' + months + '-' + days
                    if i.isdigit() is True and len(i) == 6:
                        years = i[0:4]
                        months = i[4:6]
                        month_range = calendar.monthrange(
                            int(years), int(months))
                        days = ['01', str(month_range[1])]
                        res = [years + '-' + months + '-' + day for day in days]
                        time_convert = res[0] + '~' + res[1]

                    '''年月日格式的映射'''
                    if '年' in i or '月' in i or '日' in i or '号' in i:
                        if '年' in i and '月' not in i and '日' not in i and '号' not in i:
                            time_convert = str(
                                i.split('年')[0]) + '-01-01' + '~' + str(i.split('年')[0]) + '-12-31'
                        else:
                            if '年' in i:
                                find_index_year = i.index('年')
                                find_year = i[0:int(find_index_year)]
                                if len(find_year) == 4:
                                    years = find_year
                                if len(find_year) < 4:
                                    years = str(2000 + int(find_year))
                            else:
                                years = str(datetime.datetime.now().year)
                            if '月' in i:
                                try:
                                    find_index_year = i.index('年')
                                except ValueError:
                                    find_index_year = -1
                                find_index_month = i.index('月')
                                find_month = i[int(
                                    find_index_year) + 1:int(find_index_month)]
                                months = find_month
                            else:
                                months = str(datetime.datetime.now().month)
                            if int(months) < 10 and len(months) == 1:
                                months = '0' + months
                            else:
                                months = months
                            if '日' in i or '号' in i:
                                try:
                                    find_index_month = i.index('月')
                                except ValueError:
                                    find_index_month = -1
                                try:
                                    find_index_days = i.index('日')
                                except ValueError:
                                    find_index_days = i.index('号')
                                find_days = i[int(
                                    find_index_month) + 1:int(find_index_days)]
                                days = find_days
                                if int(days) < 10 and len(days) == 1:
                                    days = '0' + days
                                else:
                                    days = days
                            else:
                                month_range = calendar.monthrange(int(years), int(months))
                                days = ['01', str(month_range[1])]
                            if isinstance(days, list):
                                res = [years + '-' + months + '-' + day for day in days]
                                time_convert = res[0] + '~' + res[1]
                            else:
                                time_convert = years + '-' + months + '-' + days
                    convert_result1.append(time_convert)
            return convert_result1
        else:
            return None

    def _infer_time_mixed(self, context):
        """
        混合型时间映射
        :param context:
        :return:
        """
        result = self.extract(context)
        if result is not None:
            convert_result = []
            time_convert = ''
            years = str(datetime.datetime.now().year)
            months = str(datetime.datetime.now().year)
            for i in result:
                i = self._num_transform(i)
                if (self._has_numbers(i) is True and
                        ('最近' in i or '本年' in i or '去年' in i or '今年' in i or '前年' in i
                         or '后年' in i or '明年' in i or '上' in i or '下' in i
                         or '前' in i or '后' in i
                         or '过去' in i or '未来' in i)) or (self._has_numbers(i) is True and '季' in i):
                    if '年' in i and '月' in i and '日' not in i and '号' not in i:
                        months = re.search(r'(\d{1,2})月', i, re.S).group(1)
                        if int(months) < 10 and len(months) == 1:
                            months = '0' + months
                        else:
                            months = months
                        years = str(datetime.datetime.now().year)
                        month_range = calendar.monthrange(
                            int(years), int(months))
                        days = ['01', str(month_range[1])]
                        if '今年' in i:
                            years = years
                        if '明年' in i:
                            years = str(datetime.datetime.now().year + 1)
                        if '前年' in i:
                            years = str(datetime.datetime.now().year - 2)
                        if '后年' in i:
                            years = str(datetime.datetime.now().year + 2)
                        if '去年' in i:
                            years = str(datetime.datetime.now().year - 1)
                        res = [
                            years +
                            '-' +
                            months +
                            '-' +
                            day for day in days]
                        time_convert = res[0] + '~' + res[1]
                    if '年' in i and '月' in i and ('日' in i or '号' in i):
                        months = re.search(r'(\d{1,2})月', i, re.S).group(1)
                        days = re.search(r'(\d{1,2})[月|日]', i, re.S).group(1)
                        if int(months) < 10 and len(months) == 1:
                            months = '0' + months
                        else:
                            months = months
                        if int(days) < 10 and len(days) == 1:
                            days = '0' + days
                        else:
                            days = days
                        years = str(datetime.datetime.now().year)
                        if i == '今年':
                            years = years
                        if i == '明年':
                            years = str(datetime.datetime.now().year + 1)
                        if i == '前年':
                            str(datetime.datetime.now().year - 2)
                        if i == '后年':
                            str(datetime.datetime.now().year + 2)
                        if i == '去年':
                            str(datetime.datetime.now().year - 1)
                        time_convert = years + '-' + months + '-' + days
                    if '年' not in i and '月' in i and ('日' in i or '号' in i):
                        months = datetime.datetime.now().month
                        days = re.search(r'(\d{1,2})[日|号]', i, re.S).group(1)
                        if int(days) < 10 and len(days) == 1:
                            days = '0' + days
                        else:
                            days = days
                        years = str(datetime.datetime.now().year)
                        if '上' in i:
                            months = str(months - 1)
                            if int(months) < 10 and len(months) == 1:
                                months = '0' + months
                            else:
                                months = months
                        if '本' in i or '这' in i:
                            if int(months) < 10 and len(months) == 1:
                                months = '0' + months
                            else:
                                months = months
                        if '下' in i:
                            months = str(months + 1)
                            if int(months) < 10 and len(months) == 1:
                                months = '0' + months
                            else:
                                months = months
                        time_convert = years + '-' + months + '-' + days
                    if '年' in i and '季' in i:
                        if self._has_numbers(i.split('年')[0]) is True:
                            years = re.search(
                                r'(\d{2,4})[年]', i, re.S).group(1)
                            if len(years) == 4:
                                years = years
                            else:
                                years = str(2000 + int(years))
                        else:
                            if '今年' in i:
                                years = str(datetime.datetime.now().year)
                            if '明年' in i:
                                years = str(datetime.datetime.now().year + 1)
                            if '去年' in i:
                                years = str(datetime.datetime.now().year - 1)
                            if '后年' in i:
                                years = str(datetime.datetime.now().year + 2)
                            if '前年' in i:
                                years = str(datetime.datetime.now().year - 2)

                        num = int(re.search(r'(\d)[季]', i, re.S).group(1))
                        if num == 1:
                            months = ['01', '03']
                        if num == 2:
                            months = ['04', '06']
                        if num == 3:
                            months = ['07', '09']
                        if num == 4:
                            months = ['10', '12']
                        month_range = calendar.monthrange(
                            int(years), int(months[1]))
                        time_convert = years + '-' + \
                            months[0] + '-01' + '~' + years + '-' + months[1] + '-' + str(month_range[1])
                    if '年' not in i and '季' in i:
                        num = int(re.search(r'(\d)[季]', i, re.S).group(1))
                        if num == 1:
                            months = ['01', '02', '03']
                        if num == 2:
                            months = ['04', '05', '06']
                        if num == 3:
                            months = ['07', '08', '09']
                        if num == 4:
                            months = ['10', '11', '12']
                        years = str(datetime.datetime.now().year)
                        res = [years + '-' + x for x in months]
                        month_range = calendar.monthrange(
                            int(years), int(months[2]))
                        days = [str(month_range[1])]
                        time_convert = res[0] + '-01' + \
                            '~' + res[2] + '-' + days[0]
                    if ('周' in i or '星期' in i or '礼拜' in i) and self._has_numbers(i[-1]) is True:
                        if '上' in i:
                            week_day = int(
                                datetime.datetime.now().isoweekday())
                            find_week = int(i[-1])
                            d_day = - week_day - (7 - find_week)
                            time_convert = (
                                datetime.datetime.now() +
                                datetime.timedelta(
                                    days=d_day)).strftime('%Y-%m-%d')
                        if '本' in i or '这' in i:
                            week_day = int(
                                datetime.datetime.now().isoweekday())
                            find_week = int(i[-1])
                            d_day = find_week - week_day
                            time_convert = (
                                datetime.datetime.now() +
                                datetime.timedelta(
                                    days=d_day)).strftime('%Y-%m-%d')
                        if '下' in i:
                            week_day = int(
                                datetime.datetime.now().isoweekday())
                            find_week = int(i[-1])
                            d_day = (7 - week_day) + find_week
                            time_convert = (
                                datetime.datetime.now() +
                                datetime.timedelta(
                                    days=d_day)).strftime('%Y-%m-%d')
                    if '过去' in i or '最近' in i:
                        years = str(datetime.datetime.now().year)
                        months = datetime.datetime.now().month
                        if months < 10 and len(str(months)) == 1:
                            months = '0' + str(months)
                        else:
                            months = months
                        if '月' in i:
                            num = re.search(
                                r'(\d{1,2})月', i.replace(
                                    '个', ''), re.S).group(1)
                            month_start = (datetime.datetime.now() + datetime.timedelta(days=-int(num)*30)).month
                            month_end = (datetime.datetime.now() + datetime.timedelta(days=-30)).month
                            year_start = (datetime.datetime.now() + datetime.timedelta(days=-int(num)*30)).year
                            year_end = (datetime.datetime.now() + datetime.timedelta(days=-30)).year
                            if month_start < 10 and len(str(month_start)) == 1:
                                month_start = '0' + str(month_start)
                            else:
                                month_start = month_start
                            if month_end < 10 and len(str(month_end)) == 1:
                                month_end = '0' + str(month_end)
                            else:
                                month_end = month_end
                            month_range = calendar.monthrange(
                                int(year_end), int(month_end))
                            time_convert = str(year_start) + '-' + str(month_start) + '-01' + '~' + \
                                str(year_end) + '-' + str(month_end) + '-' + str(month_range[1])
                        if '日' in i or '天' in i:
                            num = re.search(
                                r'(\d{1,2})[日|天]', i, re.S).group(1)
                            start_day = (datetime.datetime.now() + datetime.timedelta(days=-int(num))).day
                            end_day = (datetime.datetime.now() + datetime.timedelta(days=-1)).day
                            start_month = (datetime.datetime.now() + datetime.timedelta(days=-int(num))).month
                            end_month = (datetime.datetime.now() + datetime.timedelta(days=-1)).month
                            time_convert = str(years) + '-' + str(start_month) + '-' + str(
                                start_day) + '~' + str(years) + '-' + str(end_month) + '-' + str(end_day)
                    if '最近' in i and '年' in i:
                        num = re.search(r'(\d{1,2})年', i, re.S).group(1)
                        start_years = str(
                            datetime.datetime.now().year - int(num))
                        end_years = str(datetime.datetime.now().year)
                        month_range = calendar.monthrange(int(years), 12)
                        time_convert = start_years + '-' + '01-01' + '~' + \
                            end_years + '-' + '12' + '-' + str(month_range[1])
                    if '未来' in i:
                        years = str(datetime.datetime.now().year)
                        months = datetime.datetime.now().month
                        if months < 10 and len(str(months)) == 1:
                            months = '0' + str(months)
                        else:
                            months = months
                        if '月' in i:
                            num = re.search(
                                r'(\d{1,2})月', i.replace(
                                    '个', ''), re.S).group(1)
                            month_start = (datetime.datetime.now() + datetime.timedelta(days=30)).month
                            month_end = (datetime.datetime.now() + datetime.timedelta(days=int(num)*30)).month
                            year_start = (datetime.datetime.now() + datetime.timedelta(days=30)).year
                            year_end = (datetime.datetime.now() + datetime.timedelta(days=int(num) * 30)).year
                            if month_start < 10 and len(str(month_start)) == 1:
                                month_start = '0' + str(month_start)
                            else:
                                month_start = month_start
                            if month_end < 10 and len(str(month_end)) == 1:
                                month_end = '0' + str(month_end)
                            else:
                                month_end = month_end
                            month_range = calendar.monthrange(
                                int(year_end), int(month_end))
                            time_convert = str(year_start) + '-' + str(month_start) + '-01' + '~' + str(
                                year_end) + '-' + str(month_end) + '-' + str(month_range[1])
                        if '日' in i or '天' in i:
                            num = re.search(
                                r'(\d{1,2})[日|天]', i, re.S).group(1)
                            day_end = (datetime.datetime.now() + datetime.timedelta(days=int(num))).day
                            day_start = (datetime.datetime.now() + datetime.timedelta(days=1)).day
                            month_end = (datetime.datetime.now() + datetime.timedelta(days=int(num))).month
                            month_start = (datetime.datetime.now() + datetime.timedelta(days=1)).month
                            time_convert = str(years) + '-' + str(month_start) + '-' + str(
                                day_start) + '~' + str(years) + '-' + str(month_end) + '-' + str(day_end)
                    if ('上' in i or '前' in i) and '号' not in i and '日' not in i and self._has_numbers(i[-1]) is False:
                        if '年' in i:
                            num = re.search(r'(\d{1,2})年', i, re.S).group(1)
                            years = (datetime.datetime.now().year - int(num))
                            time_convert = str(years) + '-01-01' + '~' + str(years) + '-12-31'
                        if '月' in i:
                            num = re.search(r'(\d{1,2})月', i, re.S).group(1)
                            years = datetime.datetime.now().year
                            months = (datetime.datetime.now() + datetime.timedelta(days=-30 * int(num)))\
                                .strftime('%Y-%m-%d')
                            month_range = calendar.monthrange(int(years), int(months))
                            time_convert = str(years) + '-' + str(months) + '-01' + '~' + \
                                           str(years) + '-' + str(months) + '-' + str(month_range[1])
                        if '周' in i:
                            num = re.search(r'(\d{1,2})周', i, re.S).group(1)
                            start_time = (datetime.datetime.now() + datetime.timedelta(days= -7 * int(num))) \
                                .strftime('%Y-%m-%d')
                            end_time = (datetime.datetime.now()).strftime('%Y-%m-%d')
                            time_convert = start_time + '~' + end_time
                        if '天' in i:
                            num = re.search(r'(\d{1,2})天', i, re.S).group(1)
                            start_time = (datetime.datetime.now() +
                                          datetime.timedelta(days=-int(num))).strftime('%Y-%m-%d')
                            end_time = (datetime.datetime.now() +
                                        datetime.timedelta(days=-1)).strftime('%Y-%m-%d')
                            time_convert = start_time + '~' + end_time
                    if '下' in i or '后' in i:
                        if '年' in i:
                            num = re.search(r'(\d{1,2})年', i, re.S).group(1)
                            years = (datetime.datetime.now().year + int(num))
                            time_convert = str(years) + '-01-01' + '~' + str(years) + '-12-31'
                        if '月' in i:
                            num = re.search(r'(\d{1,2})月', i, re.S).group(1)
                            years = datetime.datetime.now().year
                            months = (datetime.datetime.now() + datetime.timedelta(days=30 * int(num))) \
                                .strftime('%Y-%m-%d')
                            month_range = calendar.monthrange(int(years), int(months))
                            time_convert = str(years) + '-' + str(months) + '-01' + '~' + \
                                    str(years) + '-' + str(months) + '-' + str(month_range[1])
                        if '周' in i:
                            num = re.search(r'(\d{1,2})周', i, re.S).group(1)
                            end_time = (datetime.datetime.now() + datetime.timedelta(days=7 * int(num))) \
                                .strftime('%Y-%m-%d')
                            start_time = (datetime.datetime.now()).strftime('%Y-%m-%d')
                            time_convert = start_time + '~' + end_time
                        if '天' in i:
                            num = re.search(r'(\d{1,2})天', i, re.S).group(1)
                            end_time = (datetime.datetime.now() +
                                datetime.timedelta(days=int(num))).strftime('%Y-%m-%d')
                            start_time = (datetime.datetime.now() +
                                datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                            time_convert = start_time + '~' + end_time
                    convert_result.append(time_convert)
            return convert_result
        else:
            return None

    def _infer_time_anglicize(self, context):
        """
        口语型时间映射
        :param context:
        :return: <list of time entity>
        """
        result = self.extract(context)
        if result is not None:
            convert_result2 = []
            for i in result:
                i = self._num_transform(i)
                time_convert = ''
                # if self._has_numbers(i) is False and (
                #         '周' in i or '星期' in i or '礼拜' in i or '季' in i or
                #         '年' in i or '天' in i or '月' in i or '最新' in i):
                if self._has_numbers(i) is False and any(map(lambda x, y:False if x in y else True,
                                                             ["周", "星期", "礼拜", "季", "年", "天", "月", "最新", "最近", "近期"], i)):
                    '''天的转换'''
                    if i == '今天':
                        time_convert = (
                            datetime.datetime.now()).strftime('%Y-%m-%d')
                    if i == '明天':
                        time_convert = (
                            datetime.datetime.now() +
                            datetime.timedelta(
                                days=1)).strftime('%Y-%m-%d')
                    if i == '昨天':
                        time_convert = (
                            datetime.datetime.now() +
                            datetime.timedelta(
                                days=-
                                1)).strftime('%Y-%m-%d')
                    if i == '后天':
                        time_convert = (
                            datetime.datetime.now() +
                            datetime.timedelta(
                                days=2)).strftime('%Y-%m-%d')
                    if i == '前天':
                        time_convert = (
                            datetime.datetime.now() +
                            datetime.timedelta(
                                days=-
                                2)).strftime('%Y-%m-%d')
                    if i == '现在':
                        time_convert = (
                            datetime.datetime.now()).strftime('%Y-%m-%d')
                    if '最新' in i or '最近' in i or '近期' in i:
                        years = str(datetime.datetime.now().year)
                        start_months = datetime.datetime.now().month-1
                        end_months = datetime.datetime.now().month
                        if start_months < 10 and len(str(start_months)):
                            start_months = '0'+str(start_months)
                        else:
                            start_months = start_months
                        if end_months < 10 and len(str(end_months)):
                            end_months = '0' + str(end_months)
                        else:
                            end_months = start_months
                        days = datetime.datetime.now().day
                        if days < 10 and len(str(days)):
                            days = '0' + str(days)
                        else:
                            days = days
                        time_convert = years + '-' + start_months + \
                            '-01' + '~' + years + '-' + end_months + '-' + str(days)
                    '''年的转换'''
                    if i == '今年':
                        month_range = calendar.monthrange(
                            datetime.datetime.now().year, 12)
                        time_convert = str(datetime.datetime.now().year) + '-01-01' + '~' + str(
                            datetime.datetime.now().year) + '-12' + '-' + str(month_range[1])
                    if i == '明年':
                        month_range = calendar.monthrange(
                            datetime.datetime.now().year + 1, 12)
                        time_convert = str(datetime.datetime.now().year + 1) + '-01-01' + '~' + str(
                            datetime.datetime.now().year + 1) + '-12' + '-' + str(month_range[1])
                    if i == '前年':
                        month_range = calendar.monthrange(
                            datetime.datetime.now().year - 2, 12)
                        time_convert = str(datetime.datetime.now().year - 2) + '-01-01' + '~' + str(
                            datetime.datetime.now().year - 2) + '-12' + '-' + str(month_range[1])
                    if i == '后年':
                        month_range = calendar.monthrange(
                            datetime.datetime.now().year + 2, 12)
                        time_convert = str(datetime.datetime.now().year + 2) + '-01-01' + '~' + str(
                            datetime.datetime.now().year + 2) + '-12' + '-' + str(month_range[1])
                    if i == '去年':
                        month_range = calendar.monthrange(
                            datetime.datetime.now().year - 1, 12)
                        time_convert = str(datetime.datetime.now().year - 1) + '-01-01' + '~' + str(
                            datetime.datetime.now().year - 1) + '-12' + '-' + str(month_range[1])

                    '''组合词汇'''
                    if ('去年' in i or '明年' in i) and '今天' in i:
                        months = str(datetime.datetime.now().month)
                        days = str(datetime.datetime.now().day)
                        if '去年' in i:
                            years = str(datetime.datetime.now().year - 1)
                            time_convert = years + '-' + months + '-' + days
                        if '明年' in i:
                            years = str(datetime.datetime.now().year + 1)
                            time_convert = years + '-' + months + '-' + days

                    '''月的转换'''
                    if '月' in i:
                        years = str(datetime.datetime.now().year)
                        if '上' in i:
                            months = (
                                datetime.datetime.now() +
                                datetime.timedelta(
                                    days=-
                                    30)).month
                            if months < 10 and len(str(months)) == 1:
                                months = '0' + str(months)
                            else:
                                months = months
                            month_range = calendar.monthrange(
                                int(years), int(months))
                            days = ['01', str(month_range[1])]
                            res = [
                                years +
                                '-' +
                                str(months) +
                                '-' +
                                day for day in days]
                            time_convert = res[0] + '~' + res[1]
                        if '本' in i or '这' in i:
                            months = datetime.datetime.now().month
                            if months < 10 and len(str(months)) == 1:
                                months = '0' + str(months)
                            else:
                                months = months
                            month_range = calendar.monthrange(
                                int(years), int(months))
                            days = ['01', str(month_range[1])]
                            res = [
                                years +
                                '-' +
                                str(months) +
                                '-' +
                                day for day in days]
                            time_convert = res[0] + '~' + res[1]
                        if '下' in i:
                            months = (
                                datetime.datetime.now() +
                                datetime.timedelta(
                                    days=30)).month
                            if months < 10 and len(str(months)) == 1:
                                months = '0' + str(months)
                            else:
                                months = months
                            month_range = calendar.monthrange(
                                int(years), int(months))
                            days = ['01', str(month_range[1])]
                            res = [
                                years +
                                '-' +
                                str(months) +
                                '-' +
                                day for day in days]
                            time_convert = res[0] + '~' + res[1]

                    '''周的转换'''
                    if '周' in i or '星期' in i or '礼拜' in i:
                        if "本" in i or "这" in i:
                            week_day = int(
                                datetime.datetime.now().isoweekday())
                            start_time = (
                                datetime.datetime.now() +
                                datetime.timedelta(
                                    days=-
                                    week_day +
                                    1)).strftime('%Y-%m-%d')
                            end_time = (
                                datetime.datetime.strptime(
                                    start_time,
                                    '%Y-%m-%d') +
                                datetime.timedelta(
                                    days=6)).strftime('%Y-%m-%d')
                            time_convert = start_time + '~' + end_time
                        if '上' in i:
                            week_day = int(
                                datetime.datetime.now().isoweekday())
                            start_time = (
                                datetime.datetime.now() +
                                datetime.timedelta(
                                    days=-
                                    week_day)).strftime('%Y-%m-%d')
                            end_time = (
                                datetime.datetime.strptime(
                                    start_time,
                                    '%Y-%m-%d') +
                                datetime.timedelta(
                                    days=-
                                    6)).strftime('%Y-%m-%d')
                            time_convert = end_time + '~' + start_time
                        if '下' in i:
                            week_day = int(datetime.datetime.now().weekday())
                            d_day = 7 - week_day
                            start_time = (
                                datetime.datetime.now() +
                                datetime.timedelta(
                                    days=d_day +
                                    1)).strftime('%Y-%m-%d')
                            end_time = (
                                datetime.datetime.strptime(
                                    start_time,
                                    '%Y-%m-%d') +
                                datetime.timedelta(
                                    days=6)).strftime('%Y-%m-%d')
                            time_convert = start_time + '~' + end_time
                    '''季度的转换'''
                    if '季度' in i:
                        if self._has_numbers(i) is False:
                            if '本' in i or '这' in i:
                                years = str(datetime.datetime.now().year)
                                month_now = datetime.datetime.now().month
                                mod_cal = month_now % 3
                                if mod_cal == 0:
                                    res1 = [str(month_now - i)
                                            for i in range(3)]
                                    res2 = [
                                        '0' + x for x in res1 if len(x) == 1]
                                    res3 = [years + '-' + x for x in res2]
                                    month_range = calendar.monthrange(
                                        int(years), int(month_now))
                                    days = [str(month_range[1])]
                                    time_convert = res3[2] + '-01' + \
                                        '~' + res3[0] + '-' + days[0]
                                if mod_cal == 1:
                                    res1 = [str(month_now + i)
                                            for i in range(3)]
                                    res2 = [
                                        '0' + x for x in res1 if len(x) == 1]
                                    res3 = [years + '-' + x for x in res2]
                                    month_range = calendar.monthrange(
                                        int(years), int(res1[2]))
                                    days = [str(month_range[1])]
                                    time_convert = res3[0] + '-01' + \
                                        '~' + res3[2] + '-' + days[0]
                                if mod_cal == 2:
                                    res1 = [
                                        str(month_now - 1), str(month_now), str(month_now + 1)]
                                    res2 = [years + '-' + x for x in res1]
                                    month_range = calendar.monthrange(
                                        int(years), int(month_now + 1))
                                    days = str(month_range[1])
                                    time_convert = res2[0] + '-01' + \
                                        '~' + res2[2] + '-' + days
                            if '上' in i:
                                years = str(datetime.datetime.now().year)
                                month_now = datetime.datetime.now().month
                                if month_now > 3:
                                    if int(month_now / 3) <= 1:
                                        time_convert = years + '-01-01' + '~' + years + '-03-31'
                                    if 1 < int(month_now / 3) <= 2:
                                        time_convert = years + '-04-01' + '~' + years + '-06-30'
                                    if 2 < int(month_now / 3) <= 3:
                                        time_convert = years + '-07-01' + '~' + years + '-09-30'
                                else:
                                    print('当前时间为本年度第一季度')
                            if '下' in i:
                                years = str(datetime.datetime.now().year)
                                month_now = datetime.datetime.now().month
                                if month_now > 3:
                                    if int(month_now / 3) <= 1:
                                        time_convert = years + '-04-01' + '~' + years + '-06-30'
                                    if 1 < int(month_now / 3) <= 2:
                                        time_convert = years + '-07-01' + '~' + years + '-09-30'
                                    if 2 < int(month_now / 3) <= 3:
                                        time_convert = years + '-10-01' + '~' + years + '-12-31'
                                else:
                                    print('当前时间为本年度第四季度')
                    convert_result2.append(time_convert)
            return convert_result2
        else:
            return None

    def _infer_time_contact(self, context):
        """判断时间实体之间的关系，确定输出结果为时间区间还是时间点"""
        result = self.extract(context)
        standard = self._infer_time_standard(context)
        mixed = self._infer_time_mixed(context)
        anglicize = self._infer_time_anglicize(context)
        convert_result4 = []
        time_convert = ''
        if result is not None:
            if len(result) == 2 and (
                    len(standard) == 2 or len(mixed) == 2 or len(anglicize) == 2):
                if len(standard) == 2:
                    standard = standard
                if len(mixed) == 2:
                    standard = mixed
                if len(anglicize) == 2:
                    standard = anglicize
                patterns = (result[0] + '(.*?)' + result[1])
                link = re.search(patterns, context['query'], re.S).group(1)
                if '到' in link or '至' in link or link == '~' or link == '-':
                    if result[1][0:4].isdigit() is False and '今年' not in result[1] and '去年' not in result[1] and '明年' \
                            not in result[1] and '后年' not in result[1] and '前年' not in result[1]:
                        years = result[0][0:4]
                        months_start = standard[0].split('-')[1]
                        days = standard[1].split('-')[2]
                        if '~' in standard[0] and '~' not in standard[1]:
                            months_end = standard[1].split('-')[1]
                            time_convert = years + '-' + months_start + '-01' + \
                                '~' + years + '-' + months_end + '-' + days
                        elif '~' in standard[0] and '~' in standard[1]:
                            months_end = standard[1].split(
                                '~')[1].split('-')[1]
                            month_range = calendar.monthrange(
                                int(years), int(months_end[1]))
                            time_convert = years + '-' + months_start + '-01' + '~' + \
                                years + '-' + months_end + '-' + str(month_range[1])
                        elif '~' not in standard[0] and '~' not in standard[1]:
                            time_convert = standard[0] + '~' + standard[1]
                        elif '~' not in standard[0] and '~' in standard[1]:
                            months_end = standard[1].split(
                                '~')[1].split('-')[1]
                            month_range = calendar.monthrange(
                                int(years), int(months_end[1]))
                            time_convert = standard[0] + '~' + years + \
                                '-' + months_end + '-' + str(month_range[1])
                    else:
                        if '~' in standard[0]:
                            start_time = standard[0].split('~')[0]
                        else:
                            start_time = standard[0]
                        if '~' in standard[1]:
                            end_time = standard[1].split('~')[1]
                        else:
                            end_time = standard[1]
                        time_convert = start_time + '~' + end_time
                    convert_result4.append(time_convert)
            return convert_result4
        else:
            return None



if __name__ == "__main__":
    '''测试test文件'''
    for line in open("test.txt", 'r'):
        contexts = dict()
        contexts['query'] = line.split(' ')[0]
        print(contexts)
        a = TimeNer()
        b = a.extract(contexts)
        print(b)
        d = a.transform(contexts)
        print(d, '\n')

    # '''单独调试'''
    # contexts = {'query': '帮我查一下一周前的用电量'}
    # a = NerRuleV1()
    # b = a.extract(contexts)
    # print(b)
    # c = a.transform(contexts)
    # print(c)
    # d = a._extract_time(contexts)
    # print(d)
    # d = a._infer_time_mixed(contexts)
    # print(d)
    # e = a._infer_time_anglicize(contexts)
    # print(e)
    # f = a._infer_time_standard(contexts)
    # print(f)
    # g = a._infer_time_entity(contexts)
    # print (g)
