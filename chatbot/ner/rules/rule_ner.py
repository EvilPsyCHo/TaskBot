# coding:utf8
# @Time    : 18-6-7 上午10:51
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from chatbot.core.entity import TimeInterval, Location, Company, Tag
from chatbot.ner.rules.company_ner import CompanyNer
from chatbot.ner.rules.time_ner import TimeNer
from chatbot.ner.rules.location_ner import LocationNer
from chatbot.ner.rules.tag_ner import TagNer
from chatbot.utils.log import get_logger


logger = get_logger("NER")


class NerRuleV1:
    def __init__(self):
        super().__init__()
        self.ner_company = CompanyNer()
        self.ner_loc = LocationNer()
        self.ner_time = TimeNer()
        self.ner_tag = TagNer()

    def extract(self, context):
        """
        :param context: context["query"]
        :return: <dict of list>
        {"TimeInterval": ["", ""]}

        """
        rst = {}
        ext_time = self.ner_time.extract(context)
        ext_location = self.ner_loc.extract(context)
        ext_company = self.ner_company.extract(context)
        ext_tag = self.ner_tag.extract(context)
        if ext_time is not None:
            rst[TimeInterval.name()] = ext_time
            logger.debug("extract time %s" % " ".join(ext_time))
        if ext_location is not None:
            rst[Location.name()] = ext_location
            logger.debug("extract loc {}".format(ext_location))
        if ext_company is not None:
            rst[Company.name()] = ext_company
            logger.debug("extract company %s" % " ".join(ext_company))
        if ext_tag is not None:
            rst[Tag.name()] = ext_tag
            # logger.debug("extract tag %s" % " ".join(ext_tag))
        return rst

    def transform(self, context):
        """
        :param context: context['query']
        :return: dict of list
        [{'end':'2018-06-13','start':'2018-06-13'}]
        """
        rst = {}
        trans_time = self.ner_time.transform(context)
        trans_location = self.ner_loc.transform(context)
        trans_company = self.ner_company.transform(context)
        trans_tag = self.ner_tag.transform(context)
        if trans_time is not None:
            rst[TimeInterval.name()] = trans_time
        if trans_location is not None:
            rst[Location.name()] = trans_location
        if trans_company is not None:
            rst[Company.name()] = trans_company
        if trans_tag is not None:
            rst[Tag.name()] = trans_tag
        return rst


if __name__ == "__main__":
    '''测试test文件'''
    for line in open("test.txt", 'r'):
        contexts = dict()
        contexts['query'] = line.split(' ')[0]
        print(contexts)
        a = NerRuleV1()
        b = a.extract(contexts)
        contexts["entities"] = b
        print(b)
        d = a.transform(contexts)
        print(d)
