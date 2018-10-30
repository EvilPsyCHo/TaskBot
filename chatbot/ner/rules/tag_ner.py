# coding:utf8
# @Time    : 18-7-20 上午10:53
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from fuzzywuzzy import process
from chatbot.core.entity import Tag


TAGMAP = {
    "电量": "电量",
    "用电": "电量",
    "峰期电量": "峰平谷电量",
    "谷期电量": "峰平谷电量",
    "平期电量": "峰平谷电量",
    "峰电量": "峰平谷电量",
    "谷电量": "峰平谷电量",
    "平电量": "峰平谷电量",
    "峰期电费": "峰平谷电费",
    "谷期电费": "峰平谷电费",
    "平期电费": "峰平谷电费",
    "功率因数": "功率因数"
}


class TagNer(object):
    def __init__(self):
        self.keywords = TAGMAP.keys()
        self.threshold = 65

    def extract(self, context):
        entities = process.extract(context["query"], self.keywords)
        print(entities)
        entities = filter(lambda x: x[1] >= self.threshold, entities)
        entities = sorted(entities, key=lambda x: x[1] + len(x[0])/10, reverse=True)
        entities = list(map(lambda x: Tag(TAGMAP[x[0]]), entities))

        if len(entities) == 0:
            return None
        return entities[0]

    def transform(self, context):
        return self.extract(context)


if __name__ == "__main__":
    queries = [{"query":"今天用电，负载"},
               {"query":"峰平谷电费"},
               {"query":"1"}]
    ner = TagNer()
    print(ner.extract(queries[0]))
    print(ner.transform(queries[1]))
