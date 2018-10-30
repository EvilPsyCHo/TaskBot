# coding:utf8
# @Time    : 18-6-2 下午12:06
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import re
import copy
import sys
sys.path.append("/home/zhouzr/project/Task-Oriented-Chatbot")
from chatbot.ner.rules.rule_ner import NerRuleV1


def inser_ner_time_extract(name, query, extract_result):
    path = "/home/zhouzr/ner_time_extract.txt"
    with open(path, "a") as f:
        f.write(name)
        f.write("\t")
        f.write(query)
        f.write("\t")
        f.write(extract_result)
        f.write("\n")

def insert(text, intent, name):
    path = "/home/zhouzr/intent_corpus.txt"
    with open(path, 'a') as f:
        s = " ".join([name, intent, text, "\n"])
        f.write(s)


ALL_INTENT = ["打招呼","再见","肯定","否定","批评","表扬","感谢",
              "文件检索","用电查询", "留言","业务咨询","公司咨询"]


if __name__ == "__main__":
    from wxpy.utils.misc import enhance_connection
    import requests

    URL = "http://www.tuling123.com/openapi/api"
    APIKEY = "7e3bf5d6d06143e39c898672592d63ad"
    USERID = "test"
    LOC = "成都"
    sesstion = requests.session()
    enhance_connection(sesstion)


    def tuling(s):
        send = dict(
            key=APIKEY,
            userid=USERID,
            info=s,
            loc=LOC
        )

        r = sesstion.post(URL, send)
        answer = r.json()
        return answer["text"]

    from wxpy import *
    # 初始化机器人，扫码登陆
    bot = Bot()
    my_friend = bot.friends()
    my_groups = bot.groups()
    print(my_groups)
    ner = NerRuleV1()

    @bot.register(my_groups)
    def reply_my_group(msg):
        text = copy.deepcopy(msg.text)
        if text.startswith("小益"):
            text = re.sub("^小益[,.!！，。？ ]{0,3}", "", text)
            if text == "任务":
                return "测试时间提取模块，请输入类似这样的询问<小益 测试 上周用电量>\n" \
                       "今日需求意图样本：闲聊，文件检索，用电查询，公司咨询，业务咨询\n\n" \
                       "如有疑问，请输入：<小益，对应类别名称>进行询问\n\n" \
                       "添加对应类别样本，输入格式如下：\n\n<小益+空格+类别名称+空格+模拟句子>\n"
            elif text == "提示":
                return "目前，意图类别有：\n" \
                       "文件检索、用电查询、留言、业务咨询、公司咨询、打招呼、再见、肯定、否定、批评、表扬、感谢\n" \
                       "添加对应类别样本，输入格式如下：<小益+空格+类别名称+空格+模拟句子>\n"\
                        "模拟句子里请不要包含空格哦\n"\
                       "如有疑问，请输入：<小益，对应类别名称>进行询问"
            elif text == "文件检索":
                return "文件检索，指客户购售电、需求侧管理等相关电力文件的检索\n" \
                       "例子1：<小益 文件检索 今年四川的政策文件>\n" \
                       "例子2：<小益 文件检索 清洁能源>\n" \
                       "例子3：<小益 文件检索 跨省跨区、水电>\n"
            elif text == "用电查询":
                return "用电查询，指客户对自身项目用电情况的查询（一期只支持电量查询）\n" \
                       "例子1：<小益 用电查询 5月3号到今天来福士用电情况>\n" \
                       "例子2：<小益 用电查询 昨天用了多少电>\n" \
                       "例子3：<小益 用电查询 上周二到本周商场空调用电>\n"
            elif text == "公司咨询":
                return "公司咨询,指客户对万益或华宇，办公地点、联系方式、邮箱的咨询\n" \
                       "例子1：<小益 公司咨询 怎么联系你们？>\n" \
                       "例子2：<小益 公司咨询 你们在哪办公>"
            elif text == "业务咨询":
                return "业务咨询,指客户对万益或华宇的产品或服务咨询\n" \
                       "例子1：<小益 公司咨询 你们卖什么？>\n" \
                       "例子2：<小益 公司咨询 提供什么服务？>"
            elif text == "闲聊":
                return "除了其他意图外的所有问题\n" \
                       "例子1：<小益 闲聊 姚明有多高>\n" \
                       "例子2：<小益 闲聊 长城有多长>"

            elif text in ["打招呼","再见","肯定","否定","批评","表扬","感谢", "留言"]:
                return "当前意图暂不需要，请输入<小益，任务>查看最新需求"
            elif len(text.split(" ")) == 2:
                if text.split(" ")[0] in ALL_INTENT:
                    insert(text.split(" ")[0], "".join(text.split(" ")[1:]), msg.member.name)
                    return "谢谢{},成功添加一条意图识别样本".format(msg.member.name)
                elif text.split(" ")[0] == "测试":
                    time_ner = ner.extract({"query": text.split(" ")[1]})
                    inser_ner_time_extract(msg.member.name, text, str(time_ner))
                    return str(time_ner)
                else:
                    try:
                        tu = tuling(text)
                        return tu + "\n\n" + "闲聊固然有趣，但小益更希望你能帮我哦，请输入<小益，任务>进行查看"
                    except:
                        return "小益今天聊不动拉。。小益更希望你能帮我哦，请输入<小益，任务>进行查看"

            else:
                try:
                    tu = tuling(text)
                    return tu + "\n\n" + "闲聊固然有趣，但小益更希望你能帮我哦，请输入<小益，任务>进行查看"
                except:
                    return "小益今天聊不动拉。。小益更希望你能帮我哦，请输入<小益，任务>进行查看"

        else:
            pass

    bot.join()
