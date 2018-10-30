# coding:utf8
# @Time    : 18-6-25 下午2:41
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import sys
sys.path.append("/home/zhouzr/project/Task-Oriented-Chatbot")
from chatbot.utils.log import get_logger
logger = get_logger("test")
from chatbot.intent.models.fast_text import FastText
from chatbot.intent.rules.rule_v1 import IntentRuleV1
from chatbot.ner.rules.rule_ner import NerRuleV1
from chatbot.cparse.label import IntentLabel
from chatbot.cparse.vocabulary import Vocabulary
from chatbot.skills.simple import SayHi, Thanks, Praise, Criticize, GoodBye, LeaveMessage, CompanyInfo, BusinessInfo
from chatbot.skills.botQA import BotQA
from chatbot.skills.data_query import TestDataQuery, DataInquiry
from chatbot.skills.help import Help
from chatbot.skills.safe import SafeResponse
from chatbot.skills.tuling import Tuling
from chatbot.skills.file_retrieval import FileRetrievalExt
from chatbot.bot import ChatBot
from chatbot.utils.path import MODEL_PATH
from wxpy import Bot
from flask import Flask,request
# from WXBizMsgCrypt import WXBizMsgCrypt
from chatbot.applications.WXBizMsgCrypt_py3 import WXBizMsgCrypt
import xml.etree.cElementTree as ET
import requests, json


intent_model = FastText.load(str(MODEL_PATH/"v0.21"/"intent_model.FastText"))
intent_rule = IntentRuleV1()
ner = NerRuleV1()
label = IntentLabel.load(str(MODEL_PATH/"v0.21"/"label"))
vocab = Vocabulary.load(str(MODEL_PATH/"v0.21"/"vocab"))
file_retrieval = FileRetrievalExt(str(MODEL_PATH/"v0.2"/"file_retrieval"/"test"),
                          str(MODEL_PATH/"v0.2"/"file_retrieval"/"policy_filev3.utf8.csv"))

cbot = ChatBot(
    intent_model=intent_model,
    intent_rule=intent_rule,
    vocab=vocab,
    label=label,
    ner=ner,
    intent2skills={
        "未知": SafeResponse(),
        "留言": LeaveMessage(),
        "帮助": Help(),
        "闲聊": Tuling(),
        "文件检索": file_retrieval,
        "数据查询": DataInquiry(),
        "公司咨询": CompanyInfo(),
        "业务咨询": BusinessInfo(),
        "chatbot": BotQA(),
        "表扬": Praise(),
        "批评": Criticize(),
        "打招呼": SayHi(),
        "再见": GoodBye(),
        "感谢": Thanks(),
    }
)


def wxchat(request_content, fromusername):
    '''

    :param request_content: <string> 用户发送的内容
    :param fromusername: <string> 用户名
    :return: <string> 对用户的回复
    '''
    # if request_content.startswith("小益"):
    if request_content !=None:
        query={
            "text": request_content.lower(),
            "user": fromusername,
            "right": [],
            "app": "wechat",
        }
        return cbot.chat(query)
    else:
        pass




app = Flask(__name__)
@app.route('/',methods=['GET','POST'])
def wechat_response():
    '''

    :return: <xml> 服务器对用户请求的响应
    '''
    #公众号信息
    sToken = 'BvuTkJCtloUGZreX62fTOQ'
    sEncodingAESKey = 'vT0ZUOoAARa0fdfDxCbku6MSE7zXIZUq1RwyGcD45mT'
    sCorpID = 'wxca34a8f715b7ff2b'
    AppSecret="73fc2156e86e5075fe33a312f03a8426"

    #获取access_token，有效期2小时，用于获得微信授权，以提取用户基本信息
    url_access_token="https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid=%s&secret=%s"% (sCorpID ,AppSecret)
    print("url_access_token:",url_access_token)
    ans_access_token=requests.get(url_access_token)
    access_token=json.loads(ans_access_token.content)['access_token']
    # print("access_token:",access_token)
    wxcpt=WXBizMsgCrypt(sToken,sEncodingAESKey,sCorpID)
    #获取url验证时微信发送的相关参数
    #get方法
    sVerifyMsgSig=request.args.get("signature")
    sVerifyTimeStamp=request.args.get('timestamp')
    sVerifyNonce=request.args.get('nonce')
    sVerifyEchoStr=request.args.get('echostr')

    #验证url
    if request.method == 'GET':
        return sVerifyEchoStr
    #接收客户端消息
    if request.method == 'POST':
        sReqData = request.data
        print(request.args)
        print("request.data:",request.data)

        #解析接收的内容
        xml_tree = ET.fromstring(sReqData)
        ToUserName=xml_tree.find("ToUserName").text
        FromUserName=xml_tree.find("FromUserName").text
        CreateTime=xml_tree.find("CreateTime").text
        MsgType_request=xml_tree.find("MsgType").text
        MsgId=xml_tree.find("MsgId").text
        MsgType_response="text"
        #TODO 获取用户昵称，前提是已通过微信认证
        # openid = FromUserName
        # url_openid = "https://api.weixin.qq.com/cgi-bin/user/info?access_token=%s&openid=%s&lang=zh_CN"% (access_token, openid)
        #
        # print("url_openid:",url_openid)
        # ans_openid = requests.get(url_openid)
        # user_nickname = json.loads(ans_openid.content)['nickname']
        user_nickname=""

        #对用户的文本请求进行回复
        if MsgType_request=="text":
            content = xml_tree.find("Content").text
            print("content:",content)
            # 被动响应消息，智能回复
            response_text = wxchat(content, FromUserName)
            print("response_text:", response_text)
            # <xml> 响应消息
            response = '''<xml><ToUserName><![CDATA[''' + FromUserName + ''']]></ToUserName>\n
                    <FromUserName><![CDATA[''' + ToUserName + ''']]></FromUserName>\n
                    <CreateTime>''' + CreateTime + '''</CreateTime>\n
                    <MsgType><![CDATA[''' + MsgType_response + ''']]></MsgType>\n
                    <Content><![CDATA[''' + response_text + user_nickname + ''']]></Content>\n
                    <MsgId>''' + MsgId + '''</MsgId>\n</xml>'''

        # 对用户的语音请求进行回复
        # 前提是已开启微信的获取语音识别结果
        elif MsgType_request=="voice":
            # 语音识别结果
            Recognition=xml_tree.find("Recognition").text
            print("Recognition:",Recognition)
            # 被动响应消息，智能回复
            response_text = wxchat(Recognition, FromUserName)
            print("response_text:", response_text)
            # <xml> 响应消息
            response = '''<xml><ToUserName><![CDATA[''' + FromUserName + ''']]></ToUserName>\n
                    <FromUserName><![CDATA[''' + ToUserName + ''']]></FromUserName>\n
                    <CreateTime>''' + CreateTime + '''</CreateTime>\n
                    <MsgType><![CDATA[''' + MsgType_response + ''']]></MsgType>\n
                    <Content><![CDATA[''' + response_text + user_nickname + ''']]></Content>\n
                    <MsgId>''' + MsgId + '''</MsgId>\n</xml>'''
    return response


if __name__ == '__main__':
    app.run(host='172.16.12.110', port=80, debug=False)
    content="小益,hello！"
    FromUserName="abc123"
    response_text = wxchat(content, FromUserName)
    print("response_text:",response_text)