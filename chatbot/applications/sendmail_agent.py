#coding:utf-8   #强制使用utf-8编码格式
import smtplib  #加载smtplib模块
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
from email.header import Header

#配置基本信息
sender = '416018233@qq.com'  # 发件人邮箱账号
sender_psw = 'fjsjxrkeugfhcbec'  # 发件人邮箱授权码
touser = 'ye_shaohui@163.com'  # 收件人邮箱账号

def mail_mainbody():
    ''' 生成邮件正文内容

    :return: MIMEMultipart()实例
    '''

    message = MIMEMultipart()
    message['From'] = formataddr(["万益能源", sender])  # 发送方昵称
    message['To'] = formataddr(["来福士", touser])  # 接收方昵称
    message['Subject'] = '这是一封提醒11:36'  # 邮件主题
    content='请问您对我们的服务满意吗？'
    message.attach(MIMEText(content, 'plain', 'utf-8'))  # 邮件正文内容
    return message

def with_picture(message):

    return message

def with_attachment(message):
    ''' 添加附件到MIMEMultipart()实例

    :param message: message对象
    :return: 添加了附件的MIMEMultipart()实例
    '''
    # 生成两个txt文本附件
    f = open("./chat/file1.txt", "w")
    f.write("文件1中的内容")
    f.close()

    f = open("./chat/file2.txt", "w")
    f.write("file2")
    f.close()

    # 构造附件1，传送当前目录下的 file1.txt 文件
    file_path = r"C:\Users\yesh\PycharmProjects\yecode_tools\chat"
    att1 = MIMEText(open(file_path + "/file1.txt", 'rb').read(), 'base64', 'utf-8')
    att1["Content-Type"] = 'application/octet-stream'
    # 这里的filename可以任意写，写什么名字，邮件中显示什么名字
    att1["Content-Disposition"] = 'attachment; filename="file1.txt"'
    message.attach(att1)

    # 构造附件2，传送当前目录下的 file2.txt 文件
    att2 = MIMEText(open(file_path + '/file2.txt', 'rb').read(), 'base64', 'utf-8')
    att2["Content-Type"] = 'application/octet-stream'
    att2["Content-Disposition"] = 'attachment; filename="file2.txt"'
    message.attach(att2)
    return message

def mail_content(picture=False,attachment=False):
    '''

    :param picture: <bool> 是否要在正文中添加图片
    :param attachment: <bool> 是否要添加附件
    :return: MIMEMultipart()实例
    '''
    message = mail_mainbody()
    if picture == True:
        pass
    if attachment==True:
        message = with_attachment(message)
    send_content=message
    return send_content

def mail():
    ret=True  # <bool> 标识符，标记是否出错
    try:
        msg=mail_content(attachment=True)
        # msg = mail_content()
        # server=smtplib.SMTP("smtp.qq.com",25)  #发件人邮箱中的SMTP服务器，根据使用的第三方 SMTP 服务器来修改域名和端口
        # server.login(sender,sender_psw)    # 登录邮箱，参数：发件人邮箱账号、邮箱密码
        server = smtplib.SMTP('localhost')
        server.sendmail(sender,[touser,],msg.as_string())   #发送邮件，参数：发件人邮箱、收件人邮箱、邮件内容
        server.quit()   #关闭连接
    except Exception:   #如果try中的语句没有执行，则会执行下面的ret=False
        ret = False
    return ret

if __name__ =="__main__":
    ret=mail()
    print("sendmail...")
    if ret:
        print("邮件发送成功") #发送成功
    else:
        print("ERROR！邮件发送失败")  #发送失败