import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

sender = 'from@runoob.com'
receivers = ['416018233@qq.com']  # 接收邮件，可设置为你的QQ邮箱或者其他邮箱

# 创建一个带附件的实例
message = MIMEMultipart()
message['From'] = Header("菜鸟教程", 'utf-8')
message['To'] = Header("测试", 'utf-8')
subject = 'Python SMTP 邮件测试'
message['Subject'] = Header(subject, 'utf-8')

# 邮件正文内容
message.attach(MIMEText('这是菜鸟教程Python 邮件发送测试……', 'plain', 'utf-8'))


# f=open("test1.txt","w")
# f.write("test1")
# f.close()
#
# f=open("test2.txt","w")
# f.write("test2")
# f.close()
#
# # 构造附件1，传送当前目录下的 test.txt 文件
# file_path=r"C:\Users\yesh\PycharmProjects\yecode_tools\chat"
# att1 = MIMEText(open(file_path+"/test1.txt", 'rb').read(), 'base64', 'utf-8')
# att1["Content-Type"] = 'application/octet-stream'
# # 这里的filename可以任意写，写什么名字，邮件中显示什么名字
# att1["Content-Disposition"] = 'attachment; filename="test.txt"'
# message.attach(att1)
#
# # 构造附件2，传送当前目录下的 runoob.txt 文件
# att2 = MIMEText(open(file_path+'/test2.txt', 'rb').read(), 'base64', 'utf-8')
# att2["Content-Type"] = 'application/octet-stream'
# att2["Content-Disposition"] = 'attachment; filename="test2.txt"'
# message.attach(att2)

try:
    smtpObj = smtplib.SMTP('localhost')
    smtpObj.sendmail(sender, receivers, message.as_string())
    print("邮件发送成功")
except smtplib.SMTPException:
    print("Error: 无法发送邮件")