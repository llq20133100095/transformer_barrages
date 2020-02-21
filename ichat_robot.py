#coding=utf8
import itchat
from tuling import get_response
from barrages_generate import init_sess_model, gen_barrage_wechcat

barrages_flag = False

transformer = init_sess_model()

@itchat.msg_register('Text')
def text_reply(msg):
    global barrages_flag, transformer
    if barrages_flag and "退出" not in msg["Text"]:
        answer_usr = msg['FromUserName']
        itchat.send("生成中......", toUserName=answer_usr)
        itchat.send("生成词语：" + gen_barrage_wechcat(msg['Text'], transformer), toUserName=answer_usr)
        return "如果要退出则输入：退出"
    elif barrages_flag and "退出" in msg["Text"]:
        barrages_flag = False
        return "成功退出弹幕生成功能"
    elif u'作者' in msg['Text'] or u'主人' in msg['Text']:
        return u'你可以在这里了解他：https://github.com/llq20133100095'
    elif u'源代码' in msg['Text'] or u'获取文件' in msg['Text']:
        itchat.send('@fil@main.py', msg['FromUserName'])
        return u'这就是现在机器人后台的代码，是不是很简单呢？'
    elif u'获取图片' in msg['Text']:
        itchat.send('@img@applaud.gif', msg['FromUserName']) # there should be a picture
    elif '开启弹幕生成功能' == msg['Text'] and not barrages_flag:
        barrages_flag = True
        return "请输入要生成的词语"
    # else:
    #     return get_response(msg['Text']) or u'收到：' + msg['Text']


@itchat.msg_register(['Picture', 'Recording', 'Attachment', 'Video'])
def atta_reply(msg):
    return ({ 'Picture': u'图片', 'Recording': u'录音',
        'Attachment': u'附件', 'Video': u'视频', }.get(msg['Type']) +
        u'已下载到本地') # download function is: msg['Text'](msg['FileName'])


@itchat.msg_register(['Map', 'Card', 'Note', 'Sharing'])
def mm_reply(msg):
    if msg['Type'] == 'Map':
        return u'收到位置分享'
    elif msg['Type'] == 'Sharing':
        return u'收到分享' + msg['Text']
    elif msg['Type'] == 'Note':
        return u'收到：' + msg['Text']
    elif msg['Type'] == 'Card':
        return u'收到好友信息：' + msg['Text']['Alias']


@itchat.msg_register('Text', isGroupChat = True)
def group_reply(msg):
    if msg['isAt']:
        return u'@%s\u2005%s' % (msg['ActualNickName'],
            get_response(msg['Text']) or u'收到：' + msg['Text'])


@itchat.msg_register('Friends')
def add_friend(msg):
    itchat.add_friend(**msg['Text'])
    itchat.send_msg(u'项目主页：github.com/littlecodersh/ItChat\n'
        + u'源代码  ：回复源代码\n' + u'图片获取：回复获取图片\n'
        + u'欢迎Star我的项目关注更新！', msg['RecommendInfo']['UserName'])


itchat.auto_login(True, enableCmdQR=2)
itchat.run()