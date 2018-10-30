# coding:utf8
# @Time    : 18-6-8 下午1:45
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com


class BaseSkill(object):
    """chatbot能力

    接收上下文输入，返回当前会话回复
    """
    def __call__(self, context):
        """skill回复逻辑封装

        满足skill需求给出回复，不满足需求，给出反问

        :param context: <class Context> 当前会话上下文
        :return: <String> 回复信息
        """
        raise NotImplementedError
        
    @classmethod
    def name(cls):
        """skill name

        :return: <String>
        """
        return cls.__name__

    @property
    def init_slots(self):
        """skill name

        :return: <dict> slots dict
        """
        raise NotImplementedError

    def contain_slots(self, entities):
        """判断是否包含skill所需词槽

        :param entities: <list of dict>
        :return: <bool> True包含，False不包含
        """
        raise NotImplementedError

    def __str__(self):
        return "Skill %s" % self.name()

    def __repr__(self):
        return self.__str__()
