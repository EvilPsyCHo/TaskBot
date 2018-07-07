# -*- coding: utf-8 -*-
# @Time    : 7/7/18 12:04
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com


class MetaSkill(object):
    """MetaSkill, a abstract class for ALL SKILL.

    skill是taskbot执行动作单元，，生成回答或不满足回复要求的反问
    """

    def __call__(self, context):
        """action of response

        return response for query if all parameters is satisfied,
        else, return the question for missing parameters.

        Args:
            context: <class Context>

        Returns:
            <string> or <dict>
        """
        raise NotImplementedError

    def slots(self):
        """slots

        Returns:
            <dict of class Entity>, key is entity name.
        """
        raise NotImplementedError

    def contain_slots(self, entities):
        """contain_slots

        check if the entities is useful for skill

        Args:
            entities: <dict of entities>

        Returns:
            <bool>
        """
        raise NotImplementedError


class MetaEstimator(object):
    """MetaEstimator, a abstract class for ALL Estimator like
    text classification model, sequence annotation model, etc.
    """

    def infer(self, *args, **kwargs):
        """interface for predict samples"""
        raise NotImplementedError


class MetaTransformer(object):
    """MetaTransformer, a abstract class for ALL Transformer like
    word segmenter, word2vec, etc.
    """

    def transform(self, *args, **kwargs):
        """interface for transform samples"""
        raise NotImplementedError

    def reverse(self, *args, **kwargs):
        """interface for r-transform samples"""
        raise NotImplementedError


class MetaSerializable(object):
    """MetaSerializable, a abstract for ALL class which can save & load"""

    @classmethod
    def save(self, path):
        """interface for saving class

        Args:
            path: <str> save path

        Returns:
            <bool> true means saving successful, otherwise it is opposite.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        """interface for loading class

        Args:
            path: <str> load path

        Returns:
            <class> the restored class
        """
        raise NotImplementedError


class MetaTrainable(object):
    """MetaTrainable, a abstract for ALL Class which is trainable"""

    def fit(self, *args, **kwargs):
        """the interface for training

        Args:
            *args:
            **kwargs:

        Returns:
            self
        """
        raise NotImplementedError


class MetaEntity(dict):
    """MetaEntity, a abstract class for ALL ENTITY"""

    def __init__(self, *args, **kwargs):
        """init dict"""
        super().__init__(*args, **kwargs)

    @classmethod
    def name(cls):
        """name of entity

        Returns:
            <str>
        """
        return cls.__name__


class MetaContext(dict):
    """MetaContext, a abstract class for ALL context

    Context record all information about the dialog
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def is_timeout(self):
        """Determine if the context is timeout.

        Returns:
            <bool> True means context is timeout.

        """
        raise NotImplementedError
