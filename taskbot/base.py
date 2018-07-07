# -*- coding: utf-8 -*-
# @Time    : 7/7/18 12:04
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import abc


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

    @property
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

    @classmethod
    def name(cls):
        """name

        Returns:
            <str> name of skill
        """
        return cls.__name__

    def __str__(self):
        return "skill %s" % self.name()

    def __repr__(self):
        return self.__str__()


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
