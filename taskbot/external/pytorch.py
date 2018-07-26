# coding:utf8
# @Time    : 18-7-17 上午9:46
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import logging
import os
import json
import numpy as np
import math

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from taskbot.external import metric
from taskbot.external import loss


def _safe_getattr(*obj, attr=None):
    for obj_i in obj:
        try:
            rst = getattr(obj_i, attr)
            return rst
        except AttributeError:
            continue
    raise AttributeError(f"Cant find {attr} in PyTorch or TaskBot.")


class BasePyTorchModel(nn.Module):
    """
    Base class for all PyTorch models
    """
    def __init__(self, config):
        super(BasePyTorchModel, self).__init__()
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)


class BasePyTorchTrainer(object):
    """Base class for all PyTorch trainer"""
    def __init__(self, model, optimizer, config, resume):
        """

        Args:
            model: <class PyTorchModel>
            loss: <torch.nn.loss>
            metric:
            config: <dict or path>
            resume: <Bool>
        """
        # TODO: path config
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.name = config["name"]
        # model setting
        self.model = _safe_getattr(model, config["model"])
        # gpu cpu setting
        self.with_cuda = config['cuda'] and torch.cuda.is_available()
        if config['cuda'] and not torch.cuda.is_available():
            self.logger.warning('Warning: There\'s no CUDA support on this machine, '
                                'training is performed on CPU.')
        else:
            self.gpu = torch.device('cuda:' + str(config['gpu']))
            self.model = self.model.to(self.gpu)
        # trainer setting
        self.loss = _safe_getattr(F, loss, attr=config["loss"])
        self.metric = _safe_getattr(metric, attr=config["metric"])
        self.optimizer = _safe_getattr(optimizer, attr=config["optimizer_type"])(model.parameters(),
                                                                                 **config.get("optimizer", dict()))
        self.epochs = config['trainer']['epochs']
        self.save_freq = config['trainer']['save_freq']
        self.verbosity = config['trainer']['verbosity']
        # GPU setting
        self.with_cuda = config.get("cuda", False) and torch.cuda.is_available()
        if config.get("cuda", False) and not torch.cuda.is_available():
            self.logger.warning('Warning: There\'s no CUDA support on this machine, '
                                'training is performed on CPU.')
        else:
            self.gpu = torch.device('cuda:' + str(config.get("gpu", 0)))
            self.model = self.model.to(self.gpu)
        # learning rate policy
        self.lr_scheduler = getattr(
            optim.lr_scheduler,
            config.get('lr_scheduler_type', None), None)
        if self.lr_scheduler:
            self.lr_scheduler = self.lr_scheduler(self.optimizer, **config['lr_scheduler'])
            self.lr_scheduler_freq = config['lr_scheduler_freq']
        # monitor
        self.monitor = config['trainer']['monitor']
        self.monitor_mode = config['trainer']['monitor_mode']
        assert self.monitor_mode == 'min' or self.monitor_mode == 'max'
        self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf
        self.start_epoch = 1
        self.checkpoint_dir = os.path.join(config['trainer']['save_dir'], self.name)
        # ensure_dir(self.checkpoint_dir)
        json.dump(config, open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
                  indent=4, sort_keys=False)
        if resume:
            self._resume_checkpoint(resume)