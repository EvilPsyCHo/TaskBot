# coding:utf8
# @Time    : 18-6-11 上午9:59
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import pickle

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from chatbot.core.serializable import Serializable
from chatbot.core.trainable import Trainable
from chatbot.core.estimator import Estimator
from chatbot.utils.path import MODEL_PATH
from chatbot.utils.log import get_logger
logger = get_logger("intent model")


class BaseIntentModel(nn.Module, Serializable, Trainable, Estimator):
    def __init__(self, param: dict, opt=torch.optim.Adam,
                 metric=accuracy_score, loss=F.cross_entropy,
                 save_path="default"):
        super().__init__()
        self.param = param

        self.opt = None
        self.loss = loss
        self.metric = metric
        self.start_epoch = 0
        if save_path == "default":
            self.save_path = MODEL_PATH / "intent"
        else:
            self.save_path = save_path

    def forward(self, x):
        raise NotImplementedError

    def evaluate(self, x, y):
        training = self.training
        self.eval()
        test_x = torch.tensor(x)
        test_y = torch.tensor(y)
        logit = self(test_x)
        loss = self.loss(logit, test_y)
        pred = torch.argmax(logit, 1).detach().numpy()
        acc = self.metric(test_y, pred)
        self.training = training
        return {"loss": loss.data, "acc": acc}

    def infer(self, x):
        """已经转换成id的单一样本输入"""
        # TODO
        x = torch.tensor([x])
        logit = F.softmax(self(x),1)
        max_logit, max_class = torch.max(logit, 1)
        return max_class.item(), max_logit.item()

    def _train_step(self, x, y):
        self.opt.zero_grad()
        x = torch.tensor(x)
        y = torch.tensor(y)
        logit = self(x)
        loss = self.loss(logit, y)
        # l2_reg = torch.tensor(.0)
        # for p in self.parameters():
        #     l2_reg += torch.norm(p)
        # loss += self.param["l2_lambda"] * l2_reg
        loss.backward()
        self.opt.step()
        return loss

    def fit(self, train_x, train_y, test_x, test_y,
              epochs, batch_size, log_freq=100, save_best=False,
            opt=torch.optim.Adam):
        self.opt=opt(params=self.parameters(), lr=self.param["lr"])
        self.train()
        best_score = 0
        for epoch in range(1, epochs+1):
            for step, (x, y) in enumerate(self._batch_generator(train_x, train_y, batch_size)):
                loss = self._train_step(x, y)
                if step % log_freq == 0:
                    eval_result = self.evaluate(test_x, test_y)
                    logger.info(
                        "Epoch: {:>2}, Step: {:>6}, train loss: {:>6.6f}, eval loss: {:>6.6f}, eval acc: {:>6.6f}.".format(
                            self.start_epoch+epoch, step, loss, eval_result["loss"], eval_result["acc"]
                        ))
                    if best_score < eval_result["acc"]:
                        best_score = eval_result["acc"]
                        if save_best:
                            name = "{}_epoch:{:2>}_step:{}_acc:{:.4f}.pkl".format(
                                    self.__class__.__name__,
                                    self.start_epoch + epoch,
                                    step,
                                    eval_result["acc"])
                            self.save(name)
        self.start_epoch += epochs

    @staticmethod
    def _batch_generator(x, y, batch):
        assert x.shape[0] == y.shape[0]
        size = x.shape[0]
        idx = np.array(list(range(0, size)))
        np.random.shuffle(idx)
        x = x[idx].copy()
        y = y[idx].copy()
        n = size // batch
        for i in range(n):
            yield x[batch * i: batch * (i + 1)], y[batch * i: batch * (i + 1)]

    def save(self, name):
        name = name + ".%s" % self.__class__.__name__
        path = str(self.save_path / name)
        try:
            with open(path, 'wb') as f:
                pickle.dump(self, f)
            logger.info("success save model in {}".format(path))
        except:
            logger.warning("failed save model in {}".format(path))


    @classmethod
    def load(cls, path):
        try:
            with open(path, 'rb') as f:
                rst = pickle.load(f)
            logger.info("success load model in {}".format(path))
            return rst
        except:
            logger.warning("failed load model in {}".format(path))

