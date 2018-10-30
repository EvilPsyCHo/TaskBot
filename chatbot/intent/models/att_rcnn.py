# -*- coding: utf-8 -*-
# @Time    : 6/13/18 10:06
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import torch
from torch import nn
from torch.nn import functional as F
from chatbot.intent.models.base_intent_model import BaseIntentModel


class AttRCNN(BaseIntentModel):
    def __init__(self, param, *args, **kwargs):
        super().__init__(param, *args, **kwargs)
        self.lookup = nn.Embedding(param["vocab_size"], param["embed_dim"])
        self.conv11 = nn.Conv2d(1, 4, kernel_size=(3, param["embed_dim"]), padding=(1, 0))
        self.conv12 = nn.Conv2d(1, 4, kernel_size=(5, param["embed_dim"]), padding=(2, 0))
        self.rnn = nn.GRU(param["embed_dim"], param["hidden_dim"], batch_first=True)
        self.fc = nn.Linear(param["hidden_dim"], param["class_num"])

    def forward(self, x):
        embed = self.lookup(x)
        att = self._attention(embed)
        o, h = self.rnn(embed)
        att = att.unsqueeze(2).expand_as(o)
        att_o = att * o
        att_o = torch.sum(att_o, dim=1, keepdim=False)
        logit = F.log_softmax(self.fc(att_o), 1)
        return logit

    def _attention(self, x_embed):
        x_embed = x_embed.unsqueeze(1)
        a1 = F.relu(self.conv11(x_embed).squeeze(3))
        a2 = F.relu(self.conv12(x_embed).squeeze(3))
        # a1, a2 shape: (batch, kernel_num, seq_lenth)
        a = torch.cat([a1, a2], 1)
        a = torch.mean(a, dim=1, keepdim=False)
        return F.softmax(a, 1)

    def get_attention(self, x):
        x_embed = self.lookup(x)
        att = self._attention(x_embed)
        return att


if __name__ == "__main__":
    import numpy as np
    from chatbot.utils.path import ROOT_PATH, MODEL_PATH
    from chatbot.utils.data import read_fasttext_file
    from chatbot.cparse.vocabulary import Vocabulary
    from chatbot.cparse.label import IntentLabel

    p = ROOT_PATH.parent / "corpus" / "intent" / "fastText"
    x, y = read_fasttext_file(str(p / "corpus"))
    train_x, train_y = x[:7000], y[:7000]
    test_x, test_y = x[7000:], y[7000:]
    import copy
    x = copy.deepcopy(train_x)
    test_x, test_y = read_fasttext_file(str(p / "demo.train.txt"))
    vocab = Vocabulary()
    vocab.fit(train_x)
    label = IntentLabel()
    label.fit(train_y)
    train_x = np.array(vocab.transform(train_x, max_length=10))
    test_x = np.array(vocab.transform(test_x, max_length=10))
    train_y = np.array(label.transform(train_y))
    test_y = np.array(label.transform(test_y))

    param = {
        "vocab_size": len(vocab),
        "embed_dim": 60,
        "class_num": len(label),
        "lr": 0.01,
        "hidden_dim": 10,
        # "dropout": 0.5,
    }
    model = AttRCNN(param)
    model.fit(train_x, train_y, test_x, test_y, 2, 32, save_best=False)
    model.param["lr"] = 0.003
    model.fit(train_x, train_y, test_x, test_y, 4, 64, save_best=False)
    # model.save("test")
    # x = FastText.load(str(MODEL_PATH / "intent" / "test.FastText"))
    s = ["你真是可爱阿", "你很喜欢学习哦", "我再也不想理你了",
         "吃饭没", "明天会下雨马", "你哥哥是谁", "你有哥哥么", "弟弟是谁",
         "我想买手机", "我是你主人", "我可以给你打分吗，评价"
         ]
    from chatbot.preprocessing.text import cut

    for i in s:
        print(i, label.reverse_one(model.infer(np.array(vocab.transform_one(cut(i), max_length=10)))[0]))
    from chatbot.evaluate.plot import plot_attention_1d
    idx=1200
    att = model.get_attention(torch.tensor(np.array(vocab.transform_one(train_x[idx], max_length=10)).reshape(-1, 10)))
    print(label.reverse_one(model.infer(train_x[idx])[0]))
    plot_attention_1d([vocab.reverse_one(train_x[idx]).split(" ")],
                      att.detach().numpy())