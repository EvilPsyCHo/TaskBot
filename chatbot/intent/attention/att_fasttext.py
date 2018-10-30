# coding:utf8
# @Time    : 18-6-19 上午11:05
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import torch
from torch import nn
from torch.nn import functional as F
from chatbot.intent.models.base_intent_model import BaseIntentModel


class AttFastText(BaseIntentModel):
    def __init__(self, param: dict):
        super().__init__(param)
        embed_dim = param['embed_dim']
        vocab_size = param['vocab_size']
        class_num = param['class_num']
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, class_num)
        self.score_w = nn.Parameter(torch.randn(param['embed_dim']))
        self.score_b = nn.Parameter(torch.randn(1))

    def _score(self, x, mask=None):
        # inputs: embedded x
        score = F.tanh(torch.matmul(x, self.score_w) + self.score_b)
        if mask:
            score.data.masked_fill_(mask, -float('inf'))
        score = F.softmax(score, 1)
        return score

    def get_attention(self, x, mask=None):
        x = torch.tensor(x)
        x =self.embed(x)
        score = self._score(x, mask)
        return score

    def forward(self, x):
        x = self.embed(x)
        score = self._score(x)
        # print(score[0])
        # x = torch.mean(x, dim=1, keepdim=False)
        x_att = torch.sum(score.unsqueeze(2).expand_as(x) * x, dim=1, keepdim=False)
        output = self.fc(x_att)
        output = F.log_softmax(output, dim=1)
        return output


if __name__ == "__main__":
    import numpy as np
    from chatbot.utils.path import ROOT_PATH, MODEL_PATH
    from chatbot.utils.data import read_fasttext_file
    from chatbot.cparse.vocabulary import Vocabulary
    from chatbot.cparse.label import IntentLabel
    from chatbot.preprocessing.text import cut

    # p = ROOT_PATH.parent / "corpus" / "intent" / "fastText"
    # x, y = read_fasttext_file(str(p / "amazon.txt"))
    # train_x, train_y = x[:7000], y[:7000]
    # test_x, test_y = x[7000:], y[7000:]
    import pandas as pd
    p = ROOT_PATH.parent / "corpus" / "intent"
    c1 = pd.read_excel(str(p / "intent_1.xlsx"))[["text", "intent"]]
    c2 = pd.read_excel(str(p / "intent_2.xlsx"))
    corpus_train = pd.concat([c1, c2]).reset_index(drop=True)
    corpus_test = pd.read_excel(str(p / "intent_valid.xlsx"))
    train_x = cut(corpus_train.text.tolist())
    train_y = corpus_train.intent.tolist()
    test_x = cut(corpus_test.text.tolist())
    test_y = corpus_test.intent.tolist()


    vocab = Vocabulary()
    vocab.fit(train_x)
    label = IntentLabel()
    label.fit(train_y)
    train_x = np.array(vocab.transform(train_x, max_length=10))
    test_x = np.array(vocab.transform(test_x, max_length=10))
    train_y = np.array(label.transform(train_y))
    test_y = np.array(label.transform(test_y))

    p = {
        "vocab_size": len(vocab),
        "embed_dim": 40,
        "class_num": len(label),
        "lr": 0.01,
        # "dropout": 0.5,
    }
    model = AttFastText(p)
    model.fit(train_x, train_y, test_x, test_y, 10, 8, save_best=False)
    model.param["lr"] = 0.005
    model.fit(train_x, train_y, test_x, test_y, 2, 32, save_best=False)
    # model.param["lr"] = 0.003
    # model.fit(train_x, train_y, test_x, test_y, 4, 32, save_best=False)
    # model.get_attention(train_x[0].reshape(1, -1))

    from chatbot.evaluate.plot import plot_attention
    idx = corpus_test[corpus_test.intent == "打招呼"].index.tolist()[:10]
    att = model.get_attention(test_x[idx].reshape(-1, 10))
    # print(label.reverse(torch.max(model(torch.tensor(test_x[idx])), 1)[1].numpy()))
    plot_attention(vocab.reverse(test_x[idx]),
                      att.detach().numpy(),
                   [i.strip("__label__") for i in label.reverse(torch.max(model(torch.tensor(test_x[idx])), 1)[1].numpy())],
                   figsize=(15, 12))
