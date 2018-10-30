# coding:utf8
# @Time    : 18-5-15 下午4:16
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import torch
from torch import nn
from torch.nn import functional as F
from chatbot.intent.models.base_intent_model import BaseIntentModel


class FastText(BaseIntentModel):
    def __init__(self, param: dict):
        super().__init__(param)
        embed_dim = param['embed_dim']
        vocab_size = param['vocab_size']
        class_num = param['class_num']
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, class_num)
        self.dropout = nn.Dropout(param["dropout"])

    def forward(self, x):
        x = self.embed(x)
        x = torch.mean(x, dim=1, keepdim=False)
        x = self.dropout(x)
        output = self.fc(x)
        output = F.log_softmax(output, dim=1)
        return output


if __name__ == "__main__":
    import numpy as np
    from chatbot.utils.path import ROOT_PATH, MODEL_PATH
    from chatbot.utils.data import read_fasttext_file
    from chatbot.cparse.vocabulary import Vocabulary
    from chatbot.cparse.label import IntentLabel

    # p = ROOT_PATH.parent / "corpus" / "intent" / "fastText"
    # train_x, train_y = read_fasttext_file(str(p / "demo.train.txt"))
    # test_x, test_y = read_fasttext_file(str(p / "demo.train.txt"))
    # x, y = read_fasttext_file(str(p / "amazon.txt"))
    # train_x, train_y = x[:7000], y[:7000]
    # test_x, test_y = x[7000:], y[7000:]
    import pandas as pd
    from chatbot.preprocessing.text import cut
    # corpus = pd.read_excel(ROOT_PATH.parent/"corpus"/"intent"/"intent_corpus.xlsx")
    # x = cut(corpus.text.tolist())
    # y = corpus.intent.tolist()

    p = ROOT_PATH.parent / "corpus" / "intent"
    c1 = pd.read_excel(str(p / "intent_1.xlsx"))[["text", "intent"]]
    c2 = pd.read_excel(str(p / "intent_2.xlsx"))
    corpus_train = pd.concat([c1, c2]).reset_index(drop=True)
    corpus_test = pd.read_excel(str(p / "intent_valid.xlsx"))
    train_x = cut(corpus_train.text.tolist())
    train_y = corpus_train.intent.tolist()
    test_x = cut(corpus_test.text.tolist())
    test_y = corpus_test.intent.tolist()

    from chatbot.preprocessing.word2vec import get_word2vec_matrix
    # vocab, w2v = get_word2vec_matrix(
    #     str(ROOT_PATH.parent/"corpus"/"word2vec"/"wiki_default")
    # )

    vocab = Vocabulary()
    vocab.fit(train_x)
    label = IntentLabel()
    # label.init_from_config("intent.v0.2.cfg")
    label.fit(train_y)
    train_x = np.array(vocab.transform(train_x, max_length=10))
    test_x = np.array(vocab.transform(test_x, max_length=10))
    train_y = np.array(label.transform(train_y))
    test_y = np.array(label.transform(test_y))


    fasttext_param = {
        "vocab_size": len(vocab),
        # "embed_dim": w2v.shape[1],
        "embed_dim": 40,
        "class_num": len(label),
        "lr": 0.01,
        "dropout": 0.1,
        # "l2_lambda": 0.0001,
    }

    model = FastText(fasttext_param)
    # model.embed.weight.data.copy_(torch.tensor(w2v).float())
    model.fit(train_x, train_y, test_x, test_y, 12, 4, save_best=False)
    # model.param["lr"] = 0.005
    # model.fit(train_x, train_y, test_x, test_y, 8, 8, save_best=False)

    def test(s):
        s = vocab.transform_sentence(cut(s), max_length=10)
        return label.reverse_one(model.infer(s)[0])
    test_sentence = ["我喜欢你", "hi", "四川购售电", "不错", "功率因数多少",
                     "你很厉害哦", "最近最大负载多少", "有什么新政策文件吗",
                     "你们电话多少", "你是谁"]
    print(list(map(test, test_sentence)))

    model.save(str(MODEL_PATH/"v0.23"/"intent_model"))
    vocab.save(str(MODEL_PATH/"v0.23"/"vocab"))
    label.save(str(MODEL_PATH / "v0.23" / "label"))
