# coding:utf8
# @Time    : 18-8-9 上午9:54
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from chatbot.preprocessing.text import cut
from chatbot.cparse.vocabulary import Vocabulary
import re
import torch
import numpy as np
import random


def load(path, MAX_LEN):
    s_vocab = Vocabulary(init_vocablary=["<sos>", "<eos>"])
    t_vocab = Vocabulary(init_vocablary=["<sos>", "<eos>"])
    with open(path, 'r') as f:
        data = f.readlines()
    size = int(len(data) / 3)
    source_seq = []
    target_seq = []
    for i in range(size):
        s_i = re.sub(" ", "", data[i * 3 + 1].strip("\n").lower()[2:])
        t_i = re.sub(" ", "", data[i * 3 + 2].strip("\n").lower()[2:])
        if len(s_i) == 0 or len(t_i) == 0 or len(t_i) > MAX_LEN or len(s_i) > MAX_LEN:
            continue
        source_seq.append(s_i)
        target_seq.append(t_i)
    source_seq = cut(source_seq, 4)
    target_seq = cut(target_seq, 4)
    for i in range(len(target_seq)):
        target_seq[i].append("<eos>")
    s_vocab.fit(source_seq)
    t_vocab.fit(target_seq)
    return source_seq, s_vocab, target_seq, t_vocab


def tensorFromSentence(sentence, vocab):
    sentence_id = vocab.transform_sentence(sentence)
    sentence_tensor = torch.tensor(sentence_id, dtype=torch.long).view(-1, 1)
    return sentence_tensor


class DataSet:
    def __init__(self, max_len):
        from chatbot.utils.path import ROOT_PATH
        p = ROOT_PATH / "seq2seq" / "data" / "xiaohuangji"
        source_seq, s_vocab, target_seq, t_vocab = load(p, max_len)
        self.source_seq = source_seq
        self.s_vocab = s_vocab
        self.target_seq = target_seq
        self.t_vocab = t_vocab

    def gen(self):
        idx = list((range(len(self.source_seq))))
        np.random.shuffle(idx)
        for i in idx:
            yield (tensorFromSentence(self.source_seq[i], self.s_vocab),
                   tensorFromSentence(self.target_seq[i], self.t_vocab))

    def source_size(self):
        return len(self.source_seq)

    def target_size(self):
        return len(self.target_seq)


if __name__ == "__main__":
    data = DataSet(20)
    print(data.t_vocab.size())
    print(len(data.target_seq))
