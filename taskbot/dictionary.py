# coding: utf-8
# Author: Kevin Zhou
# Mail  : evilpsycho42@gmail.com
# Time  : 10/10/18
import re

from taskbot.base import SaveLoad
from taskbot.utils.common import reverse_dict
from taskbot.constant import EOS, SOS, PAD, UNK


class Seq2SeqDictionary(SaveLoad):
    def __init__(self, token_pattern=" ", tokenizer=None):
        self.token_pattern = token_pattern
        self.tokenizer = tokenizer
        self.token2id = {}
        self.id2token = {}
        for token in [EOS, SOS, PAD, UNK]:
            self._add_token(token)
        self.id2token = reverse_dict(self.token2id)

    def _build_tokenizer(self):
        """dynamically build tokenizer function"""
        if self._tokenizer:
            return self._tokenizer
        else:
            token_pattern = re.compile(self._token_pattern)
            return lambda doc: token_pattern.split(doc)

    def _add_token(self, token):
        if token not in self.token2id:
            self.token2id[token] = self.size

    def fit(self, docs):
        tokenizer = self._build_tokenizer()
        for raw in docs:
            tokens = tokenizer(raw)
            for token in set(tokens):
                self._add_token(token)
        self.id2token = reverse_dict(self.token2id)

    def doc2seq(self, raw_documents):
        tokenizer = self._build_tokenizer()
        unk_index = self.token2id[UNK]
        result = []
        for raw in raw_documents:
            tokens = tokenizer(raw)
            result_raw = [self.token2id.get(token, unk_index) for token in tokens]
            result.append(result_raw)
        return result

    def seq2doc(self, seqs, join_char=None):
        rest = []
        join_func = join_char.join if join_char else " ".join
        for seq in seqs:
            rest.append(
                join_func([self.id2token.get(i) for i in seq if self.id2token.get(i) is not None])
            )
        return rest


    @property
    def size(self):
        return len(self.token2id)
