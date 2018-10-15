# coding: utf-8
# Author: Kevin Zhou
# Mail  : evilpsycho42@gmail.com
# Time  : 9/30/18
import re

from taskbot.base import SaveLoad
from taskbot.utils.common import reverse_dict
from taskbot.constant import EOS, SOS, PAD, UNK, EOS_IDX, SOS_IDX, UNK_IDX, PAD_IDX, PUNC, MUTISPACE


class BasicVocabulary(SaveLoad):

    def __init__(self, token_pattern=" ", tokenizer=None, min_df=1, max_df=1.0,
                 stop_words=None, filters=PUNC):
        self.id2token = {EOS_IDX: EOS, SOS_IDX: SOS, PAD_IDX: PAD, UNK_IDX: UNK}
        self.token2id = reverse_dict(self.id2token)
        self.dfs = {}
        self.n_docs = 0
        self._tokenizer = tokenizer
        self._token_pattern = token_pattern
        self._min_df = min_df
        self._max_df = max_df
        self._keep = set([EOS_IDX, SOS_IDX, UNK_IDX, PAD_IDX])
        self._stop_words = stop_words
        self._filters = filters
        self.max_df = max_df
        self.min_df = min_df

    def _build_tokenizer(self):
        """dynamically build tokenizer function"""
        if self._tokenizer:
            return self._tokenizer
        else:
            token_pattern = re.compile(self._token_pattern)
            return lambda x: token_pattern.split(x)

    def _build_filter(self):
        cpl1 = re.compile(self._filters)
        cpl2 = re.compile(MUTISPACE)
        return lambda raw: cpl2.sub("", cpl1.sub("", raw))

    def _add_token(self, token):
        if token in self.token2id:
            self.dfs[self.token2id[token]] += 1
        else:
            self.token2id[token] = len(self.token2id)
            self.dfs[self.token2id[token]] = 1

    def fit(self, raw_documents):
        tokenizer = self._build_tokenizer()
        filters = self._build_filter()
        for raw in raw_documents:
            raw = filters(raw)
            tokens = tokenizer(raw)
            for token in set(tokens):
                self._add_token(token)
        self._filter_dfs()
        self._compact()

    def _compact(self):
        """Assign new word ids to all words, shrinking gaps."""
        # build mapping from old id -> new id
        idmap = dict(zip(sorted(self.token2id.values()), range(len(self.token2id))))
        # reassign mappings to new ids
        self.token2id = {token: idmap[tokenid] for token, tokenid in self.token2id.items()}
        self.id2token = {}
        self.dfs = {idmap[tokenid]: freq for tokenid, freq in self.dfs.items()}
        self.id2token = reverse_dict(self.token2id)

    def _filter_ids(self, bad_ids=None, good_ids=None):
        """Remove the selected `bad_ids` tokens from :class:`~gensim.corpora.dictionary.Dictionary`.
        Alternative - keep selected `good_ids` in :class:`~gensim.corpora.dictionary.Dictionary` and remove the rest.

        Parameters
        ----------
        bad_ids : iterable of int, optional
            Collection of word ids to be removed.
        good_ids : collection of int, optional
            Keep selected collection of word ids and remove the rest.
        """
        if bad_ids is not None:
            bad_ids = set(bad_ids) - self._keep if self._keep else set()
            self.token2id = {token: tokenid for token, tokenid in self.token2id.items() if tokenid not in bad_ids}
            self.dfs = {tokenid: freq for tokenid, freq in self.dfs.items() if tokenid not in bad_ids}
        if good_ids is not None:
            good_ids = set(good_ids) | self._keep if self._keep else set()
            self.token2id = {token: tokenid for token, tokenid in self.token2id.items() if tokenid in good_ids}
            self.dfs = {tokenid: freq for tokenid, freq in self.dfs.items() if tokenid in good_ids}
        self._compact()

    def _filter_dfs(self):
        """filter token by the min and max token frequency in documents"""
        if self.max_df:
            max_df = self.max_df if isinstance(self.max_df, int) else int(self.max_df * self.size())
            most_frequent_ids = dict(filter(lambda x: x[1] > max_df, self.dfs.items()))
            self._filter_ids(bad_ids=most_frequent_ids.keys())

        if self.min_df:
            min_df = self.min_df if isinstance(self.min_df, int) else int(self.min_df * self.size())
            most_frequent_ids = dict(filter(lambda x: x[1] < min_df, self.dfs.items()))
            self._filter_ids(bad_ids=most_frequent_ids.keys())

    def doc2seq(self, raw_documents, return_length=False, pad_length=None, seq2seq=False):
        tokenizer = self._build_tokenizer()
        filters = self._build_filter()
        result = []
        for raw in raw_documents:
            raw = filters(raw)
            tokens = tokenizer(raw)
            result_raw = [self.token2id.get(t, UNK_IDX) for t in tokens]
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

    def size(self):
        return len(self.token2id)
