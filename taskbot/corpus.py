# -*- coding: utf-8 -*-
# @Time    : 9/26/18 14:36
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import re
from collections import defaultdict
import scipy.sparse
import numpy as np

from taskbot.base import SaveLoad
from taskbot.utils.common import reverse_dict
from taskbot.constant import EOS, SOS, PAD, UNK


__all__ = ["Dictionary"]


class BasicDictionary(SaveLoad):
    def __init__(self, token_pattern=" ", tokenizer=None, min_df=1, max_df=1.0, ngram_range=(1, 1)):
        self.id2token = {}
        self.token2id = reverse_dict(self.id2token)
        self.dfs = {}
        self.n_docs = 0
        self._tokenizer = tokenizer
        self._token_pattern = token_pattern
        self._min_df = min_df
        self._max_df = max_df
        self._ngram_range = ngram_range
        self._keep_ids = None
        self.max_df = max_df
        self.min_df = min_df

    def _build_tokenizer(self):
        """dynamically build tokenizer function"""
        if self._tokenizer:
            return self._tokenizer
        else:
            token_pattern = re.compile(self._token_pattern)
            return lambda x: token_pattern.split(x)

    def _word_ngrams(self, tokens):
        """Turn tokens into a sequence of n-grams after stop words filtering"""
        min_n, max_n = self._ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # don't need to do any slicing for unigrams
                # just iterate through the original tokens
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append
            space_join = " ".join

            for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i: i + n]))
        return tokens

    def _add_token(self, token):
        if token in self.token2id:
            self.dfs[self.token2id[token]] += 1
        else:
            self.token2id[token] = len(self.token2id)
            self.dfs[self.token2id[token]] = 1

    def fit(self, raw_documents):
        tokenizer = self._build_tokenizer()
        for raw in raw_documents:
            tokens = tokenizer(raw)
            tokens = self._word_ngrams(tokens)
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
            bad_ids = set(bad_ids) - self._keep_ids if self._keep_ids else set()
            self.token2id = {token: tokenid for token, tokenid in self.token2id.items() if tokenid not in bad_ids}
            self.dfs = {tokenid: freq for tokenid, freq in self.dfs.items() if tokenid not in bad_ids}
        if good_ids is not None:
            good_ids = set(good_ids) + self._keep_ids if self._keep_ids else set()
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

    def doc2seq(self, raw_documents):
        tokenizer = self._build_tokenizer()
        result = []
        for raw in raw_documents:
            tokens = tokenizer(raw)
            tokens = self._word_ngrams(tokens)
            result_raw = [self.token2id.get(t) for t in tokens if self.token2id.get(t) is not None]
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

    def doc2bow(self, raw_documents):
        tokenizer = self._build_tokenizer()
        result = []
        for raw in raw_documents:
            tokens = tokenizer(raw)
            tokens = [self.token2id.get(i) for i in self._word_ngrams(tokens) if self.token2id.get(i) is not None]
            result_raw = defaultdict(int)
            for t in tokens:
                result_raw[t] += 1
            result_raw = [(k, v) for k, v in result_raw.items()]
            result.append(result_raw)
        return result

    def doc2mat(self, raw_documents):
        tokenizer = self._build_tokenizer()
        values = []
        col_index = []
        raw_index = []
        for i, raw in enumerate(raw_documents):
            tokens = tokenizer(raw)
            tokens = [self.token2id.get(i) for i in self._word_ngrams(tokens) if self.token2id.get(i) is not None]
            result_raw = defaultdict(int)
            for t in tokens:
                result_raw[t] += 1
            values.extend(result_raw.values())
            raw_index.extend([i] * len(result_raw))
            col_index.extend(result_raw.keys())
        return scipy.sparse.csr_matrix(
            (values, (raw_index, col_index)),
            shape=(len(raw_documents), self.size()),
            dtype=np.int64
        )

    def size(self):
        return len(self.token2id)



class Seq2SeqDictionary(BasicDictionary):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fit([PAD, SOS, EOS])
        self._keep_ids = set([self.token2id[i] for i in [PAD, UNK, SOS, EOS]])

    def doc2seq(self, raw_documents, max_len, unk=False, sos=False, eos=False, return_len=False):
        tokenizer = self._build_tokenizer()
        result = []
        for raw in raw_documents:
            tokens = tokenizer(raw)
            tokens = self._word_ngrams(tokens)
            result_raw = [self.token2id.get(t) for t in tokens if self.token2id.get(t) is not None]
            result.append(result_raw)
        return result


class Dictionary(SaveLoad):
    def __init__(self, stop_words=None, filters='',
                 ngram_range=(1, 1), min_df=1, max_df=1.0, lowercase=True,
                 token_pattern=" ", tokenizer=None, pad=False, eos=False, sos=False, unk=False):
        self.id2token = {}
        self.token2id = reverse_dict(self.id2token)
        self.dfs = {}
        self.n_docs = 0
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.tokenizer = tokenizer
        self.token_pattern = token_pattern
        self.lowercase = lowercase
        self.filters = re.compile(filters)
        self._pad = PAD if pad else None
        self._eos = EOS if eos else None
        self._sos = SOS if sos else None
        self._unk = UNK if unk else None
        self._init_dictionary()

    def _init_dictionary(self):
        if self._pad:
            self._add_token(self._pad)
        if self._eos:
            self._add_token(self._eos)
        if self._sos:
            self._add_token(self._sos)
        if self._unk:
            self._add_token(self._unk)

    def _build_tokenizer(self):
        """dynamically build tokenizer function"""
        if self.tokenizer:
            return self.tokenizer
        else:
            token_pattern = re.compile(self.token_pattern)
            return lambda x: token_pattern.split(x)

    def _word_ngrams(self, tokens):
        """Turn tokens into a sequence of n-grams after stop words filtering"""
        if self.stop_words is not None:
            tokens = [w for w in tokens if w not in self.stop_words]

        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # don't need to do any slicing for unigrams
                # just iterate through the original tokens
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append
            space_join = " ".join

            for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i: i + n]))
        return tokens

    def _add_token(self, token):
        if token in self.token2id:
            self.dfs[self.token2id[token]] += 1
        else:
            self.token2id[token] = len(self.token2id)
            self.dfs[self.token2id[token]] = 1

    def fit(self, raw_documents):
        tokenizer = self._build_tokenizer()
        for raw in raw_documents:
            tokens = tokenizer(raw)
            tokens = self._word_ngrams(tokens)
            for token in set(tokens):
                self._add_token(token)
        self._filter_dfs()
        self._compact()

    def doc2seq(self, raw_documents, unk=False, seq2seq=False, pad_len=None):
        tokenizer = self._build_tokenizer()
        result = []
        unk_id = self.token2id[self._unk] if self._unk and unk else None
        for raw in raw_documents:
            tokens = tokenizer(raw)
            tokens = self._word_ngrams(tokens)
            result_raw = [self.token2id.get(t, unk_id) for t in tokens if self.token2id.get(t, unk_id) is not None]
            result.append(result_raw)
        return result

    def seq2doc(self, seqs, join_char=None):
        rest = []
        join_func = join_char.join if join_char else "".join
        for seq in seqs:
            rest.append(
                join_func([self.id2token.get(i) for i in seq if self.id2token.get(i) is not None])
            )
        return rest

    def doc2bow(self, raw_documents):
        tokenizer = self._build_tokenizer()
        result = []
        for raw in raw_documents:
            tokens = tokenizer(raw)
            tokens = [self.token2id.get(i) for i in self._word_ngrams(tokens) if self.token2id.get(i) is not None]
            result_raw = defaultdict(int)
            for t in tokens:
                result_raw[t] += 1
            result_raw = [(k, v) for k, v in result_raw.items()]
            result.append(result_raw)
        return result

    def doc2mat(self, raw_documents):
        tokenizer = self._build_tokenizer()
        values = []
        col_index = []
        raw_index = []
        for i, raw in enumerate(raw_documents):
            tokens = tokenizer(raw)
            tokens = [self.token2id.get(i) for i in self._word_ngrams(tokens) if self.token2id.get(i) is not None]
            result_raw = defaultdict(int)
            for t in tokens:
                result_raw[t] += 1
            values.extend(result_raw.values())
            raw_index.extend([i] * len(result_raw))
            col_index.extend(result_raw.keys())
        return scipy.sparse.csr_matrix(
            (values, (raw_index, col_index)),
            shape=(len(raw_documents), self.size()),
            dtype=np.int64
        )

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
            bad_ids = set(bad_ids)
            if self._pad: bad_ids -= set([self.token2id[self._pad]])
            if self._sos: bad_ids -= set([self.token2id[self._sos]])
            if self._eos: bad_ids -= set([self.token2id[self._eos]])
            if self._unk: bad_ids -= set([self.token2id[self._unk]])
            self.token2id = {token: tokenid for token, tokenid in self.token2id.items() if tokenid not in bad_ids}
            self.dfs = {tokenid: freq for tokenid, freq in self.dfs.items() if tokenid not in bad_ids}
        if good_ids is not None:
            good_ids = set(good_ids)
            if self._pad: good_ids += set([self.token2id[self._pad]])
            if self._sos: good_ids += set([self.token2id[self._sos]])
            if self._eos: good_ids += set([self.token2id[self._eos]])
            if self._unk: good_ids += set([self.token2id[self._unk]])
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

    def size(self):
        return len(self.dfs)

    def __getitem__(self, tokenid):
        """Get token by provided `tokenid`"""
        if len(self.id2token) != len(self.token2id):
            # the word->id mapping has changed (presumably via add_documents);
            # recompute id->word accordingly
            self.id2token = reverse_dict(self.token2id)
        return self.id2token[tokenid]  # will throw for non-existent ids


if __name__ == "__main__":
    docs = ["哈哈 哈哈 我 说过 这个 吗", "哈哈 是不是 哦", "好吧", "哈哈 不错 哦"]
    d = BasicDictionary(ngram_range=(1, 1))
    d = Dictionary()
    d.fit(docs)
    d.dfs
    d.token2id
    [d[i] for i in d.doc2seq(docs)[0]]
    d.doc2bow(docs)
    d.doc2mat(docs)
    d.seq2doc(d.doc2seq(docs))
    d.size()
    d.ngram_range
    d.save("haha")
    x=d.load("haha")
    import os
    os.getcwd()
    os.path.exists("haha")