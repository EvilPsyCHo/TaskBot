# -*- coding: utf-8 -*-
# @Time    : 9/4/18 14:20
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import re
import logging
from collections import defaultdict

from taskbot.utils import SaveLoad
from taskbot.utils.common import reverse_dict

logger = logging.getLogger("taskbot.Dictionary")
_default_filters = '`~!@#$%^&*()-_=+[{]}\\|,<.>/?～！@#￥%…&*（）-—=+【『】』；：‘“、|，《。》/？'

from sklearn.feature_extraction.text import CountVectorizer

class Dictionary(SaveLoad):
    def __init__(self, stop_words=None, filters=_default_filters, start="<sos>", end="<end>", pad="<pad>",
                 ngram_range=[1, 1], min_df=1, max_df=1.0, lowercase=True,
                 token_pattern=" ", tokenizer=None):
        assert start and end and pad
        self.id2token = {0: pad, 1: start, 2: end}
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

    def _build_tokenizer(self):
        if self.tokenizer:
            return self.tokenizer
        else:
            token_pattern = re.compile(self.token_pattern)
            return lambda x: token_pattern.split(x)

    def _word_ngrams(self, tokens):
        """Turn tokens into a sequence of n-grams after stop words filtering"""
        # handle stop words
        if self.stop_words is not None:
            tokens = [w for w in tokens if w not in self.stop_words]

        # handle token n-grams
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # no need to do any slicing for unigrams
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

    def _fit_raw_documents(self, raw_documents):
        tokenizer = self._build_tokenizer()
        for raw in raw_documents:
            tokens = tokenizer(raw)
            tokens = self._word_ngrams(tokens)
            for token in set(tokens):
                if token in self.token2id:
                    self.dfs[self.token2id[token]] += 1
                else:
                    self.token2id[token] = len(self.token2id)
                    self.dfs[self.token2id[token]] = 1
        self.compact()

    def compact(self):
        """Assign new word ids to all words, shrinking gaps."""
        logger.debug("rebuilding dictionary, shrinking gaps")

        # build mapping from old id -> new id
        idmap = dict(zip(sorted(self.token2id.values()), range(len(self.token2id))))

        # reassign mappings to new ids
        self.token2id = {token: idmap[tokenid] for token, tokenid in self.token2id.items()}
        self.id2token = {}
        self.dfs = {idmap[tokenid]: freq for tokenid, freq in self.dfs.items()}

    def filter_ids(self, bad_ids=None, good_ids=None):
        """Remove the selected `bad_ids` tokens from :class:`~gensim.corpora.dictionary.Dictionary`.
        Alternative - keep selected `good_ids` in :class:`~gensim.corpora.dictionary.Dictionary` and remove the rest.

        Parameters
        ----------
        bad_ids : iterable of int, optional
            Collection of word ids to be removed.
        good_ids : collection of int, optional
            Keep selected collection of word ids and remove the rest.

        Examples
        --------
        >>> from gensim.corpora import Dictionary
        >>>
        >>> corpus = [["máma", "mele", "maso"], ["ema", "má", "máma"]]
        >>> dct = Dictionary(corpus)
        >>> 'ema' in dct.token2id
        True
        >>> dct.filter_tokens(bad_ids=[dct.token2id['ema']])
        >>> 'ema' in dct.token2id
        False
        >>> len(dct)
        4
        >>> dct.filter_tokens(good_ids=[dct.token2id['maso']])
        >>> len(dct)
        1

        """
        if bad_ids is not None:
            bad_ids = set(bad_ids)
            self.token2id = {token: tokenid for token, tokenid in self.token2id.items() if tokenid not in bad_ids}
            self.dfs = {tokenid: freq for tokenid, freq in self.dfs.items() if tokenid not in bad_ids}
        if good_ids is not None:
            good_ids = set(good_ids)
            self.token2id = {token: tokenid for token, tokenid in self.token2id.items() if tokenid in good_ids}
            self.dfs = {tokenid: freq for tokenid, freq in self.dfs.items() if tokenid in good_ids}
        self.compactify()

    def filter_min_max_freq(self):
        """Filter out the 'remove_n' most frequent tokens that appear in the documents.

        Parameters
        ----------
        remove_n : int
            Number of the most frequent tokens that will be removed.

        Examples
        --------
        >>> from gensim.corpora import Dictionary
        >>>
        >>> corpus = [["máma", "mele", "maso"], ["ema", "má", "máma"]]
        >>> dct = Dictionary(corpus)
        >>> len(dct)
        5
        >>> dct.filter_n_most_frequent(2)
        >>> len(dct)
        3

        """
        # determine which tokens to keep
        if self.max_df:
            remove_n = self.max_df if isinstance(self.max_df, int) else int(self.max_df * self.size())
            most_frequent_ids = (v for v in self.token2id.values())
            most_frequent_ids = sorted(most_frequent_ids, key=self.dfs.get, reverse=True)
            most_frequent_ids = most_frequent_ids[:remove_n]
            # do the actual filtering, then rebuild dictionary to remove gaps in ids
            most_frequent_words = [(self[idx], self.dfs.get(idx, 0)) for idx in most_frequent_ids]
            logger.info("discarding %i tokens: %s...", len(most_frequent_ids), most_frequent_words[:10])
            self.filter_ids(bad_ids=most_frequent_ids)
            logger.info("resulting dictionary: %s", self)

        if self.min_df:
            remove_n = self.min_df if isinstance(self.min_df, int) else int(self.min_df * self.size())
            most_frequent_ids = (v for v in self.token2id.values())
            most_frequent_ids = sorted(most_frequent_ids, key=self.dfs.get, reverse=False)
            most_frequent_ids = most_frequent_ids[:remove_n]
            # do the actual filtering, then rebuild dictionary to remove gaps in ids
            most_frequent_words = [(self[idx], self.dfs.get(idx, 0)) for idx in most_frequent_ids]
            logger.info("discarding %i tokens: %s...", len(most_frequent_ids), most_frequent_words[:10])
            self.filter_ids(bad_ids=most_frequent_ids)
            logger.info("resulting dictionary: %s", self)

    def size(self):
        return len(self.token2id)

    def __getitem__(self, tokenid):
        """Get token by provided `tokenid`.

        Parameters
        ----------
        tokenid : int
            Id of token

        Returns
        -------
        str
            Token corresponding to `tokenid`.

        Raises
        ------
        KeyError
            If `tokenid` isn't contained in :class:`~gensim.corpora.dictionary.Dictionary`.

        """
        if len(self.id2token) != len(self.token2id):
            # the word->id mapping has changed (presumably via add_documents);
            # recompute id->word accordingly
            self.id2token = reverse_dict(self.token2id)
        return self.id2token[tokenid]  # will throw for non-existent ids



d = Dictionary(ngram_range=[1, 2])
d._fit_raw_documents(["哈哈 我 说过 这个 吗", "哈哈 是不是 哦"])
d.filter_min_max_freq()

class _Dictionary(SaveLoad):
    def __init__(self, stop_words=None, filters=_default_filters, start="<sos>", end="<end>", pad="<pad>",
                 ngram_range=None, min_df=None, max_df=None, lowercase=True, split=" "):
        assert start and end and pad
        self.id2tokens = {0: pad, 1: start, 2:end}
        self.tokens2id = reverse_dict(self.id2tokens)
        self.dfs = {}
        self.n_docs = 0
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.lowercase = lowercase
        self.split = split
        self.filters = re.compile(filters)

    def _doc2seq(self, doc):
        if self.lowercase:
            doc = doc.lower()
            doc = self.filters.sub("", doc)
        if self.stop_words:
            doc = [i for i in doc if i not in self.stop_words]
        return doc

    def fit(self, documents):
        assert isinstance(documents, list)
        assert len(documents) >= 1
        assert isinstance(documents[0], str)
        for doc in documents:
            self.n_docs += 1
            seq = self._doc2seq(doc)

    def _add_a_doc(self, doc):
        seq = self._doc2seq(doc)
        self.n_docs += 1

    def compactify(self):
        """Assign new word ids to all words, shrinking gaps."""
        logger.debug("rebuilding dictionary, shrinking gaps")

        # build mapping from old id -> new id
        idmap = dict(zip(sorted(self.token2id.values()), range(len(self.token2id))))

        # reassign mappings to new ids
        self.token2id = {token: idmap[tokenid] for token, tokenid in self.token2id.items()}
        self.id2token = {}
        self.dfs = {idmap[tokenid]: freq for tokenid, freq in self.dfs.items()}

    def filter_ids(self, bad_ids=None, good_ids=None):
        """Remove the selected `bad_ids` tokens from :class:`~gensim.corpora.dictionary.Dictionary`.
        Alternative - keep selected `good_ids` in :class:`~gensim.corpora.dictionary.Dictionary` and remove the rest.

        Parameters
        ----------
        bad_ids : iterable of int, optional
            Collection of word ids to be removed.
        good_ids : collection of int, optional
            Keep selected collection of word ids and remove the rest.

        Examples
        --------
        >>> from gensim.corpora import Dictionary
        >>>
        >>> corpus = [["máma", "mele", "maso"], ["ema", "má", "máma"]]
        >>> dct = Dictionary(corpus)
        >>> 'ema' in dct.token2id
        True
        >>> dct.filter_tokens(bad_ids=[dct.token2id['ema']])
        >>> 'ema' in dct.token2id
        False
        >>> len(dct)
        4
        >>> dct.filter_tokens(good_ids=[dct.token2id['maso']])
        >>> len(dct)
        1

        """
        if bad_ids is not None:
            bad_ids = set(bad_ids)
            self.token2id = {token: tokenid for token, tokenid in self.token2id.items() if tokenid not in bad_ids}
            self.dfs = {tokenid: freq for tokenid, freq in self.dfs.items() if tokenid not in bad_ids}
        if good_ids is not None:
            good_ids = set(good_ids)
            self.token2id = {token: tokenid for token, tokenid in self.token2id.items() if tokenid in good_ids}
            self.dfs = {tokenid: freq for tokenid, freq in self.dfs.items() if tokenid in good_ids}
        self.compactify()

    def filter_min_max_freq(self):
        """Filter out the 'remove_n' most frequent tokens that appear in the documents.

        Parameters
        ----------
        remove_n : int
            Number of the most frequent tokens that will be removed.

        Examples
        --------
        >>> from gensim.corpora import Dictionary
        >>>
        >>> corpus = [["máma", "mele", "maso"], ["ema", "má", "máma"]]
        >>> dct = Dictionary(corpus)
        >>> len(dct)
        5
        >>> dct.filter_n_most_frequent(2)
        >>> len(dct)
        3

        """
        # determine which tokens to keep
        if self.max_df:
            remove_n = self.max_df if isinstance(self.max_df, int) else int(self.max_df * self.size)
            most_frequent_ids = (v for v in self.token2id.values())
            most_frequent_ids = sorted(most_frequent_ids, key=self.dfs.get, reverse=True)
            most_frequent_ids = most_frequent_ids[:remove_n]
            # do the actual filtering, then rebuild dictionary to remove gaps in ids
            most_frequent_words = [(self[idx], self.dfs.get(idx, 0)) for idx in most_frequent_ids]
            logger.info("discarding %i tokens: %s...", len(most_frequent_ids), most_frequent_words[:10])
            self.filter_ids(bad_ids=most_frequent_ids)
            logger.info("resulting dictionary: %s", self)

        if self.min_df:
            remove_n = self.min_df if isinstance(self.min_df, int) else int(self.min_df * self.size)
            most_frequent_ids = (v for v in self.token2id.values())
            most_frequent_ids = sorted(most_frequent_ids, key=self.dfs.get, reverse=False)
            most_frequent_ids = most_frequent_ids[:remove_n]
            # do the actual filtering, then rebuild dictionary to remove gaps in ids
            most_frequent_words = [(self[idx], self.dfs.get(idx, 0)) for idx in most_frequent_ids]
            logger.info("discarding %i tokens: %s...", len(most_frequent_ids), most_frequent_words[:10])
            self.filter_ids(bad_ids=most_frequent_ids)
            logger.info("resulting dictionary: %s", self)

    @property
    def size(self):
        return len(self.token2id)

    def __getitem__(self, tokenid):
        """Get token by provided `tokenid`.

        Parameters
        ----------
        tokenid : int
            Id of token

        Returns
        -------
        str
            Token corresponding to `tokenid`.

        Raises
        ------
        KeyError
            If `tokenid` isn't contained in :class:`~gensim.corpora.dictionary.Dictionary`.

        """
        if len(self.id2token) != len(self.token2id):
            # the word->id mapping has changed (presumably via add_documents);
            # recompute id->word accordingly
            self.id2token = reverse_dict(self.token2id)
        return self.id2token[tokenid]  # will throw for non-existent ids
