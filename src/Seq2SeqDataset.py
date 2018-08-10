import sys
import json
import pickle
from pathlib import Path
import numpy as np

import chainer


class Seq2SeqDatasetBase(chainer.dataset.DatasetMixin):
    """
    Chainer Dataset Class for Seq2Seq

    Attributes
    ----------
    source_word_ids : dict
        map to source ids to tokens.

    target_word_ids : dict
        map to target ids to tokens.

    inv_wource_word_ids : dict
        map to source tokens to word_ids.

    inv_target_word_ids : dict
        map to target tokens to word_ids.

    pairs : list of numpy.ndarray
        pair data contains source and target sentences.

    """

    def __init__(
            self,
            source_sentence_path,
            target_sentence_path,
            source_vocab_path,
            target_vocab_path,
            n_source_min_tokens=1,
            n_source_max_tokens=50,
            n_target_min_tokens=1,
            n_target_max_tokens=50,
    ):
        """
        Parameters
        ----------
        source_sentence_path : str
            path to dataset which contains source sentences processed by preprocess_tokens.py.
        target_sentence_path : str
            path to dataset which contains target sentences processed by preprocess_tokens.py.
        source_vocab_path : str
            path to source vocaburaly dict processed by preprocess_tokens.py.
        target_vocab_path : str
            path to target vocaburaly dict processed by preprocess_tokens.py.
        n_source_min_tokens : int, default 1
            minimum number of tokens to be used in source sentence.
        n_source_max_tokens : int, default 50
            maximum number of tokens to be used in source sentence.
        n_target_min_tokens : int, default 1
            minimum number of tokens to be used in target sentence.
        n_target_max_tokens : int, default 50
            minimum number of tokens to be used in target sentence.
        """
        try:
            if Path(source_sentence_path).exists() \
                    and Path(target_sentence_path).exists():
                source_data = self.load_data(source_sentence_path)
                target_data = self.load_data(target_sentence_path)
                assert len(source_data) == len(target_data)
            else:
                if not Path(source_sentence_path).exists():
                    msg = "File %s is not found." % source_sentence_path
                    FileNotFoundError(msg)
                elif not Path(target_sentence_path).exists():
                    msg = "File %s is not found." % target_sentence_path
                    FileNotFoundError(msg)

            if Path(source_vocab_path).exists():
                self.source_word_ids = self.load_data(source_vocab_path)
            else:
                msg = "File %s is not found." % source_vocab_path
                FileNotFoundError(msg)
            if Path(target_vocab_path).exists():
                self.target_word_ids = self.load_data(target_vocab_path)
            else:
                msg = "File %s is not found." % target_vocab_path
                FileNotFoundError(msg)

        except Exception as ex:
            print(ex, file=sys.stderr)
            sys.exit()

        self.pairs = [
            (np.array(s['tokens'], np.int32),
             np.array(t['tokens'], np.int32))
            for s, t in zip(source_data, target_data)
            if n_source_min_tokens <= len(s['tokens']) <= n_source_max_tokens and
            n_target_min_tokens <= len(t['tokens']) <= n_target_max_tokens
        ]

        self.inv_source_word_ids = {
            v: k for k, v in self.source_word_ids.items()
        }
        self.inv_target_word_ids = {
            v: k for k, v in self.target_word_ids.items()
        }

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def load_data(path):
        """helper to load data."""
        in_path = Path(path)
        ext = in_path.suffix
        if ext == '.pkl':
            with in_path.open('rb') as f:
                data = pickle.load(f)
        elif ext == '.json':
            with in_path.open('r') as f:
                data = json.load(f)
        else:
            msg = 'File %s can be loaded.\n \
                    choose json or pickle format' % path
            raise TypeError(msg)

        return data

    # return in percenrage ratio
    def calc_unk_ratio(self, data):
        """base function for callculate <UNK> ratio"""
        unk = sum((s == self.source_word_ids['<UNK>']).sum() for s in data)
        words = sum(s.size for s in data)

        return round((unk / words) * 100, 3)

    @staticmethod
    def index2token(indices, inv_word_ids):
        """return tokens from indices."""
        return [inv_word_ids[index] for index in indices]

    @staticmethod
    def token2index(tokens, word_ids):
        """return indies from tokens."""
        return [word_ids[token] for token in tokens]

    def get_example(self, i):
        """get formatted data"""
        return self.pairs[i]

    def source_token2index(self, tokens):
        """return source indies from source tokens."""
        return self.token2index(tokens, self.source_word_ids)

    def source_index2token(self, indices):
        """return source tokens from source indices."""
        return self.index2token(indices, self.inv_source_word_ids)

    def target_token2index(self, tokens):
        """return target indies from target tokens."""
        return self.token2index(tokens, self.target_word_ids)

    def target_index2token(self, indices):
        """return target tokens from target indices."""
        return self.index2token(indices, self.inv_target_word_ids)

    @property
    def get_source_word_ids(self):
        """get source word_ids"""
        return self.source_word_ids

    @property
    def get_target_word_ids(self):
        """get target word_ids"""
        return self.target_word_ids

    @property
    def source_unk_ratio(self):
        """get <UNK> ratio in source sentences"""
        return self.calc_unk_ratio([s for s, _ in self.pairs])

    @property
    def target_unk_ratio(self):
        """get <UNK> ratio in target sentences"""
        return self.calc_unk_ratio([t for _, t in self.pairs])

    @property
    def get_configurations(self):
        """get configurations"""
        res = {}

        res['Source_vocabulary_size'] = len(self.get_source_word_ids)
        res['Target_vocabulary_size'] = len(self.get_target_word_ids)
        res['Train_data_size'] = len(self.pairs)
        res['Source_unk_ratio'] = self.source_unk_ratio
        res['Target_unk_ratio'] = self.target_unk_ratio

        return res
