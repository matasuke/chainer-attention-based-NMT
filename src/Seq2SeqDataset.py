import json
import pickle
from pathlib import Path
import numpy as np

import chainer

# TODO: separate vocaburaly and sentences to reduce memory size when using devaloping dataset

class Seq2SeqDatasetBase(chainer.dataset.DatasetMixin):
    def __init__(
            self,
            source_sentence_path,
            target_sentence_path,
            n_source_min_tokens,
            n_source_max_tokens,
            n_target_min_tokens,
            n_target_max_tokens,
    ):

        # TODO: chenge it to use p.open() and exception

        if Path(source_sentence_path).exists() \
                and Path(target_sentence_path).exists():
            source_data = self.load_data(source_sentence_path)
            target_data = self.load_data(target_sentence_path)
            assert len(source_data['sentences']) == len(target_data['sentences'])

        self.pairs = [
            (np.array(s['encoded_tokens'], np.int32),
             np.array(t['encoded_tokens'], np.int32))
            for s, t in zip(source_data['sentences'], target_data['sentences'])
            if n_source_min_tokens <= len(s['encoded_tokens']) <= n_source_max_tokens and
            n_target_min_tokens <= len(t['encoded_tokens']) <= n_target_max_tokens
        ]

        self.source_word_ids = source_data['word_ids']
        self.target_word_ids = target_data['word_ids']
        self.inv_source_word_ids = {
            v: k for k, v in source_data['word_ids'].items()
        }
        self.inv_target_word_ids = {
            v: k for k, v in target_data['word_ids'].items()
        }

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def load_data(path):
        in_path = Path(path)
        ext = in_path.suffix
        if ext == '.pickle':
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
        unk = sum((s == self.source_word_ids['<UNK>']).sum() for s in data)
        words = sum(s.size for s in data)

        return round((unk / words) * 100, 3)

    @staticmethod
    def index2token(indices, inv_word_ids):
        return [inv_word_ids[index] for index in indices]

    @staticmethod
    def token2index(tokens, word_ids):
        return [word_ids[token] for token in tokens]

    def get_example(self, i):
        return self.pairs[i]

    def source_token2index(self, tokens):
        return self.token2index(tokens, self.source_word_ids)

    def source_index2token(self, indices):
        return self.index2token(indices, self.inv_source_word_ids)

    def target_token2index(self, tokens):
        return self.token2index(tokens, self.target_word_ids)

    def target_index2token(self, indices):
        return self.index2token(indices, self.inv_target_word_ids)

    @property
    def get_source_word_ids(self):
        return self.source_word_ids

    @property
    def get_target_word_ids(self):
        return self.target_word_ids

    @property
    def source_unk_ratio(self):
        return self.calc_unk_ratio([s for s, _ in self.pairs])

    @property
    def target_unk_ratio(self):
        return self.calc_unk_ratio([t for _, t in self.pairs])

    @property
    def get_configurations(self):
        res = {}

        res['Source_vocabulary_size'] = len(self.get_source_word_ids)
        res['Target_vocabulary_size'] = len(self.get_target_word_ids)
        res['Train_data_size'] = len(self.pairs)
        res['Source_unk_ratio'] = self.source_unk_ratio
        res['Target_unk_ratio'] = self.target_unk_ratio

        # TODO: get number of sentences in validation dataset

        return res
