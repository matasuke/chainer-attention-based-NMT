import unittest
from pathlib import Path
import numpy as np

from Seq2SeqDataset import Seq2SeqDatasetBase

class TestSeq2SeqDatasetBase(unittest.TestCase):
    def setUp(self):
        SOURCE_SENTENCE_PATH = Path('data/ja_dataset.pkl')
        TARGET_SENTENCE_PATH = Path('data/en_dataset.pkl')
        N_SOURCE_MIN_TOKENS = 1
        N_SOURCE_MAX_TOKENS = 50
        N_TARGET_MIN_TOKENS = 1
        N_TARGET_MAX_TOKENS = 50

        self.assertTrue(SOURCE_SENTENCE_PATH.exists())
        self.assertTrue(TARGET_SENTENCE_PATH.exists())

        self.dataset = Seq2SeqDatasetBase(
            source_sentence_path=str(SOURCE_SENTENCE_PATH),
            target_sentence_path=str(TARGET_SENTENCE_PATH),
            n_source_min_tokens=N_SOURCE_MIN_TOKENS,
            n_source_max_tokens=N_SOURCE_MAX_TOKENS,
            n_target_min_tokens=N_TARGET_MIN_TOKENS,
            n_target_max_tokens=N_TARGET_MAX_TOKENS,
        )

        randn = np.random.randint(0, len(self.dataset))
        self.target_index = self.dataset.get_example(randn)
        self.reference_index = self.dataset.pairs[randn]
        self.source_target_index = self.target_index[0]
        self.source_reference_index = self.reference_index[0]
        self.target_target_index = self.target_index[1]
        self.target_reference_index = self.reference_index[1]

    def test_get_example(self):
        self.assertIsInstance(self.target_index, tuple)
        self.assertEqual(self.target_index, self.reference_index)

    def test_source_index2token(self):
        target_tokens = self.dataset.source_index2token(self.source_target_index)
        reference_tokens = self.dataset.source_index2token(self.source_reference_index)

        self.assertEqual(target_tokens, reference_tokens)

    def test_target_index2token(self):
        target_tokens = self.dataset.source_index2token(self.target_target_index)
        reference_tokens = self.dataset.source_index2token(self.target_reference_index)

        self.assertEqual(target_tokens, reference_tokens)

    def test_source_token2index(self):
        target_tokens = self.dataset.source_index2token(self.source_target_index)
        reversed_index = self.dataset.source_token2index(target_tokens)

        self.assertEqual(reversed_index, list(self.source_reference_index))

    def test_target_token2index(self):
        target_tokens = self.dataset.target_index2token(self.target_target_index)
        reversed_index = self.dataset.target_token2index(target_tokens)

        self.assertEqual(reversed_index, list(self.target_reference_index))

    def test_get_source_word_ids(self):
        word_ids = self.dataset.get_source_word_ids
        reference = self.dataset.source_word_ids
        self.assertEqual(word_ids, reference)

    def test_get_target_word_ids(self):
        word_ids = self.dataset.get_target_word_ids
        reference = self.dataset.target_word_ids
        self.assertEqual(word_ids, reference)


if __name__ == '__main__':
    unittest.main()
