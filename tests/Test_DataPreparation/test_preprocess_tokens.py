import numpy as np
import unittest

from src.data_preparation.preprocess_tokens import Tokenizer
class TestPreprocessTokens(unittest.TestCase):
    def setUp(self):

        # jaanese tokenizer with all preprocesses
        self.tokenizer_wapp = Tokenizer(
            lang='jp',
            tokenize=True,
            to_lower=True,
            replace_digits=True
        )

        # japanese tokenizer with no preprocess
        self.tokenizer_wnpp = Tokenizer(
            lang='jp',
            tokenize=False,
            to_lower=False,
            replace_digits=False
        )

        # english tokenizer with all preprocesses
        self.tokenizer_en_wapp = Tokenizer(
            lang='en',
            tokenize=True,
            to_lower=True,
            replace_digits=True
        )

        # english tokenizer with no preprocess
        self.tokenizer_en_wnpp = Tokenizer(
            lang='en',
            tokenizer=False,
            to_lower=False,
            replace_digits=False
        )

        # chinese tokenizer with all preprocesses
        self.tokenizer_ch_wapp = Tokenizer(
            lang='ch',
            tokenize=True,
            to_lower=True,
            replace_digits=True
        )

        # chinese tokenizer with no preprocess
        self.tokenizer_ch_wnpp = Tokenizer(
            lang='ch',
            tokenize=False,
            to_lower=False,
            replace_digits=False
        )
