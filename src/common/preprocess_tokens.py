"""
This module preprocess text corpus.

it pre-process corpus data into processed tokens.
mainly it tokenize sentences, lower characters, remove suffix and replace all digits with 0.
"""

import argparse
import collections
import json
import sys
import pickle
import re
from itertools import dropwhile
from pathlib import Path
from tqdm import tqdm


class Tokenizer(object):
    """
    preprocess sentences into pieces.

    This class split sentences into morphs with
    prossesing of lower case, remove suffix and so on.

    Attributes
    ----------
    lang: str
        language to be processed.
    tokenize: bool
        tokenize sentences into pieces.
        set false if input sentence is already tokenized.
    to_lower: bool
        lower input sentences or not.
    remove_suffix: bool
        whether or not remove characters like [.,!?"'";:。、].
    replace_digits: bool
        whether or not replace digits with 0.
    removed_char: re
        regular expression to find
        some special characters that should be deleted
    split_digits: re
        regular expression to find digits
    """

    __slots__ = [
        'lang',
        'tokenize',
        'normalize',
        'removed_char',
        'split_digits',
        'ja_tokenizer',
        'segmenter'
    ]

    def __init__(
            self,
            lang='ja',
            tokenize=True,
            normalize=True
    ):
        """
        initialize parameters which is used for preprocessing sentences.

        Parameters
        ----------
        lang: str
            language to be processed.
            it should be one of ja, en, ch.
        tokenize: bool
            tokenize sentences into pieces.
            set false if input sentence is already tokenized.
        to_lower:
            lower input sentences or not.
        remove_suffix: bool
            whether or not remove characters like [.,!?"'";:。、].
        replace_digits: bool
            whether or not replace digits with 0.
        """
        self.lang = lang
        self.tokenize = tokenize
        self.normalize = normalize
        self.removed_char = re.compile(r'[.,!?"\'\";:。、]')
        self.split_digits = re.compile(r'\d')

        if tokenize:
            if lang == 'ja':
                from janome.tokenizer import Tokenizer
                self.ja_tokenizer = Tokenizer()
                self.segmenter = lambda sen: list(
                    token.surface for token in self.ja_tokenizer.tokenize(sen)
                )

            elif lang == 'ch':
                import jieba
                self.segmenter = lambda sen: list(jieba.cut(sen))

            elif lang == 'en':
                import nltk
                self.segmenter = lambda sen: list(nltk.word_tokenize(sen))
            else:
                msg = "lang has to be one of 'ja', 'en', 'ch'"
                ValueError(msg)

    def normalize_sentence(self, sen):
        """normalize sentences"""
        sen = sen.strip().lower()
        sen = self.removed_char.sub('', sen)
        sen = self.split_digits.sub('0', sen)

        return sen

    def pre_process(self, sen):
        """
        pre_process sentences into pieces.

        This function pre-process sentences into tokens
        with lower_case, remove some characters and replace all digits into 0.

        Parameters
        ----------
        sen: str
            sentences to be processed.

        Returns
        -------
        self.segmenter(sen) or sen.split()
            if self.tokenize == True, then return
            tokenized sentences with some processing.
            if not, then return splitted sentences.
        """
        if self.normalize:
            sen = self.normalize_sentence(sen)

        return self.segmenter(sen) if self.tokenize else sen.split()


def token2index(tokens, word_ids):
    """
    transform tokens into word_ids.

    Parameters
    ----------
    tokens: list
        list of tokens
    word_ids: list
        word_ids list

    Returns
    -------
    list of word ids
    """
    return [word_ids[token] if token in word_ids
            else word_ids['<UNK>'] for token in tokens]


def encode_sentences(sentences, word_index):
    """encode sentences into indices based on word_index"""
    for sentence in tqdm(sentences):
        sentence['tokens'] = token2index(sentence['tokens'], word_index)

    return sentences


def load_data(in_file):
    """helper to load file."""
    in_path = Path(in_file)
    suffix = in_path.suffix

    if suffix == '.pkl':
        with in_path.open('rb') as f:
            sentences = pickle.load(f)
    elif suffix == '.json':
        with in_path.open('r') as f:
            sentences = json.load(f)
    else:
        with in_path.open('r') as f:
            sentences = f.readlines()

    return sentences


def save_data(in_file, out_file):
    """helper to save file."""
    out_path = Path(out_file)
    suffix = out_path.suffix

    if suffix == '.pkl':
        with out_path.open('wb') as f:
            pickle.dump(in_file, f, pickle.HIGHEST_PROTOCOL)
    elif suffix == '.json':
        with out_path.open('w') as f:
            json.dump(in_file, f)


def create_sentences(sentences, tokenizer):
    sentence_idx = 0
    encoded_sentences = []
    word_counter = collections.Counter()

    for sentence in tqdm(sentences):
        tokens = []
        tokens += ['<SOS>']
        tokens += tokenizer.pre_process(sentence)
        tokens += ['<EOS>']
        encoded_sentences.append({
            'tokens': tokens,
            'sentence_idx': sentence_idx
        })

        sentence_idx += 1

        # add each words to word_counter
        word_counter.update(tokens)

    return encoded_sentences, word_counter


def create_word_dict(word_counter, cutoff=5, vocab_size=False):
    '''
    create word dictionary

    Parameters
    ----------
    word_counter: collenctions.Counter
        Counter object returned from create_captions.
    cutoff: int
        cutoff to dispose words.l
    vocab_size: int
        designate vocabrary size saved in word dictionary.
        default value is set to False.
    '''
    word_ids = collections.Counter({
        '<UNK>': 0,
        '<SOS>': 1,
        '<EOS>': 2
    })

    # create word dictionary
    print("total distinct words:{0}".format(len(word_counter)))
    print('top 30 frequent words:')
    for word, num in word_counter.most_common(30):
        print('{0} - {1}'.format(word, num))

    # delete words less than cutoff
    for word, num in dropwhile(
            lambda word_num: word_num[1] >= cutoff, word_counter.most_common()
    ):
        del word_counter[word]

    # pick up words of vocab_size
    # minus 1 because unk is included.
    word_counter = word_counter.most_common(
        vocab_size-1 if vocab_size else len(word_counter)
    )

    for word, num in tqdm(word_counter):
        if word not in word_ids:
            word_ids[word] = len(word_ids)

    print('total distinct words more than {0} : {1}'.format(cutoff, len(word_counter)))

    return word_ids


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('IN_PATH', type=str,
                        help="input path to corpas")
    parser.add_argument('OUT_PATH', type=str,
                        help="output path to result")
    parser.add_argument('VOCAB', type=str,
                        help="path to input bocaburaly dict for encoding sentences.")
    parser.add_argument('--lang', type=str, choices=['ja', 'en', 'ch'],
                        help="language to be processed")
    parser.add_argument('--validation', action='store_true', default=False,
                        help="precess tokens for validation.(VOCAB has to be path to vocab dict.)")
    parser.add_argument('--tokenize', action='store_true', default=False,
                        help='tokenize in_path file')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help="normalize sentences which include replacing digits with zero, \
                              lowering characters, remove all special characters.")
    parser.add_argument('--cutoff', type=int, default=5,
                        help="cutoff words less than the number digignated (default: %(default)d")
    parser.add_argument('--vocab_size', type=int, default=0,
                        help='vocabrary size (default: all words contained IN_PATH)')
    args = parser.parse_args()

    try:
        if not Path(args.IN_PATH).exists():
            msg = "IN_PATH: %s is not found." % args.IN_PATH
            raise FileNotFoundError(msg)
        if not Path(args.VOCAB) and args.validation:
            msg = "VOCAB: %s is not found." % args.VOCAB
            raise FileNotFoundError(msg)
        if args.cutoff < 0:
            msg = "cutoff has to be >= 0"
            ValueError(msg)
        if args.vocab_size < 0:
            msg = "vocab size has to be >= 0"
            ValueError(msg)
    except Exception as ex:
        parser.print_usage(file=sys.stderr)
        print(ex, file=sys.stderr)
        sys.exit()

    IN_PATH = Path(args.IN_PATH)
    OUT_PATH = Path(args.OUT_PATH)
    VOCAB_PATH = Path(args.VOCAB)

    tokenizer = Tokenizer(
        lang=args.lang,
        tokenize=args.tokenize,
        normalize=args.normalize
    )

    sentences = load_data(IN_PATH)

    print('tokenizing...')
    sentences, word_counter = create_sentences(sentences, tokenizer)

    if args.validation:
        word_index = load_data(VOCAB_PATH)
    else:
        word_index = create_word_dict(word_counter, args.cutoff, args.vocab_size)

    print('encoding...')
    sentences = encode_sentences(sentences, word_index)

    save_data(sentences, OUT_PATH)

    if not args.validation:
        save_data(word_index, VOCAB_PATH)

    print('Done')
