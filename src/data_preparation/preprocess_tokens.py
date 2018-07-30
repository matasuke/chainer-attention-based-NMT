"""
This module preprocess text corpus.

it pre-process corpus data into processed tokens.
mainly it tokenize sentences, lower characters, remove suffix and replace all digits with 0.
"""

import argparse
import collections
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
        'to_lower',
        'remove_suffix',
        'replace_digits',
        'removed_char',
        'split_digits',
        'ja_tokenizer',
        'segmenter'
    ]

    def __init__(
            self,
            lang='jp',
            tokenize=True,
            to_lower=True,
            remove_suffix=True,
            replace_digits=True
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
        self.to_lower = to_lower
        self.remove_suffix = remove_suffix
        self.replace_digits = replace_digits
        self.removed_char = re.compile(r'[.,!?"\'\";:。、]')
        self.split_digits = re.compile(r'\d')

        if self.tokenize:
            if lang == 'jp':
                import janome
                self.ja_tokenizer = janome.tokenizer.Tokenizer()
                self.segmenter = lambda sen: list(
                    token.surface for token in self.ja_tokenizer.tokenize(sen)
                )

            elif lang == 'ch':
                import jieba
                self.segmenter = lambda sen: list(jieba.cut(sen))

            elif lang == 'en':
                import nltk
                self.segmenter = lambda sen: list(nltk.word_tokenize(sen))

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
        if self.to_lower:
            sen = sen.strip().lower()

        if self.remove_suffix:
            sen = self.removed_char.sub('', sen)

        if self.replace_digits:
            sen = self.split_digits.sub('0', sen)

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


# def index2token(encoded_tokens, word_ids):
#    pass


def load_pickle(in_file):
    """load pickle file."""
    in_path = Path(in_file)
    with in_path.open('rb') as f:
        row_data = pickle.load(f)
    return row_data


def save_pickle(in_file, out_file):
    """save pickle file."""
    out_path = Path(out_file)
    with out_path.open('wb') as f:
        pickle.dump(in_file, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str,
                        help="input path to corpas")
    parser.add_argument('out_path', type=str,
                        help="output path to result")
    parser.add_argument('--tokenize', action='store_false',
                        help='disable tokenize in_path file')
    parser.add_argument('--lang', type=str, choices=['jp', 'en', 'ch'],
                        help="language to be processed")
    parser.add_argument('--tolower', action='store_false',
                        help="disable lower all characters for all sentences.")
    parser.add_argument('--remove_suffix', action='store_false',
                        help="disable remove all suffix like ?,!.")
    parser.add_argument('--replace_digits', action='store_false',
                        help="disable replace digits to 0 for all sentences.")
    parser.add_argument('--cutoff', type=int, default=5,
                        help="cutoff words less than the number digignated")
    parser.add_argument('--vocab_size', type=int, default=0,
                        help='vocabrary size')
    parser.add_argument('--val_in_path', type=str,
                        help='validation dataset')
    parser.add_argument('--val_out_path', type=str,
                        help='validation out path')
    args = parser.parse_args()

    tokenizer = Tokenizer(
        lang=args.lang,
        tokenize=args.tokenize,
        to_lower=args.tolower,
        remove_suffix=args.remove_suffix
    )

    word_counter = collections.Counter()
    word_ids = collections.Counter({
        '<UNK>': 0,
        '<SOS>': 1,
        '<EOS>': 2
    })

    # read files
    f = open(args.in_path, 'r')
    lines = f.readlines()

    sentence_idx = 0
    sentences = []

    # tokenize sentences
    for line in tqdm(lines):
        tokens = []
        tokens += ['<SOS>']
        tokens += tokenizer.pre_process(line)
        tokens += ['<EOS>']

        sentences.append({
            'sentence': line.strip(),
            'tokens': tokens,
            'encoded_tokens': tokens,
            'sentence_idx': sentence_idx
        })

        sentence_idx += 1

        # add each word to word_counter
        word_counter.update(tokens)

    print("total distinct words:{0}".format(len(word_counter)))
    print('top 30 frequent words:')
    for word, num in word_counter.most_common(30):
        print('{0} - {1}'.format(word, num))

    # delete words less than cutoff
    for word, num in dropwhile(
            lambda word_num: word_num[1] >= args.cutoff, word_counter.most_common()
    ):
        del word_counter[word]

    # pick up words of vocab_size
    # minus 1 because unk is included.
    word_counter = word_counter.most_common(
        args.vocab_size-1 if args.vocab_size else len(word_counter)
    )

    for word, num in tqdm(word_counter):
        if word not in word_ids:
            word_ids[word] = len(word_ids)

    print('the number of words more than {0}: {1}'.format(args.cutoff, len(word_ids)))

    # encoding
    for sentence in tqdm(sentences):
        sentence['encoded_tokens'] = \
            token2index(sentence['encoded_tokens'], word_ids)

    output_dataset = {}
    output_dataset['word_ids'] = word_ids
    output_dataset['sentences'] = sentences

    save_pickle(output_dataset, args.out_path)

    if args.val_in_path:
        val_f = open(args.val_in_path, 'r')
        val_lines = val_f.readlines()

        val_sentence_idx = 0
        val_sentences = []

        for line in tqdm(val_lines):
            tokens = []
            tokens += ['<SOS>']
            tokens += tokenizer.pre_process(line)
            tokens += ['<EOS>']

            val_sentences.append({
                'sentence': line.strip(),
                'tokens': tokens,
                'encoded_tokens': tokens,
                'sentence_idx': val_sentence_idx
            })

            val_sentence_idx += 1

        for val_sentence in tqdm(val_sentences):
            val_sentence['encoded_tokens'] = \
                token2index(val_sentence['encoded_tokens'], word_ids)

        val_output_dataset = {}
        val_output_dataset['word_ids'] = word_ids
        val_output_dataset['sentences'] = val_sentences

        save_pickle(val_output_dataset, args.val_out_path)
