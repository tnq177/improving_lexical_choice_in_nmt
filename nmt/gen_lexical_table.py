from __future__ import print_function
from __future__ import division

import argparse
from codecs import open
from os.path import exists

import numpy
import all_constants as ac

def init_vocab(vocab_file):
    if not exists(vocab_file):
        raise ValueError('    Vocab file {} not found'.format(vocab_file))

    vocab = {}
    idx = 0
    with open(vocab_file, 'r', 'utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                word = word.split()[0]
                vocab[word] = idx
                idx += 1

    ivocab = {i: w for w, i in vocab.items()}
    return vocab, ivocab

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', required=True)
    parser.add_argument('-t', required=True)
    parser.add_argument('--src-vocab', required=True)
    parser.add_argument('--trg-vocab', required=True)
    parser.add_argument('-l', required=True)
    args = parser.parse_args()

    src_lang = args.s
    trg_lang = args.t
    src_vocab_file = args.src_vocab
    trg_vocab_file = args.trg_vocab
    lexical_file = args.l

    src_vocab, _ = init_vocab(src_vocab_file)
    trg_vocab, _ = init_vocab(trg_vocab_file)

    # p(trg|src)
    lex_table = numpy.zeros([len(src_vocab), len(trg_vocab)], dtype=numpy.float32)
    with open(lexical_file, 'r', 'utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            src_word, trg_word, prob = line.strip().split()
            if src_word not in src_vocab:
                continue

            prob = numpy.float32(prob)
            if prob < 0. or prob > 1.:
                print("{} --> {}: {}".format(src_word, trg_word, prob))

            src_id = src_vocab[src_word]
            trg_id = trg_vocab.get(trg_word, ac.UNK_ID)
            lex_table[src_id, trg_id] = prob

    for src_id in xrange(len(src_vocab)):
        sum_p_ef = numpy.sum(lex_table[src_id]) - lex_table[src_id, ac.UNK_ID]
        lex_table[src_id, ac.UNK_ID] = 1.0 - sum_p_ef

    numpy.save('./{}2{}.npy'.format(src_lang, trg_lang), lex_table)