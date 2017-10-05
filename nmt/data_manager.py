from __future__ import print_function
from __future__ import division

import sys
import os
import time
from os.path import join
from os.path import exists
from itertools import izip, islice
from collections import Counter
from codecs import open
import numpy
import tensorflow as tf

import nmt.utils as ut
import nmt.all_constants as ac


numpy.random.seed(ac.SEED)


class DataManager(object):

    def __init__(self, config):
        super(DataManager, self).__init__()
        self.logger = ut.get_logger(config['log_file'])

        self.src_lang = config['src_lang']
        self.trg_lang = config['trg_lang']
        self.data_dir = config['data_dir']
        self.batch_size = config['batch_size']
        self.reverse = config['reverse']

        self.vocab_sizes = {
            self.src_lang: config['src_vocab_size'],
            self.trg_lang: config['trg_vocab_size']
        }

        self.max_src_length = config['max_src_length']
        self.max_trg_length = config['max_trg_length']

        self.data_files = {
            ac.TRAINING: {
                self.src_lang: join(self.data_dir, 'train.{}'.format(self.src_lang)),
                self.trg_lang: join(self.data_dir, 'train.{}'.format(self.trg_lang))
            },
            ac.VALIDATING: {
                self.src_lang: join(self.data_dir, 'dev.{}'.format(self.src_lang)),
                self.trg_lang: join(self.data_dir, 'dev.{}'.format(self.trg_lang))
            },
            ac.TESTING: {
                self.src_lang: join(self.data_dir, 'test.{}'.format(self.src_lang)),
                self.trg_lang: join(self.data_dir, 'test.{}'.format(self.trg_lang))
            } 
        }
        self.length_files = {
            ac.TRAINING: join(self.data_dir, 'train.length'),
            ac.VALIDATING: join(self.data_dir, 'dev.length'),
            ac.TESTING: join(self.data_dir, 'test.length')
        }
        self.clean_files = {
            self.src_lang: join(self.data_dir, 'train.{}.clean-{}'.format(self.src_lang, self.max_src_length)),
            self.trg_lang: join(self.data_dir, 'train.{}.clean-{}'.format(self.trg_lang, self.max_trg_length))
        }
        self.ids_files = {
            ac.TRAINING: join(self.data_dir, 'train.ids'),
            ac.VALIDATING: join(self.data_dir, 'dev.ids'),
            ac.TESTING: join(self.data_dir, 'test.ids')
        }
        self.vocab_files = {
            self.src_lang: join(self.data_dir, 'vocab-{}.{}'.format(self.vocab_sizes[self.src_lang], self.src_lang)),
            self.trg_lang: join(self.data_dir, 'vocab-{}.{}'.format(self.vocab_sizes[self.trg_lang], self.trg_lang))
        }

        self.setup()

    def setup(self):
        self.parallel_filter_long_sentences()
        self.create_vocabs(self.src_lang)
        self.create_vocabs(self.trg_lang)
        self.parallel_data_to_token_ids(mode=ac.TRAINING)
        self.parallel_data_to_token_ids(mode=ac.VALIDATING)

        if exists(self.data_files[ac.TESTING][self.src_lang]) and exists(self.data_files[ac.TESTING][self.trg_lang]):
            self.parallel_data_to_token_ids(mode=ac.TESTING)

    def parallel_filter_long_sentences(self):
        src_train_file = self.data_files[ac.TRAINING][self.src_lang]
        trg_train_file = self.data_files[ac.TRAINING][self.trg_lang]
        src_train_clean_file = self.clean_files[self.src_lang]
        trg_train_clean_file = self.clean_files[self.trg_lang]
        max_src_length = self.max_src_length
        max_trg_length = self.max_trg_length

        self.logger.info('Filter {} & {} by length {} & {}'.format(src_train_file, trg_train_file, max_src_length, max_trg_length))
        if exists(src_train_clean_file) and exists(trg_train_clean_file):
            self.logger.info(
                '    Length-filtered files exist at {} & {}'.format(src_train_clean_file, trg_train_clean_file))
            return

        open(src_train_clean_file, 'w').close()
        open(trg_train_clean_file, 'w').close()

        with open(src_train_file, 'r', 'utf-8') as in_src_f, \
                open(trg_train_file, 'r', 'utf-8') as in_trg_f, \
                open(src_train_clean_file, 'w', 'utf-8') as out_src_f, \
                open(trg_train_clean_file, 'w', 'utf-8') as out_trg_f:

            for src_line, trg_line in izip(in_src_f, in_trg_f):
                if src_line.strip() and trg_line.strip():
                    src_len = len(src_line.strip().split())
                    trg_len = len(trg_line.strip().split())

                    if 0 < src_len <= max_src_length and 0 < trg_len <= max_trg_length:
                        out_src_f.write(src_line)
                        out_trg_f.write(trg_line)

    def create_vocabs(self, lang):
        data_file = self.clean_files[lang]
        vocab_file = self.vocab_files[lang]
        max_vocab_size = self.vocab_sizes[lang]

        self.logger.info('Create vocabulary for {} with size {} from {}'.format(lang, max_vocab_size, data_file))
        if exists(vocab_file):
            self.logger.info('    Vocabulary for data {} exists at {}'.format(data_file, vocab_file))
            return

        with open(data_file, 'r', 'utf-8') as f:
            vocab = Counter()
            count = 0
            for line in f:
                count += 1
                if count % 10000 == 0:
                    self.logger.info('   processing line {}'.format(count))

                if line.strip():
                    vocab.update(line.strip().split())

            vocab_list = ac._START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) < max_vocab_size:
                msg = '    The actual vocab size {} is < than your required {}\n'.format(
                    len(vocab_list), max_vocab_size)
                msg += '    Due to shameful reason, this cannot be handled automatically\n'
                msg += '    Please change the vocab size for {} to {}'.format(
                    vocab_file, len(vocab_list))
                self.logger.info(msg)
                raise ValueError(msg)
            else:
                vocab_list = vocab_list[:max_vocab_size]

            with open(vocab_file, 'w', 'utf-8') as fout:
                for w in vocab_list:
                    tok_freq = 0 if w in ac._START_VOCAB else vocab[w]
                    fout.write(u'{} {}\n'.format(w, tok_freq))

    def init_vocab(self, lang):
        vocab_file = self.vocab_files[lang]
        self.logger.info('Initialize {} vocab from {}'.format(lang, vocab_file))

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

    def parallel_data_to_token_ids(self, mode=ac.TRAINING):
        src_file = self.data_files[mode][self.src_lang]
        trg_file = self.data_files[mode][self.trg_lang]
        if mode == ac.TRAINING:
            # In training, we use the length-limited data
            src_file = self.clean_files[self.src_lang]
            trg_file = self.clean_files[self.trg_lang]

        src_vocab, _ = self.init_vocab(self.src_lang)
        trg_vocab, _ = self.init_vocab(self.trg_lang)

        joint_file = self.ids_files[mode]
        joint_length_file = self.length_files[mode]

        msg = 'Parallel convert tokens from {} & {} to ids and save to {}'.format(src_file, trg_file, joint_file)
        msg += '\nAlso save the total data length to {}'.format(joint_length_file)
        self.logger.info(msg)
        
        if exists(joint_file) and exists(joint_length_file):
            self.logger.info('    Token-id-ed data exists at {}'.format(joint_file))
            return

        open(joint_file, 'w').close()
        open(joint_length_file, 'w').close()
        num_lines = 0
        with open(src_file, 'r', 'utf-8') as src_f, \
                open(trg_file, 'r', 'utf-8') as trg_f, \
                open(joint_file, 'w', 'utf-8') as tokens_f:

            for src_line, trg_line in izip(src_f, trg_f):
                src_toks = src_line.strip()
                trg_toks = trg_line.strip()

                if src_toks and trg_toks:
                    num_lines += 1
                    if num_lines % 10000 == 0:
                        self.logger.info('    converting line {}'.format(num_lines))

                    src_toks = src_toks.split()
                    trg_toks = trg_toks.split()
                    src_ids = [src_vocab.get(w, ac.UNK_ID) for w in src_toks]
                    trg_ids = [trg_vocab.get(w, ac.UNK_ID) for w in trg_toks]
                    src_ids = map(str, src_ids)
                    trg_ids = map(str, trg_ids)
                    data = u'{}|||{}\n'.format(u' '.join(src_ids), u' '.join(trg_ids))
                    tokens_f.write(data)

        with open(joint_length_file, 'w', 'utf-8') as f:
            f.write('{}\n'.format(str(num_lines)))

    def process_n_batches(self, n_batches_string_list, mode=ac.TRAINING):
        batch_size = self.batch_size if mode == ac.TRAINING else 1

        src_inputs = []
        src_seq_lengths = []
        trg_inputs = []
        trg_seq_lengths = []

        num_samples = 0
        for line in n_batches_string_list:
            data = line.strip()
            if data:
                num_samples += 1
                data = data.split('|||')
                _src_input = data[0].strip().split()
                _trg_input = data[1].strip().split()

                _src_input = map(int, _src_input)
                _trg_input = map(int, _trg_input)

                if self.reverse:
                    _src_input = _src_input[::-1]
                _trg_input = [ac.BOS_ID] + _trg_input

                _src_len = len(_src_input)
                _trg_len = len(_trg_input)

                src_inputs.append(_src_input)
                src_seq_lengths.append(_src_len)
                trg_inputs.append(_trg_input)
                trg_seq_lengths.append(_trg_len)

        src_inputs = numpy.array(src_inputs)
        src_seq_lengths = numpy.array(src_seq_lengths)
        trg_inputs = numpy.array(trg_inputs)
        trg_seq_lengths = numpy.array(trg_seq_lengths)

        if mode == ac.TRAINING:
            # I find sorting by trg sent lengths is a bit faster but sorting by src sent length
            # yield better result (> 1. BLEU on en2vi data)
            sorted_idxs = numpy.argsort(src_seq_lengths) 
            src_inputs = src_inputs[sorted_idxs]
            src_seq_lengths = src_seq_lengths[sorted_idxs]
            trg_inputs = trg_inputs[sorted_idxs]
            trg_seq_lengths = trg_seq_lengths[sorted_idxs]

        actual_num_batches = num_samples // batch_size

        src_input_batches = []
        src_seq_length_batches = []
        trg_input_batches = []
        trg_target_batches = []
        target_weight_batches = []

        for b in xrange(actual_num_batches):
            s_idx = b * batch_size
            e_idx = (b + 1) * batch_size
            max_src_length = numpy.max(src_seq_lengths[s_idx:e_idx])
            max_trg_length = numpy.max(trg_seq_lengths[s_idx:e_idx])

            src_input_batch = numpy.zeros([batch_size, max_src_length], dtype=numpy.int32)
            src_seq_length_batch = src_seq_lengths[s_idx:e_idx]
            trg_input_batch = numpy.zeros([batch_size, max_trg_length], dtype=numpy.int32)
            trg_target_batch = numpy.zeros([batch_size, max_trg_length], dtype=numpy.int32)
            target_weight_batch = numpy.ones([batch_size, max_trg_length], dtype=numpy.float32)

            for i in xrange(s_idx, e_idx):
                src_input_batch[i-s_idx] = src_inputs[i] + (max_src_length - src_seq_lengths[i]) * [ac.PAD_ID]
                trg_input_batch[i-s_idx] = trg_inputs[i] + (max_trg_length - trg_seq_lengths[i]) * [ac.PAD_ID]
                trg_target_batch[i-s_idx] = trg_inputs[i][1:] + [ac.EOS_ID] + (max_trg_length - trg_seq_lengths[i]) * [ac.PAD_ID] 

            target_weight_batch[trg_target_batch==ac.PAD_ID] = 0.

            src_input_batches.append(src_input_batch)
            src_seq_length_batches.append(src_seq_length_batch)
            trg_input_batches.append(trg_input_batch)
            trg_target_batches.append(trg_target_batch)
            target_weight_batches.append(target_weight_batch)


        return src_input_batches, src_seq_length_batches, trg_input_batches, trg_target_batches, target_weight_batches

    def get_batch(self, mode=ac.TRAINING, num_batches=1000):
        with tf.device('/cpu:0'):
            ids_file = self.ids_files[mode]
            batch_size = self.batch_size if mode == ac.TRAINING else 1
            shuffle = mode == ac.TRAINING
            if shuffle:
                # First we shuffle training data
                start = time.time()
                ut.shuffle_file(ids_file)
                self.logger.info('Shuffling {} takes {} seconds'.format(ids_file, time.time() - start))

            batch_num = 0
            with open(ids_file, 'r', 'utf-8') as f:
                while True:
                    next_n_lines = list(islice(f, num_batches * batch_size))
                    if not next_n_lines:
                        break

                    batches = self.process_n_batches(next_n_lines, mode=mode)
                    for batch_data in izip(*batches):
                        batch_num += 1
                        yield batch_num, batch_data

    def get_trans_input(self, input_file):
        src_vocab, _ = self.init_vocab(self.src_lang)
        with tf.device('/cpu:0'):
            with open(input_file, 'r', 'utf-8') as f:
                for line in f:
                    toks = line.strip().split()
                    if self.reverse:
                        toks = toks[::-1]
                    
                    src_input = [src_vocab.get(w, ac.UNK_ID) for w in toks]
                    src_len = len(src_input)

                    src_input = numpy.array(src_input).reshape([1, -1])
                    yield src_input, [src_len], toks