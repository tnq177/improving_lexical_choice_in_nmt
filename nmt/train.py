from __future__ import print_function
from __future__ import division

import os
import time
import argparse
import numpy
import tensorflow as tf
from os.path import join
from os.path import exists

import nmt.all_constants as ac
import nmt.utils as ut
from nmt.model import Model
from nmt.data_manager import DataManager
import nmt.configurations as configurations
from nmt.validator import Validator


class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.config = getattr(configurations, args.proto)()
        self.num_preload = args.num_preload
        self.logger = ut.get_logger(self.config['log_file'])

        self.lr = self.config['lr']
        self.max_epochs = self.config['max_epochs']
        self.cpkt_path = None
        self.validate_freq = None
        self.train_perps = []

        self.saver = None
        self.train_m = None
        self.dev_m = None

        self.data_manager = DataManager(self.config)
        self.validator = Validator(self.config, self.data_manager)
        self.validate_freq = ut.get_validation_frequency(self.data_manager.length_files[ac.TRAINING],
                                                        self.config['validate_freq'],
                                                        self.config['batch_size'])
        self.logger.info('Evaluate every {} batches'.format(self.validate_freq))

        _, self.src_ivocab = self.data_manager.init_vocab(self.data_manager.src_lang)
        _, self.trg_ivocab = self.data_manager.init_vocab(self.data_manager.trg_lang)

        # For logging
        self.log_freq = 100  # log train stat every this-many batches
        self.log_train_loss = 0. # total train loss every log_freq batches
        self.log_train_weights = 0.
        self.num_batches_done = 0 # number of batches done for the whole training
        self.epoch_batches_done = 0 # number of batches done for this epoch
        self.epoch_loss = 0. # total train loss for whole epoch
        self.epoch_weights = 0. # total train weights (# target words) for whole epoch
        self.epoch_time = 0. # total exec time for whole epoch, sounds like that tabloid

    def get_model(self, mode):
        reuse = mode != ac.TRAINING
        d = self.config['init_range']
        initializer = tf.random_uniform_initializer(-d, d, seed=ac.SEED)
        with tf.variable_scope(self.config['model_name'], reuse=reuse, initializer=initializer):
            return Model(self.config, mode)

    def reload_and_get_cpkt_saver(self, config, sess):
        cpkt_path = join(config['save_to'], '{}.cpkt'.format(config['model_name']))
        saver = tf.train.Saver(var_list=tf.trainable_variables(),
                               max_to_keep=config['n_best'] + 1)

        # TF some version >= 0.11, no longer .cpkt so check for meta file instead
        if exists(cpkt_path + '.meta') and config['reload']:
            self.logger.info('Reload model from {}'.format(cpkt_path))
            saver.restore(sess, cpkt_path)

        self.cpkt_path = cpkt_path
        self.saver = saver

    def train(self):
        tf.reset_default_graph()
        with tf.Graph().as_default():
            tf.set_random_seed(ac.SEED)
            self.train_m = self.get_model(ac.TRAINING)
            self.dev_m = self.get_model(ac.VALIDATING)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                self.reload_and_get_cpkt_saver(self.config, sess)
                self.logger.info('Set learning rate to {}'.format(self.lr))
                sess.run(tf.assign(self.train_m.lr, self.lr))

                # Initially normalize src embeddings to embed_norm
                sess.run(self.train_m.normalize_src_embeds)

                for e in xrange(self.max_epochs):
                    for b, batch_data in self.data_manager.get_batch(mode=ac.TRAINING, num_batches=self.num_preload):
                        self.run_and_log(sess, b, e, batch_data)
                        self.maybe_validate(sess)

                    self.save_checkpoint(sess)
                    self.report_epoch(e)

                self.save_checkpoint(sess)
                self.logger.info('It is finally done, mate!')
                self.logger.info('Train perplexities:')
                self.logger.info(', '.join(map(str, self.train_perps)))
                numpy.save(join(self.config['save_to'], 'train_perps.npy'), self.train_perps)

                self.logger.info('Save final checkpoint')
                self.saver.save(sess, self.cpkt_path)

                # Evaluate on test
                test_file = self.data_manager.data_files[ac.TESTING][self.data_manager.src_lang]
                if exists(test_file):
                    self.logger.info('Evaluate on test')
                    best_bleu = numpy.max(self.validator.best_bleus)
                    best_cpkt_path = self.validator.get_cpkt_path(best_bleu)
                    self.logger.info('Restore best cpkt from {}'.format(best_cpkt_path))
                    self.saver.restore(sess, best_cpkt_path)
                    self.validator.translate(sess, self.dev_m, test_file, unk_repl=True)

    def report_epoch(self, e):
        self.logger.info('Finish epoch {}'.format(e + 1))
        self.logger.info('    It takes {}'.format(ut.format_seconds(self.epoch_time)))
        self.logger.info('    Avergage # words/second    {}'.format(self.epoch_weights / self.epoch_time))
        self.logger.info('    Average seconds/batch    {}'.format(self.epoch_time / self.epoch_batches_done))

        train_perp = self.epoch_loss / self.epoch_weights
 
        self.epoch_batches_done = 0
        self.epoch_time = 0.
        self.epoch_loss = 0.
        self.epoch_weights = 0.

        train_perp = numpy.exp(train_perp) if train_perp < 300 else float('inf')
        self.train_perps.append(train_perp)
        self.logger.info('    train perplexity: {}'.format(train_perp))

    def sample_input(self, batch_data):
        # TODO: more on sampling
        src_sent = batch_data[0][0]
        src_length = batch_data[1][0]
        trg_sent = batch_data[2][0]
        target = batch_data[3][0]
        weight = batch_data[4][0]

        src_sent = map(self.src_ivocab.get, src_sent)
        src_sent = u' '.join(src_sent)
        trg_sent = map(self.trg_ivocab.get, trg_sent)
        trg_sent = u' '.join(trg_sent)
        target = map(self.trg_ivocab.get, target)
        target = u' '.join(target)
        weight = ' '.join(map(str, weight))

        self.logger.info('Sample input data:')
        self.logger.info(u'Src: {}'.format(src_sent))
        self.logger.info(u'Src len: {}'.format(src_length))
        self.logger.info(u'Trg: {}'.format(trg_sent))
        self.logger.info(u'Tar: {}'.format(target))
        self.logger.info(u'W: {}'.format(weight))

    def run_and_log(self, sess, b, e, batch_data):
        start = time.time()
        src_inputs, src_seq_lengths, trg_inputs, trg_targets, target_weights = batch_data
        feed = {
            self.train_m.src_inputs: src_inputs,
            self.train_m.src_seq_lengths: src_seq_lengths,
            self.train_m.trg_inputs: trg_inputs,
            self.train_m.trg_targets: trg_targets,
            self.train_m.target_weights: target_weights
        }
        loss, _ = sess.run([self.train_m.loss, self.train_m.train_op], feed)

        num_trg_words = numpy.sum(target_weights)
        self.num_batches_done += 1
        self.epoch_batches_done += 1
        self.epoch_loss += loss
        self.epoch_weights += num_trg_words
        self.log_train_loss += loss
        self.log_train_weights += num_trg_words
        self.epoch_time += time.time() - start


        if self.num_batches_done % (10 * self.log_freq) == 0: 
            self.sample_input(batch_data)
            
        if self.num_batches_done % self.log_freq == 0:
            acc_speed_word = self.epoch_weights / self.epoch_time
            acc_speed_time = self.epoch_time / self.epoch_batches_done

            avg_word_perp = self.log_train_loss / self.log_train_weights
            avg_word_perp = numpy.exp(avg_word_perp) if avg_word_perp < 300 else float('inf')
            self.log_train_loss = 0.
            self.log_train_weights = 0.

            self.logger.info('Batch {}, epoch {}/{}:'.format(b, e + 1, self.max_epochs))
            self.logger.info('   avg word perp:   {0:.2f}'.format(avg_word_perp))
            self.logger.info('   acc trg words/s: {}'.format(int(acc_speed_word)))
            self.logger.info('   acc sec/batch:   {0:.2f}'.format(acc_speed_time))

    def maybe_validate(self, sess):
        if self.num_batches_done % self.validate_freq == 0:
            self.save_checkpoint(sess)
            self.validator.validate_and_save(sess, self.dev_m, self.saver)

    def save_checkpoint(self, sess):
        start = time.time()
        self.saver.save(sess, self.cpkt_path)
        self.logger.info('Save model to {}, takes {}'.format(self.cpkt_path, ut.format_seconds(time.time() - start)))

    def _call_me_maybe(self):
        pass # NO