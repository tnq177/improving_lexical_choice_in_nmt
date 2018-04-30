from __future__ import print_function
from __future__ import division

import os
import time
from itertools import izip
from codecs import open

import numpy
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import nmt.all_constants as ac
import nmt.utils as ut
from nmt.model import Model
from nmt.data_manager import DataManager
import nmt.configurations as configurations

class Translator(object):
    def __init__(self, args):
        super(Translator, self).__init__()
        self.config = getattr(configurations, args.proto)()
        self.reverse = self.config['reverse']
        self.logger = ut.get_logger(self.config['log_file'])

        self.input_file = args.input_file
        self.model_file = args.model_file
        self.plot_align = args.plot_align
        self.unk_repl   = args.unk_repl
        self.lex_table_path = args.lexical_file

        if self.input_file is None or self.model_file is None or not os.path.exists(self.input_file) or not os.path.exists(self.model_file + '.meta'):
            raise ValueError('Input file or model file does not exist')

        self.data_manager = DataManager(self.config)
        _, self.src_ivocab = self.data_manager.init_vocab(self.data_manager.src_lang)
        _, self.trg_ivocab = self.data_manager.init_vocab(self.data_manager.trg_lang)
        self.translate()

    def ids_to_trans(self, trans_ids, trans_alignments, no_unk_src_toks):
        words = []
        word_ids = []
        # Could have done better but this is clearer to me

        if not self.unk_repl:
            for idx, word_idx in enumerate(trans_ids):
                if word_idx == ac.EOS_ID:
                    break
                words.append(self.trg_ivocab[word_idx])
                word_ids.append(word_idx)
        else:
            for idx, word_idx in enumerate(trans_ids):
                if word_idx == ac.EOS_ID:
                    break
                    
                if word_idx == ac.UNK_ID:
                    # Replace UNK with higest attention source words
                    alignment = trans_alignments[idx]
                    highest_att_src_tok_pos = numpy.argmax(alignment)
                    words.append(no_unk_src_toks[highest_att_src_tok_pos])
                else:
                    words.append(self.trg_ivocab[word_idx])
                word_ids.append(word_idx)


        return u' '.join(words), word_ids

    def get_model(self, mode):
        reuse = mode != ac.TRAINING
        d = self.config['init_range']
        initializer = tf.random_uniform_initializer(-d, d)
        with tf.variable_scope(self.config['model_name'], reuse=reuse, initializer=initializer):
            return Model(self.config, mode, self.lex_table_path)

    def get_trans(self, probs, scores, symbols, parents, alignments, no_unk_src_toks):
        sorted_rows = numpy.argsort(scores[:, -1])[::-1]
        best_trans_alignments = []
        best_trans = None
        best_tran_ids = None
        beam_trans = []
        for i, r in enumerate(sorted_rows):
            row_idx = r
            col_idx = scores.shape[1] - 1

            trans_ids = []
            trans_alignments = []
            while True:
                if col_idx < 0:
                    break

                trans_ids.append(symbols[row_idx, col_idx])
                align = alignments[row_idx, col_idx, :]
                trans_alignments.append(align)

                if i == 0:
                    best_trans_alignments.append(align if not self.reverse else align[::-1])

                row_idx = parents[row_idx, col_idx]
                col_idx -= 1

            trans_ids = trans_ids[::-1]
            trans_alignments = trans_alignments[::-1]
            trans_out, trans_out_ids = self.ids_to_trans(trans_ids, trans_alignments, no_unk_src_toks)
            beam_trans.append(u'{} {:.2f} {:.2f}'.format(trans_out, scores[r, -1], probs[r, -1]))
            if i == 0: # highest prob trans
                best_trans = trans_out
                best_tran_ids = trans_out_ids

        return best_trans, best_tran_ids, u'\n'.join(beam_trans), best_trans_alignments[::-1]

    def plot_head_map(self, mma, target_labels, target_ids, source_labels, source_ids, filename):
        """https://github.com/EdinburghNLP/nematus/blob/master/utils/plot_heatmap.py
        Change the font in family param below. If the system font is not used, delete matplotlib 
        font cache https://github.com/matplotlib/matplotlib/issues/3590
        """
        fig, ax = plt.subplots()
        heatmap = ax.pcolor(mma, cmap=plt.cm.Blues)

        # put the major ticks at the middle of each cell
        ax.set_xticks(numpy.arange(mma.shape[1]) + 0.5, minor=False)
        ax.set_yticks(numpy.arange(mma.shape[0]) + 0.5, minor=False)

        # without this I get some extra columns rows
        # http://stackoverflow.com/questions/31601351/why-does-this-matplotlib-heatmap-have-an-extra-blank-column
        ax.set_xlim(0, int(mma.shape[1]))
        ax.set_ylim(0, int(mma.shape[0]))

        # want a more natural, table-like display
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        # source words -> column labels
        ax.set_xticklabels(source_labels, minor=False, family='Source Code Pro')
        for xtick, idx in zip(ax.get_xticklabels(), source_ids):
            if idx == ac.UNK_ID:
                xtick.set_color('b')
        # target words -> row labels
        ax.set_yticklabels(target_labels, minor=False, family='Source Code Pro')
        for ytick, idx in zip(ax.get_yticklabels(), target_ids):
            if idx == ac.UNK_ID:
                ytick.set_color('b')

        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')

    def translate(self):
        with tf.Graph().as_default():
            train_model = self.get_model(ac.TRAINING)
            model = self.get_model(ac.VALIDATING)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                self.logger.info('Restore model from {}'.format(self.model_file))
                saver = tf.train.Saver(var_list=tf.trainable_variables())
                saver.restore(sess, self.model_file)

                best_trans_file = self.input_file + '.best_trans'
                beam_trans_file = self.input_file + '.beam_trans'
                open(best_trans_file, 'w').close()
                open(beam_trans_file, 'w').close()
                ftrans = open(best_trans_file, 'w', 'utf-8')
                btrans = open(beam_trans_file, 'w', 'utf-8')

                self.logger.info('Start translating {}'.format(self.input_file))
                start = time.time()
                count = 0
                for (src_input, src_seq_len, no_unk_src_toks) in self.data_manager.get_trans_input(self.input_file):
                    feed = {
                        model.src_inputs: src_input,
                        model.src_seq_lengths: src_seq_len
                    }
                    probs, scores, symbols, parents, alignments = sess.run([model.probs, model.scores, model.symbols, model.parents, model.alignments], feed_dict=feed)
                    alignments = numpy.transpose(alignments, axes=(1, 0, 2))

                    probs = numpy.transpose(numpy.array(probs))
                    scores = numpy.transpose(numpy.array(scores))
                    symbols = numpy.transpose(numpy.array(symbols))
                    parents = numpy.transpose(numpy.array(parents))

                    best_trans, best_trans_ids, beam_trans, best_trans_alignments = self.get_trans(probs, scores, symbols, parents, alignments, no_unk_src_toks)
                    ftrans.write(best_trans + '\n')
                    btrans.write(beam_trans + '\n\n')

                    if self.plot_align:
                        src_input = numpy.reshape(src_input, [-1])
                        if self.reverse:
                            src_input = src_input[::-1]
                            no_unk_src_toks = no_unk_src_toks[::-1]
                        trans_toks = best_trans.split()
                        best_trans_alignments = numpy.array(best_trans_alignments)[:len(trans_toks)]
                        filename = '{}_{}.png'.format(self.input_file, count)

                        self.plot_head_map(best_trans_alignments, trans_toks, best_trans_ids, no_unk_src_toks, src_input, filename)

                    count += 1
                    if count % 100 == 0:
                        self.logger.info('  Translating line {}, average {} seconds/sent'.format(count, (time.time() - start) / count))

                ftrans.close()
                btrans.close()

                self.logger.info('Done translating {}, it takes {} minutes'.format(self.input_file, float(time.time() - start) / 60.0))


