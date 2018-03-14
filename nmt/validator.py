from __future__ import print_function
from __future__ import division

import os
import re
import time
import shutil
from codecs import open
from subprocess import Popen, PIPE
from os.path import join
from os.path import exists

import numpy

import nmt.utils as ut
import nmt.all_constants as ac


class Validator(object):
    def __init__(self, config, data_manager):
        super(Validator, self).__init__()
        self.logger = ut.get_logger(config['log_file'])
        self.logger.info('Initializing validator')

        self.data_manager = data_manager
        self.restore_segments = config['restore_segments']
        
        def get_cpkt_path(score):
            return join(config['save_to'], '{}-{}.cpkt'.format(config['model_name'], score))

        self.get_cpkt_path = get_cpkt_path
        self.n_best = config['n_best']

        self.bleu_script = './scripts/multi-bleu.perl'
        assert exists(self.bleu_script)

        self.save_to = config['save_to']
        if not exists(self.save_to):
            os.makedirs(self.save_to)
            
        _, self.trg_ivocab = self.data_manager.init_vocab(self.data_manager.trg_lang)

        self.val_trans_out = config['val_trans_out']
        self.val_beam_out = config['val_beam_out']

        self.dev_ref = self.data_manager.data_files[ac.VALIDATING][self.data_manager.trg_lang]
        if self.restore_segments:
            self.dev_ref = self.remove_bpe(self.dev_ref)

        # I'll leave test alone for now since this version of the code doesn't automatically
        # report BLEU on test anw. The reason is it's up to the dataset to use multi-bleu
        # or NIST bleu. I'll include it in the future
        self.test_ref = self.data_manager.data_files[ac.TESTING][self.data_manager.trg_lang]

        self.bleu_curve_path = join(self.save_to, 'bleu_scores.npy')
        self.best_bleus_path = join(self.save_to, 'best_bleu_scores.npy')
        self.bleu_curve = numpy.array([], dtype=numpy.float32)
        self.best_bleus = numpy.array([], dtype=numpy.float32)

        if exists(self.bleu_curve_path):
            self.bleu_curve = numpy.load(self.bleu_curve_path)
        if exists(self.best_bleus_path):
            self.best_bleus = numpy.load(self.best_bleus_path)

    def _ids_to_trans(self, trans_ids):
        words = []
        for idx in trans_ids:
            if idx == ac.EOS_ID:
                break
            words.append(self.trg_ivocab[idx])

        return u' '.join(words)

    def _get_trans(self, probs, scores, symbols, parents):
        sorted_rows = numpy.argsort(scores[:, -1])[::-1]
        best_trans = None
        beam_trans = []
        for i, r in enumerate(sorted_rows):
            row_idx = r
            col_idx = scores.shape[1] - 1

            trans_ids = []
            while True:
                if col_idx < 0:
                    break

                trans_ids.append(symbols[row_idx, col_idx])
                row_idx = parents[row_idx, col_idx]
                col_idx -= 1

            trans_ids = trans_ids[::-1]
            trans_out = self._ids_to_trans(trans_ids)
            beam_trans.append(u'{} {:.2f} {:.2f}'.format(trans_out, scores[r, -1], probs[r, -1]))
            if i == 0: # highest prob trans
                best_trans = trans_out

        return best_trans, u'\n'.join(beam_trans)


    def evaluate(self, sess, dev_m, mode=ac.VALIDATING):
        if mode == ac.VALIDATING:
            val_trans_out = self.val_trans_out
            val_beam_out = self.val_beam_out
            ref_file = self.dev_ref
        elif mode == ac.TESTING:
            test_file = self.data_manager.ids_files[mode]
            if not exists(test_file):
                raise ValueError('{} not found'.format(test_file))

            val_trans_out = test_file + '.trans_out'
            val_beam_out = test_file + '.beam_out'
            ref_file = self.test_ref
        else:
            msg = '...currently this eval function evals on dev/test only'
            self.logger.info(msg)
            raise ValueError(msg)

        open(val_trans_out, 'w').close()
        open(val_beam_out, 'w').close()

        ftrans = open(val_trans_out, 'w', 'utf-8')
        fbeam = open(val_beam_out, 'w', 'utf-8')

        start_time = time.time()
        count = 0
        weights = 0.
        total_loss = 0.
        for b, batch_data in self.data_manager.get_batch(mode=mode):
            src_inputs, src_seq_lengths, trg_inputs, trg_targets, target_weights = batch_data
            feed = {
                dev_m.src_inputs: src_inputs,
                dev_m.src_seq_lengths: src_seq_lengths,
                dev_m.trg_inputs: trg_inputs,
                dev_m.trg_targets: trg_targets,
                dev_m.target_weights: target_weights
            }
            loss, probs, scores, symbols, parents = sess.run([dev_m.loss, dev_m.probs, dev_m.scores, dev_m.symbols, dev_m.parents], feed)
            probs = numpy.transpose(numpy.array(probs))
            scores = numpy.transpose(numpy.array(scores))
            symbols = numpy.transpose(numpy.array(symbols))
            parents = numpy.transpose(numpy.array(parents))
            
            weights += numpy.sum(target_weights)
            total_loss += loss
            best_trans, beam_trans = self._get_trans(probs, scores, symbols, parents)

            ftrans.write(best_trans + '\n')
            fbeam.write(beam_trans + '\n\n')

            if b % 100 == 0:
                self.logger.info('  Translating line {}, average {} seconds/sent'.format(b, (time.time() - start_time) / b))

        ftrans.close()
        fbeam.close()

        perp = total_loss / weights
        perp = numpy.exp(perp) if perp < 300 else float('inf')
        perp = round(perp, ndigits=3)

        self.logger.info('Done translating.')
        self.logger.info('dev perplexity: {}'.format(perp))

        # Remove BPE
        if self.restore_segments:
            val_trans_out = self.remove_bpe(val_trans_out)
            val_beam_out = self.remove_bpe(val_beam_out)

        multibleu_cmd = ['perl', self.bleu_script, ref_file, '<', val_trans_out]
        p = Popen(' '.join(multibleu_cmd), shell=True, stdout=PIPE)
        output, _ = p.communicate()
        out_parse = re.match(r'BLEU = [-.0-9]+', output)
        self.logger.info(output)
        self.logger.info('Validation took: {} minutes'.format(float(time.time() - start_time) / 60.0))

        bleu = float('-inf')
        if out_parse is None:
            msg = '\n    Error extracting BLEU score, out_parse is None'
            msg += '\n    It is highly likely that your model just produces garbage.'
            msg += '\n    Be patient yo, it will get better.'
            self.logger.info(msg)
        else:
            bleu = float(out_parse.group()[6:])

        validation_file = "{}-{}".format(val_trans_out, bleu)
        shutil.copyfile(val_trans_out, validation_file)

        beam_file = "{}-{}".format(val_beam_out, bleu)
        shutil.copyfile(val_beam_out, beam_file)        

        return bleu       

    def _is_valid_to_save(self, bleu_score):
        if len(self.best_bleus) < self.n_best:
            return None, True
        else:
            min_idx = numpy.argmin(self.best_bleus)
            min_bleu = self.best_bleus[min_idx]
            if min_bleu >= bleu_score:
                return None, False
            else:
                return min_idx, True

    def maybe_save(self, sess, saver, bleu_score):
        min_idx, save_please = self._is_valid_to_save(bleu_score)

        if min_idx is not None:
            min_bleu = self.best_bleus[min_idx]
            self.logger.info('Current best bleus: {}'.format(', '.join(map(str, numpy.sort(self.best_bleus)))))
            self.logger.info('Delete {} & use {} instead'.format(min_bleu, bleu_score))
            self.best_bleus = numpy.delete(self.best_bleus, min_idx)

            # Delete the right checkpoint
            cpkt_path = self.get_cpkt_path(min_bleu)
            cpkt_path_data = cpkt_path + '.data-00000-of-00001'
            cpkt_path_meta = cpkt_path + '.meta'
            cpkt_path_index = cpkt_path + '.index'

            if exists(cpkt_path_data):
                self.logger.info('Delete {} & {} & {}'.format(cpkt_path_data, cpkt_path_meta, cpkt_path_index))
                os.remove(cpkt_path_data)
                os.remove(cpkt_path_meta)
                os.remove(cpkt_path_index)

        if save_please:
            self.logger.info('Save {} to list of best bleu scores'.format(bleu_score))
            self.best_bleus = numpy.append(self.best_bleus, bleu_score)
            cpkt_path = self.get_cpkt_path(bleu_score)
            saver.save(sess, cpkt_path)
            self.logger.info('Save new best model to {}'.format(cpkt_path))
            self.logger.info('Best bleu scores so far: {}'.format(', '.join(map(str, numpy.sort(self.best_bleus)))))

        self.bleu_curve = numpy.append(self.bleu_curve, bleu_score)
        numpy.save(self.best_bleus_path, self.best_bleus)
        numpy.save(self.bleu_curve_path, self.bleu_curve)

    def validate_and_save(self, sess, dev_m, saver):
        self.logger.info('Start validation')
        bleu_score = self.evaluate(sess, dev_m, ac.VALIDATING)
        self.maybe_save(sess, saver, bleu_score)

    def remove_bpe(self, infile, outfile=None):
        if not outfile:
            outfile = infile + '.nobpe'

        open(outfile, 'w').close()
        Popen("sed -r 's/(@@ )|(@@ ?$)//g' < {} > {}".format(infile, outfile), shell=True, stdout=PIPE).communicate()
        return outfile

    def translate(self, sess, dev_m, input_file, unk_repl=True):
        # Very redundant
        def _ids_to_trans(trans_ids, trans_alignments, no_unk_src_toks):
            words = []
            word_ids = []
            # Could have done better but this is clearer to me

            if not unk_repl:
                for idx, word_idx in enumerate(trans_ids):
                    words.append(self.trg_ivocab[word_idx])
                    word_ids.append(word_idx)
                    if word_idx == ac.EOS_ID:
                        break
            else:
                for idx, word_idx in enumerate(trans_ids):
                    if word_idx == ac.UNK_ID:
                        # Replace UNK with higest attention source words
                        alignment = trans_alignments[idx]
                        highest_att_src_tok_pos = numpy.argmax(alignment)
                        words.append(no_unk_src_toks[highest_att_src_tok_pos])
                    else:
                        words.append(self.trg_ivocab[word_idx])
                    word_ids.append(word_idx)

                    if word_idx == ac.EOS_ID:
                        break

            return u' '.join(words), word_ids

        def _get_trans(probs, scores, symbols, parents, alignments, no_unk_src_toks):
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
                        best_trans_alignments.append(align if not self.data_manager.reverse else align[::-1])

                    row_idx = parents[row_idx, col_idx]
                    col_idx -= 1

                trans_ids = trans_ids[::-1]
                trans_alignments = trans_alignments[::-1]
                trans_out, trans_out_ids = _ids_to_trans(trans_ids, trans_alignments, no_unk_src_toks)
                beam_trans.append(u'{} {:.2f} {:.2f}'.format(trans_out, scores[r, -1], probs[r, -1]))
                if i == 0: # highest prob trans
                    best_trans = trans_out
                    best_tran_ids = trans_out_ids

            return best_trans, best_tran_ids, u'\n'.join(beam_trans), best_trans_alignments[::-1]

        best_trans_file = input_file + '.best_trans'
        beam_trans_file = input_file + '.beam_trans'
        open(best_trans_file, 'w').close()
        open(beam_trans_file, 'w').close()
        ftrans = open(best_trans_file, 'w', 'utf-8')
        btrans = open(beam_trans_file, 'w', 'utf-8')

        self.logger.info('Start translating {}'.format(input_file))
        start = time.time()
        count = 0
        for (src_input, src_seq_len, no_unk_src_toks) in self.data_manager.get_trans_input(input_file):
            feed = {
                dev_m.src_inputs: src_input,
                dev_m.src_seq_lengths: src_seq_len
            }
            probs, scores, symbols, parents, alignments = sess.run([dev_m.probs, dev_m.scores, dev_m.symbols, dev_m.parents, dev_m.alignments], feed_dict=feed)
            alignments = numpy.transpose(alignments, axes=(1, 0, 2))

            probs = numpy.transpose(numpy.array(probs))
            scores = numpy.transpose(numpy.array(scores))
            symbols = numpy.transpose(numpy.array(symbols))
            parents = numpy.transpose(numpy.array(parents))

            best_trans, best_trans_ids, beam_trans, best_trans_alignments = _get_trans(probs, scores, symbols, parents, alignments, no_unk_src_toks)
            best_trans_wo_eos = best_trans.split()[:-1]
            best_trans_wo_eos = u' '.join(best_trans_wo_eos)
            ftrans.write(best_trans_wo_eos + '\n')
            btrans.write(beam_trans + '\n\n')

            count += 1
            if count % 100 == 0:
                self.logger.info('  Translating line {}, average {} seconds/sent'.format(count, (time.time() - start) / count))

        ftrans.close()
        btrans.close()

        self.logger.info('Done translating {}, it takes {} minutes'.format(input_file, float(time.time() - start) / 60.0))