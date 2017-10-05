from __future__ import print_function
from __future__ import division

from functools import partial

import tensorflow as tf
from layers import Attention, FeedForward, Encoder, Softmax
from tensorflow.contrib.seq2seq import sequence_loss
import nmt.all_constants as ac
import nmt.utils as ut


class Model(object):
    def __init__(self, config, mode):
        super(Model, self).__init__()
        self.logger = ut.get_logger(config['log_file'])

        ENC_SCOPE = 'encoder'
        DEC_SCOPE = 'decoder'
        ATT_SCOPE = 'attention'
        OUT_SCOPE = 'outputer'
        SFM_SCOPE = 'softmax'

        batch_size = config['batch_size']
        feed_input = config['feed_input']
        grad_clip = config['grad_clip']
        beam_size = config['beam_size']
        beam_alpha = config['beam_alpha']
        num_layers = config['num_layers']
        rnn_type = config['rnn_type']
        score_func_type = config['score_func_type']

        src_vocab_size = config['src_vocab_size']
        trg_vocab_size = config['trg_vocab_size']
        
        src_embed_size = config['src_embed_size']
        trg_embed_size = config['trg_embed_size']

        enc_rnn_size = config['enc_rnn_size']
        dec_rnn_size = config['dec_rnn_size']

        input_keep_prob = config['input_keep_prob']
        output_keep_prob = config['output_keep_prob']

        attention_maps = {
            ac.SCORE_FUNC_DOT: Attention.DOT,
            ac.SCORE_FUNC_GEN: Attention.GEN,
            ac.SCORE_FUNC_BAH: Attention.BAH
        }
        score_func_type = attention_maps[score_func_type]

        if mode != ac.TRAINING:
            batch_size = 1
            input_keep_prob = 1.0
            output_keep_prob = 1.0

        # Placeholder
        self.src_inputs         = tf.placeholder(tf.int32, [batch_size, None])
        self.src_seq_lengths    = tf.placeholder(tf.int32, [batch_size])
        self.trg_inputs         = tf.placeholder(tf.int32, [batch_size, None])
        self.trg_targets        = tf.placeholder(tf.int32, [batch_size, None])
        self.target_weights     = tf.placeholder(tf.float32, [batch_size, None])

        # First, define the src/trg embeddings
        with tf.variable_scope(ENC_SCOPE):
            self.src_embedding = tf.get_variable('embedding',
                                            shape=[src_vocab_size, src_embed_size],
                                            dtype=tf.float32)
        with tf.variable_scope(DEC_SCOPE):
            self.trg_embedding = tf.get_variable('embedding',
                                            shape=[trg_vocab_size, trg_embed_size],
                                            dtype=tf.float32)

        # Then select the RNN cell, reuse if not in TRAINING mode
        if rnn_type != ac.LSTM:
            raise NotImplementedError

        reuse = mode != ac.TRAINING # if dev/test, reuse cell
        encoder_cell = ut.get_lstm_cell(ENC_SCOPE, num_layers, enc_rnn_size, output_keep_prob=output_keep_prob, seed=ac.SEED, reuse=reuse)

        att_state_size = dec_rnn_size
        decoder_cell = ut.get_lstm_cell(DEC_SCOPE, num_layers, dec_rnn_size, output_keep_prob=output_keep_prob, seed=ac.SEED, reuse=reuse)

        # The model
        encoder = Encoder(encoder_cell, ENC_SCOPE)
        decoder = Encoder(decoder_cell, DEC_SCOPE)
        outputer = FeedForward(enc_rnn_size + dec_rnn_size, att_state_size, OUT_SCOPE, activate_func=tf.tanh)
        self.softmax = softmax = Softmax(att_state_size, trg_vocab_size, SFM_SCOPE)

        # Encode source sentence
        encoder_inputs = tf.nn.embedding_lookup(self.src_embedding, self.src_inputs)
        encoder_inputs = tf.nn.dropout(encoder_inputs, input_keep_prob, seed=ac.SEED)
        encoder_outputs, last_state = encoder.encode(encoder_inputs,
                                                     sequence_length=self.src_seq_lengths,
                                                     initial_state=None)
        # Define an attention layer over encoder outputs
        attention = Attention(ATT_SCOPE, score_func_type, encoder_outputs, enc_rnn_size, dec_rnn_size, common_dim=enc_rnn_size if score_func_type==Attention.BAH else None)

        # This function takes an decoder's output, make it attend to encoder's outputs and 
        # spit out the attentional state which is used for predicting next target word
        def decoder_output_func(h_t):
            alignments, c_t = attention.calc_context(self.src_seq_lengths, h_t)
            c_t_h_t = tf.concat([c_t, h_t], 1)
            output = outputer.transform(c_t_h_t)
            return output, alignments


        # Fit everything in the decoder & start decoding
        decoder_inputs = tf.nn.embedding_lookup(self.trg_embedding, self.trg_inputs)
        decoder_inputs = tf.nn.dropout(decoder_inputs, input_keep_prob, seed=ac.SEED)
        attentional_outputs = decoder.decode(decoder_inputs,
                                             decoder_output_func, att_state_size,
                                             feed_input=feed_input, initial_state=last_state,
                                             reuse=False)
        attentional_outputs = tf.reshape(attentional_outputs, [-1, att_state_size])

        # Loss
        logits = softmax.calc_logits(attentional_outputs)
        logits = tf.reshape(logits, [batch_size, -1, trg_vocab_size])
        loss = sequence_loss(logits,
                             self.trg_targets,
                             self.target_weights,
                             average_across_timesteps=False,
                             average_across_batch=False)

        if mode != ac.TRAINING:
            self.loss = tf.stop_gradient(tf.reduce_sum(loss))

            max_output_length = 3 * self.src_seq_lengths[0]
            tensor_to_state = partial(ut.tensor_to_lstm_state, num_layers=config['num_layers'])
            beam_outputs = decoder.beam_decode(self.trg_embedding, ac.BOS_ID, ac.EOS_ID,
                                               decoder_output_func, att_state_size,
                                               softmax.calc_logprobs, trg_vocab_size,
                                               max_output_length, tensor_to_state,
                                               alpha=beam_alpha, beam_size=beam_size, feed_input=feed_input,
                                               initial_state=last_state, reuse=True)
            self.probs, self.scores, self.symbols, self.parents, self.alignments = beam_outputs

        # If in training, do the grad backpropagate
        if mode == ac.TRAINING:
            self.loss = tf.reduce_sum(loss)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), grad_clip)
            self.lr = tf.Variable(1.0, trainable=False, name='lr')
            if config['optimizer'] == ac.ADADELTA:
                optimizer = tf.train.AdadeltaOptimizer(
                    learning_rate=self.lr, rho=0.95, epsilon=1e-6)
            else:
                optimizer = tf.train.GradientDescentOptimizer(self.lr)

            self.train_op = optimizer.apply_gradients(zip(grads, tvars))


        # Finally, log out some model's stats
        if mode == ac.TRAINING:
            def num_params(var):
                shape = var.get_shape().as_list()
                var_count = 1
                for dim in shape:
                    var_count = var_count * dim

                return var_count

            self.logger.info('{} model:'.format('train' if mode == ac.TRAINING else 'dev/test'))
            self.logger.info('Num trainable variables {}'.format(len(tvars)))
            self.logger.info('Num params: {:,}'.format(sum([num_params(v) for v in tvars])))
            self.logger.info('List of all trainable parameters:')
            for v in tvars:
                self.logger.info('   {}'.format(v.name))

