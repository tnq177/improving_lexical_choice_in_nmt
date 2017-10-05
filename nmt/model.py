from __future__ import print_function
from __future__ import division

from functools import partial

import tensorflow as tf
from layers import Attention, FeedForward, Encoder
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
        LEX_SCOPE = 'lexical'

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
        embed_norm = config['embed_norm']

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
        self.src_embedding = tf.get_variable('src_embedding',
                                        shape=[src_vocab_size, src_embed_size],
                                        dtype=tf.float32)
        self.trg_embedding = tf.get_variable('trg_embedding',
                                        shape=[trg_vocab_size, trg_embed_size],
                                        dtype=tf.float32)
        self.sm_bias = tf.get_variable('sm_bias',
                                        shape=[trg_vocab_size],
                                        dtype=tf.float32)
        self.lex_embedding = tf.get_variable('lex_embedding',
                                        shape=[trg_vocab_size, trg_embed_size],
                                        dtype=tf.float32)
        self.lex_bias = tf.get_variable('lex_bias',
                                        shape=[trg_vocab_size],
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
        lex_hider = FeedForward(src_embed_size, att_state_size, LEX_SCOPE, activate_func=tf.tanh)

        # Encode source sentence
        encoder_inputs = tf.nn.embedding_lookup(self.src_embedding, self.src_inputs)
        encoder_inputs = tf.nn.dropout(encoder_inputs, input_keep_prob, seed=ac.SEED)
        encoder_outputs, last_state = encoder.encode(encoder_inputs,
                                                     sequence_length=self.src_seq_lengths,
                                                     initial_state=None)
        # Define an attention layer over encoder outputs
        attention = Attention(ATT_SCOPE, score_func_type, encoder_outputs, enc_rnn_size, dec_rnn_size, common_dim=enc_rnn_size if score_func_type==Attention.BAH else None)

        def project_embeds(x, axis=1):
            return embed_norm * tf.nn.l2_normalize(x, axis)

        def decoder_output_func(h_t):
            alignments, c_t = attention.calc_context(self.src_seq_lengths, h_t)
            c_t_h_t = tf.concat([c_t, h_t], 1)
            output = outputer.transform(c_t_h_t)

            c_embed = tf.multiply(tf.reshape(alignments, [batch_size, -1, 1]), encoder_inputs)
            c_embed = tf.reduce_sum(c_embed, 1)
            c_embed = tf.tanh(c_embed)

            return output, c_embed, alignments

        def logit_func(att_output, c_embed):
            _att_output = tf.reshape(att_output, [-1, att_state_size])
            _att_output = project_embeds(_att_output)
            _nmt_logit = tf.matmul(_att_output, self.trg_embedding, transpose_b=True) + self.sm_bias

            _c_embed = tf.reshape(c_embed, [-1, att_state_size])
            _c_embed = lex_hider.transform(_c_embed) + _c_embed
            _c_embed = tf.nn.dropout(_c_embed, input_keep_prob, seed=ac.SEED)
            _c_embed = project_embeds(_c_embed)
            _lex_logit = tf.matmul(_c_embed, self.lex_embedding, transpose_b=True) + self.lex_bias

            return _nmt_logit + _lex_logit 

        # Fit everything in the decoder & start decoding
        decoder_inputs = tf.nn.embedding_lookup(self.trg_embedding, self.trg_inputs)
        decoder_inputs = tf.nn.dropout(decoder_inputs, input_keep_prob, seed=ac.SEED)
        att_outputs, c_embeds = decoder.decode(decoder_inputs, decoder_output_func, 
                                             att_state_size, feed_input=feed_input, 
                                             initial_state=last_state, reuse=False)

        # Loss
        logits = logit_func(att_outputs, c_embeds)
        logits = tf.reshape(logits, [batch_size, -1, trg_vocab_size])
        loss = sequence_loss(logits,
                             self.trg_targets,
                             self.target_weights,
                             average_across_timesteps=False,
                             average_across_batch=False)

        # Lexions
        lex_inputs = tf.tanh(self.src_embedding)
        lexicons = lex_hider.transform(lex_inputs) + lex_inputs
        lexicons = project_embeds(lexicons)
        lexicons = tf.matmul(lexicons, self.lex_embedding, transpose_b=True) + self.lex_bias
        self.lexicons = tf.nn.softmax(lexicons)

        if mode != ac.TRAINING:
            self.loss = tf.stop_gradient(tf.reduce_sum(loss))

            max_output_length = 3 * self.src_seq_lengths[0]
            tensor_to_state = partial(ut.tensor_to_lstm_state, num_layers=config['num_layers'])
            beam_outputs = decoder.beam_decode(self.trg_embedding, ac.BOS_ID, ac.EOS_ID,
                                               decoder_output_func, att_state_size,
                                               logit_func, trg_vocab_size,
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

            # We don't normalize src embeds, we just give it initial norm of embed_norm
            self.normalize_src_embeds = tf.assign(self.src_embedding, project_embeds(self.src_embedding, 1))
            # But we do normalize trg + lex embeds every now and then
            self.normalize_trg_embeds = tf.assign(self.trg_embedding, project_embeds(self.trg_embedding, 1))
            self.normalize_lex_embeds = tf.assign(self.lex_embedding, project_embeds(self.lex_embedding, 1))

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

