from __future__ import print_function
from __future__ import division

import tensorflow as tf
from feedforwards import FeedForward

class Attention(object):
    DOT = 0
    GEN = 1
    BAH = 2

    def __init__(self, scope, score_func_type, h_ss, src_size, trg_size, common_dim=None):
        super(Attention, self).__init__()
        self.h_ss = h_ss
        self.score_func_type = score_func_type
        self.src_size = src_size
        self.trg_size = trg_size

        if score_func_type == self.DOT:
            if src_size != trg_size:
                raise ValueError('To use DOT attention, src_size == trg_size')

        if score_func_type == self.GEN:
            self.feed_forward = FeedForward(trg_size, src_size, scope)

        if score_func_type == self.BAH:
            if common_dim is None:
                raise ValueError('To use Bahdanau attention, common_dim != None')

            self.common_dim = common_dim
            self.src_ff = FeedForward(src_size, common_dim, scope + '_src', bias=False)
            self.trg_ff = FeedForward(trg_size, common_dim, scope + '_trg', bias=False)
            with tf.variable_scope(scope):
                self.v_T = tf.get_variable('v_T', [common_dim, 1])


    def calc_context(self, sequence_length, h_t):
        shape = tf.shape(self.h_ss)
        batch_size = shape[0]
        num_steps = shape[1]

        _h_t = h_t
        if self.score_func_type == self.GEN:
            _h_t = self.feed_forward.transform(h_t)

        if self.score_func_type == self.GEN or self.score_func_type == self.DOT:
            _h_t = tf.reshape(_h_t, [batch_size, self.src_size, 1])
            scores = tf.matmul(self.h_ss, _h_t)
        else:
            src_part = tf.reshape(self.h_ss, [-1, self.src_size])
            src_part = self.src_ff.transform(src_part)
            src_part = tf.reshape(src_part, [batch_size, num_steps, self.common_dim])
            trg_part = self.trg_ff.transform(h_t) # [batch_size, common_dim]
            trg_part = tf.reshape(trg_part, [batch_size, 1, -1])
            scores = src_part + trg_part # [batch_size, num_steps, common_dim]
            scores = tf.tanh(scores)
            scores = tf.reshape(scores, [-1, self.common_dim])
            scores = tf.matmul(scores, self.v_T)

        mask = tf.sequence_mask(sequence_length, num_steps)
        mask = tf.cast(mask, tf.float32)
        scores = tf.reshape(scores, [batch_size, num_steps])
        scores = mask * scores + (1.0 - mask) * tf.float32.min

        alignments = tf.nn.softmax(scores)

        c_t = tf.multiply(tf.reshape(alignments, [batch_size, num_steps, 1]), self.h_ss)
        c_t = tf.reduce_sum(c_t, 1)

        return alignments, c_t
        