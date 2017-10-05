from __future__ import print_function
from __future__ import division

import os
import logging
from datetime import timedelta

import tensorflow as tf
import subprocess


def get_logger(logfile=None):
    _logfile = logfile if logfile else './DEBUG.log'
    """Global logger for every logging"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s:%(filename)s:%(lineno)s - %(funcName)20s(): %(message)s')

    if not logger.handlers:
        debug_handler = logging.FileHandler(_logfile)
        debug_handler.setFormatter(formatter)
        debug_handler.setLevel(logging.DEBUG)
        logger.addHandler(debug_handler)

    return logger


def get_lstm_cell(scope, num_layers, rnn_size, output_keep_prob=1.0, seed=42, reuse=False):
    def get_cell(_rnn_size, _output_keep_prob, _reuse, _seed):
        cell = tf.contrib.rnn.LSTMCell(_rnn_size, state_is_tuple=True, reuse=_reuse)
        return tf.contrib.rnn.DropoutWrapper(
            cell,
            output_keep_prob=_output_keep_prob,
            dtype=tf.float32,
            seed=_seed)

    with tf.variable_scope(scope):
        if num_layers <= 1:
            return get_cell(rnn_size, output_keep_prob, reuse, seed)
        else:
            cells = []
            for i in xrange(num_layers):
                cell = get_cell(rnn_size, output_keep_prob, reuse, seed)
                cells.append(cell)

            return tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)


def tensor_to_lstm_state(state_tensor, num_layers):
    if num_layers == 1:
        return tf.contrib.rnn.LSTMStateTuple(state_tensor[0, :, :], state_tensor[1, :, :])
    else:
        state_list = []
        for k in xrange(num_layers):
            c_m = state_tensor[k, :, :, :]
            state_list.append(tf.contrib.rnn.LSTMStateTuple(c_m[0, :, :], c_m[1, :, :]))

        return tuple(state_list)


def shuffle_file(input_file):
    shuffled_file = input_file + '.shuf'
    commands = 'bash ./scripts/shuffle_file.sh {} {}'.format(input_file, shuffled_file)
    subprocess.check_call(commands, shell=True)
    subprocess.check_call('mv {} {}'.format(shuffled_file, input_file), shell=True)


def get_validation_frequency(train_length_file, val_frequency, batch_size):
    if val_frequency > 1.0:
        return val_frequency
    else:
        with open(train_length_file) as f:
            line = f.readline().strip()
            num_train_sents = int(line)

        return int((num_train_sents * val_frequency) // batch_size)

def format_seconds(seconds):
    return str(timedelta(seconds=seconds))