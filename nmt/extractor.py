from __future__ import print_function
from __future__ import division

import os
import time
import operator
from itertools import izip
from codecs import open

import numpy
import tensorflow as tf

import nmt.all_constants as ac
import nmt.utils as ut
from nmt.model import Model
import nmt.configurations as configurations

class Extractor(object):
    def __init__(self, args):
        super(Extractor, self).__init__()
        config = getattr(configurations, args.proto)()
        self.logger = ut.get_logger(config['log_file'])
        self.model_file = args.model_file

        var_list = args.var_list
        save_to = args.save_to

        if var_list is None:
            raise ValueError('Empty var list')

        if self.model_file is None or not os.path.exists(self.model_file + '.meta'):
            raise ValueError('Input file or model file does not exist')

        if not os.path.exists(save_to):
            os.makedirs(save_to)

        self.logger.info('Extracting these vars: {}'.format(', '.join(var_list)))

        with tf.Graph().as_default(), tf.Session() as sess:
            d = config['init_range']
            initializer = tf.random_uniform_initializer(-d, d)
            with tf.variable_scope(config['model_name'], reuse=False, initializer=initializer):
                model = Model(config, ac.TRAINING)

            saver = tf.train.Saver(var_list=tf.trainable_variables())
            saver.restore(sess, self.model_file)

            var_values = operator.attrgetter(*var_list)(model)
            var_values = sess.run(var_values)

            if len(var_list) == 1:
                var_values = [var_values]
                
            for var, var_value in izip(var_list, var_values):
                var_path = os.path.join(save_to, var + '.npy')
                numpy.save(var_path, var_value)
