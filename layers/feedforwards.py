from __future__ import print_function
from __future__ import division

import tensorflow as tf

class FeedForward(object):
    def __init__(self, input_size, output_size, scope, bias=True, activate_func=None):
        super(FeedForward, self).__init__()
        
        self.bias = bias

        with tf.variable_scope(scope):
            self.W = tf.get_variable('W', [input_size, output_size])
            if bias:
                self.b = tf.get_variable('b', [output_size])

        self.activate_func = activate_func

    def transform(self, inputs):
        outputs = tf.matmul(inputs, self.W)
        if self.bias:
            outputs += self.b
            
        if self.activate_func:
            outputs = self.activate_func(outputs)

        return outputs
        