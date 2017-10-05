from __future__ import print_function
from __future__ import division

import tensorflow as tf
from layers import FeedForward

class Softmax(object):
    def __init__(self, num_features, num_classes, scope):
        super(Softmax, self).__init__()
        
        self.logit_layer = FeedForward(num_features, num_classes, scope)

    def calc_logits(self, inputs):
        return self.logit_layer.transform(inputs)

    def calc_logprobs(self, inputs):
        logits = self.calc_logits(inputs)
        return tf.nn.log_softmax(logits)

    def calc_probs(self, inputs):
        logits = self.calc_logits(inputs)
        return tf.nn.softmax(logits)