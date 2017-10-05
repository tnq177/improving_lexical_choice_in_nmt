from __future__ import print_function
from __future__ import division

import re

SCORE_FUNC_DOT    = 0
SCORE_FUNC_GEN    = 1
SCORE_FUNC_BAH    = 2

LSTM = 0
# Todo: allow using BILSTM or GRU
BILSTM = 1
GRU = 2

# Special vocabulary symbols - we always put them at the start.
_PAD = b'_PAD'
_BOS = b'_BOS'
_EOS = b'_EOS'
_UNK = b'_UNK'
_START_VOCAB = [_PAD, _BOS, _EOS, _UNK]

PAD_ID = 0
BOS_ID = 1
# It's crucial that EOS_ID != 0 (see beam_search decoder)
EOS_ID = 2
UNK_ID = 3

_DIGIT_RE = re.compile(br'\d')

TRAINING = 'training'
VALIDATING = 'validating'
TESTING = 'testing'

SGD = 0
ADADELTA = 1

SEED = 42
