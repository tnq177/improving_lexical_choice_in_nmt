from __future__ import print_function
from __future__ import division

import os
import nmt.all_constants as ac

def ta2en():
    config = {}

    config['model_name']        = 'ta2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'ta'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/ta2en'
    config['log_file']          = './nmt/DEBUG.log'
    config['rnn_type']          = ac.LSTM
    config['batch_size']        = 32
    config['num_layers']        = 1
    config['enc_rnn_size']      = 512
    config['dec_rnn_size']      = 512
    config['src_embed_size']    = 512
    config['trg_embed_size']    = 512
    config['embed_norm']        = 3.5
    config['max_src_length']    = 50
    config['max_trg_length']    = 50
    config['init_range']        = 0.01
    config['max_epochs']        = 50
    config['lr']                = 1.0
    config['lr_decay']          = 0.5
    config['optimizer']         = ac.ADADELTA
    config['input_keep_prob']   = 0.8
    config['output_keep_prob']  = 0.8
    config['src_vocab_size']    = 4000
    config['trg_vocab_size']    = 3400
    config['grad_clip']         = 5.0
    config['reverse']           = True
    config['score_func_type']   = ac.SCORE_FUNC_GEN
    config['feed_input']        = True
    config['reload']            = True
    config['validate_freq']     = 1.0
    config['save_freq']         = 200
    config['beam_size']         = 12
    config['beam_alpha']        = 0.8
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config

def ur2en():
    config = {}

    config['model_name']        = 'ur2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'ur'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/ur2en'
    config['log_file']          = './nmt/DEBUG.log'
    config['rnn_type']          = ac.LSTM
    config['batch_size']        = 32
    config['num_layers']        = 1
    config['enc_rnn_size']      = 512
    config['dec_rnn_size']      = 512
    config['src_embed_size']    = 512
    config['trg_embed_size']    = 512
    config['embed_norm']        = 3.5
    config['max_src_length']    = 50
    config['max_trg_length']    = 50
    config['init_range']        = 0.01
    config['max_epochs']        = 50
    config['lr']                = 1.0
    config['lr_decay']          = 0.5
    config['optimizer']         = ac.ADADELTA
    config['input_keep_prob']   = 0.8
    config['output_keep_prob']  = 0.8
    config['src_vocab_size']    = 4200
    config['trg_vocab_size']    = 4200
    config['grad_clip']         = 5.0
    config['reverse']           = True
    config['score_func_type']   = ac.SCORE_FUNC_GEN
    config['feed_input']        = True
    config['reload']            = True
    config['validate_freq']     = 1.0
    config['save_freq']         = 200
    config['beam_size']         = 12
    config['beam_alpha']        = 0.8
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config

def ha2en():
    config = {}

    config['model_name']        = 'ha2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'ha'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/ha2en'
    config['log_file']          = './nmt/DEBUG.log'
    config['rnn_type']          = ac.LSTM
    config['batch_size']        = 32
    config['num_layers']        = 2
    config['enc_rnn_size']      = 512
    config['dec_rnn_size']      = 512
    config['src_embed_size']    = 512
    config['trg_embed_size']    = 512
    config['embed_norm']        = 3.5
    config['max_src_length']    = 50
    config['max_trg_length']    = 50
    config['init_range']        = 0.01
    config['max_epochs']        = 50
    config['lr']                = 1.0
    config['lr_decay']          = 0.5
    config['optimizer']         = ac.ADADELTA
    config['input_keep_prob']   = 0.8
    config['output_keep_prob']  = 0.8
    config['src_vocab_size']    = 10600
    config['trg_vocab_size']    = 10400
    config['grad_clip']         = 5.0
    config['reverse']           = True
    config['score_func_type']   = ac.SCORE_FUNC_GEN
    config['feed_input']        = True
    config['reload']            = True
    config['validate_freq']     = 1.0
    config['save_freq']         = 200
    config['beam_size']         = 12
    config['beam_alpha']        = 0.8
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config

def tu2en():
    config = {}

    config['model_name']        = 'tu2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'tu'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/tu2en'
    config['log_file']          = './nmt/DEBUG.log'
    config['rnn_type']          = ac.LSTM
    config['batch_size']        = 32
    config['num_layers']        = 2
    config['enc_rnn_size']      = 512
    config['dec_rnn_size']      = 512
    config['src_embed_size']    = 512
    config['trg_embed_size']    = 512
    config['embed_norm']        = 3.5
    config['max_src_length']    = 50
    config['max_trg_length']    = 50
    config['init_range']        = 0.01
    config['max_epochs']        = 50
    config['lr']                = 1.0
    config['lr_decay']          = 0.5
    config['optimizer']         = ac.ADADELTA
    config['input_keep_prob']   = 0.8
    config['output_keep_prob']  = 0.8
    config['src_vocab_size']    = 21100
    config['trg_vocab_size']    = 13300
    config['grad_clip']         = 5.0
    config['reverse']           = True
    config['score_func_type']   = ac.SCORE_FUNC_GEN
    config['feed_input']        = True
    config['reload']            = True
    config['validate_freq']     = 1.0
    config['save_freq']         = 200
    config['beam_size']         = 12
    config['beam_alpha']        = 0.8
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config

def uz2en():
    config = {}

    config['model_name']        = 'uz2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'uz'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/uz2en'
    config['log_file']          = './nmt/DEBUG.log'
    config['rnn_type']          = ac.LSTM
    config['batch_size']        = 32
    config['num_layers']        = 2
    config['enc_rnn_size']      = 512
    config['dec_rnn_size']      = 512
    config['src_embed_size']    = 512
    config['trg_embed_size']    = 512
    config['embed_norm']        = 3.5
    config['max_src_length']    = 50
    config['max_trg_length']    = 50
    config['init_range']        = 0.01
    config['max_epochs']        = 50
    config['lr']                = 1.0
    config['lr_decay']          = 0.5
    config['optimizer']         = ac.ADADELTA
    config['input_keep_prob']   = 0.8
    config['output_keep_prob']  = 0.8
    config['src_vocab_size']    = 29800
    config['trg_vocab_size']    = 17400
    config['grad_clip']         = 5.0
    config['reverse']           = True
    config['score_func_type']   = ac.SCORE_FUNC_GEN
    config['feed_input']        = True
    config['reload']            = True
    config['validate_freq']     = 1.0
    config['save_freq']         = 200
    config['beam_size']         = 12
    config['beam_alpha']        = 0.8
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config

def hu2en():
    config = {}

    config['model_name']        = 'hu2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'hu'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/hu2en'
    config['log_file']          = './nmt/DEBUG.log'
    config['rnn_type']          = ac.LSTM
    config['batch_size']        = 32
    config['num_layers']        = 2
    config['enc_rnn_size']      = 512
    config['dec_rnn_size']      = 512
    config['src_embed_size']    = 512
    config['trg_embed_size']    = 512
    config['embed_norm']        = 3.5
    config['max_src_length']    = 50
    config['max_trg_length']    = 50
    config['init_range']        = 0.01
    config['max_epochs']        = 50
    config['lr']                = 1.0
    config['lr_decay']          = 0.5
    config['optimizer']         = ac.ADADELTA
    config['input_keep_prob']   = 0.8
    config['output_keep_prob']  = 0.8
    config['src_vocab_size']    = 27300
    config['trg_vocab_size']    = 15700
    config['grad_clip']         = 5.0
    config['reverse']           = True
    config['score_func_type']   = ac.SCORE_FUNC_GEN
    config['feed_input']        = True
    config['reload']            = True
    config['validate_freq']     = 1.0
    config['save_freq']         = 200
    config['beam_size']         = 12
    config['beam_alpha']        = 0.8
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config

def en2vi():
    config = {}

    config['model_name']        = 'en2vi'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'en'
    config['trg_lang']          = 'vi'
    config['data_dir']          = './nmt/data/en2vi'
    config['log_file']          = './nmt/DEBUG.log'
    config['rnn_type']          = ac.LSTM
    config['batch_size']        = 32
    config['num_layers']        = 2
    config['enc_rnn_size']      = 512
    config['dec_rnn_size']      = 512
    config['src_embed_size']    = 512
    config['trg_embed_size']    = 512
    config['embed_norm']        = 3.5
    config['max_src_length']    = 50
    config['max_trg_length']    = 50
    config['init_range']        = 0.01
    config['max_epochs']        = 50
    config['lr']                = 1.0
    config['lr_decay']          = 0.5
    config['optimizer']         = ac.ADADELTA
    config['input_keep_prob']   = 0.8
    config['output_keep_prob']  = 0.8
    config['src_vocab_size']    = 17000
    config['trg_vocab_size']    = 7700
    config['grad_clip']         = 5.0
    config['reverse']           = True
    config['score_func_type']   = ac.SCORE_FUNC_GEN
    config['feed_input']        = True
    config['reload']            = True
    config['validate_freq']     = 1.0
    config['save_freq']         = 200
    config['beam_size']         = 12
    config['beam_alpha']        = 0.8
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config

def en2ja():
    config = {}

    config['model_name']        = 'en2ja'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'en'
    config['trg_lang']          = 'ja'
    config['data_dir']          = './nmt/data/en2ja'
    config['log_file']          = './nmt/DEBUG.log'
    config['rnn_type']          = ac.LSTM
    config['batch_size']        = 32
    config['num_layers']        = 4
    config['enc_rnn_size']      = 768
    config['dec_rnn_size']      = 768
    config['src_embed_size']    = 768
    config['trg_embed_size']    = 768
    config['embed_norm']        = 3.5
    config['max_src_length']    = 50
    config['max_trg_length']    = 50
    config['init_range']        = 0.01
    config['max_epochs']        = 50
    config['lr']                = 1.0
    config['lr_decay']          = 0.5
    config['optimizer']         = ac.ADADELTA
    config['input_keep_prob']   = 0.8
    config['output_keep_prob']  = 0.8
    config['src_vocab_size']    = 48200
    config['trg_vocab_size']    = 49100
    config['grad_clip']         = 5.0
    config['reverse']           = True
    config['score_func_type']   = ac.SCORE_FUNC_GEN
    config['feed_input']        = True
    config['reload']            = True
    config['validate_freq']     = 0.5
    config['save_freq']         = 200
    config['beam_size']         = 12
    config['beam_alpha']        = 0.8
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config

def en2ja_btec():
    config = {}

    config['model_name']        = 'en2ja_btec'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'en'
    config['trg_lang']          = 'ja'
    config['data_dir']          = './nmt/data/en2ja_btec'
    config['log_file']          = './nmt/DEBUG.log'
    config['rnn_type']          = ac.LSTM
    config['batch_size']        = 32
    config['num_layers']        = 4
    config['enc_rnn_size']      = 768
    config['dec_rnn_size']      = 768
    config['src_embed_size']    = 768
    config['trg_embed_size']    = 768
    config['embed_norm']        = 3.5
    config['max_src_length']    = 50
    config['max_trg_length']    = 50
    config['init_range']        = 0.01
    config['max_epochs']        = 50
    config['lr']                = 1.0
    config['lr_decay']          = 0.5
    config['optimizer']         = ac.ADADELTA
    config['input_keep_prob']   = 0.8
    config['output_keep_prob']  = 0.8
    config['src_vocab_size']    = 17800
    config['trg_vocab_size']    = 21800
    config['grad_clip']         = 5.0
    config['reverse']           = True
    config['score_func_type']   = ac.SCORE_FUNC_GEN
    config['feed_input']        = True
    config['reload']            = True
    config['validate_freq']     = 0.5
    config['save_freq']         = 200
    config['beam_size']         = 12
    config['beam_alpha']        = 0.8
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config

def en2de():
    config = {}

    config['model_name']        = 'en2de'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'en'
    config['trg_lang']          = 'de'
    config['data_dir']          = './nmt/data/en2de'
    config['log_file']          = './nmt/DEBUG.log'
    config['rnn_type']          = ac.LSTM
    config['batch_size']        = 64
    config['num_layers']        = 4
    config['enc_rnn_size']      = 1024
    config['dec_rnn_size']      = 1024
    config['src_embed_size']    = 1024
    config['trg_embed_size']    = 1024
    config['embed_norm']        = 3.5
    config['max_src_length']    = 50
    config['max_trg_length']    = 50
    config['init_range']        = 0.01
    config['max_epochs']        = 12
    config['lr']                = 1.0
    config['lr_decay']          = 0.5
    config['optimizer']         = ac.ADADELTA
    config['input_keep_prob']   = 0.8
    config['output_keep_prob']  = 0.8
    config['src_vocab_size']    = 50000
    config['trg_vocab_size']    = 50000
    config['grad_clip']         = 5.0
    config['reverse']           = True
    config['score_func_type']   = ac.SCORE_FUNC_GEN
    config['feed_input']        = True
    config['reload']            = True
    config['validate_freq']     = 0.25
    config['save_freq']         = 2000
    config['beam_size']         = 12
    config['beam_alpha']        = 0.8
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config
