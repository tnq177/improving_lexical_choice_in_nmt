from __future__ import print_function
from __future__ import division

import sys
import argparse
from nmt.train import Trainer
from nmt.translate import Translator
from nmt.extractor import Extractor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'translate', 'extract'], default='train')
    parser.add_argument('--proto', type=str, required=True,
                        help='Training config function defined in configurations.py')
    parser.add_argument('--num-preload', type=int, default=20,
                        help="""Number of train batches prefetched to memory
                                Small is slower but too big might make training data
                                less randomized.""")
    parser.add_argument('--input-file', type=str, 
                        help='Input file if mode == translate')
    parser.add_argument('--lexical-file', type=str, required=True,
                        help='Path to lexical file')
    parser.add_argument('--model-file', type=str,
                        help='Path to saved checkpoint if mode == translate')
    parser.add_argument('--var-list', nargs='+',
                        help='List of model vars to be fixed during training (--var-list )')
    parser.add_argument('--save-to', required='--var-list' in sys.argv, help='Directory to save extracted vars to')
    parser.add_argument('--plot-align', action='store_true', help='Plot alignments in translation')
    parser.add_argument('--unk-repl', action='store_true', help='Replace UNK with source word with highest attention scores')
    args = parser.parse_args()

    if args.mode == 'train':
        trainer = Trainer(args)
        trainer.train()
    elif args.mode == 'translate':
        translator = Translator(args)
    elif args.mode == 'extract':
        extractor = Extractor(args)
    else:
        print('Yo wassup!')
