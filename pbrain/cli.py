# -*- coding: utf-8 -*-
"""Main command-line interface to pbrain."""

import argparse
import sys

from pbrain.train import train as _train
from pbrain.predict import predict as _predict
from pbrain.pval import pval as _pval

from pbrain.util import clean_csv
from pbrain.util import setup_exceptionhook



def create_parser():
    """Return argument parser for pbrain training interface."""
    p = argparse.ArgumentParser()

    p.add_argument('--debug', action='store_true', dest='debug',
                    help='Do not catch exceptions and show exception '
                    'traceback')


    subparsers = p.add_subparsers(
        dest="subparser_name", title="subcommands",
        description="valid subcommands")

    cp = subparsers.add_parser('csv', help="Create cleaned csv"
     "containing files that can be opened with nibabel")
    c = cp.add_argument_group('csv arguments')
    c.add_argument(
        '--input-csv', required=True,
        help="Path to CSV of features, labels for training.")
    
    c.add_argument(
        '--output-csv', required=True,
        help="Path to output CSV containing files that can be loaded by nibabel.")


    # Training subparser
    tp = subparsers.add_parser('train', help="Train models")

    m = tp.add_argument_group('model arguments')
    m.add_argument(
        '--model-dir',
        help="Directory in which to save model checkpoints. If an existing"
             " directory, will resume training from last checkpoint. If not"
             " specified, will use a temporary directory.")

    t = tp.add_argument_group('train arguments')
    t.add_argument(
        '--csv', required=True,
        help="Path to CSV of features, labels for training.")
    t.add_argument(
        '-o', '--optimizer', required=False,
        help="Optimizer to use for training")
    t.add_argument(
        '-l', '--learning-rate', required=False, type=float,default=0.001,
        help="Learning rate to use with optimizer for training")
    t.add_argument(
        '-b', '--batch-size', required=False, type=int,default=1,
        help="Number of samples per batch. If `--multi-gpu` is specified,"
             " batch is split across available GPUs.")
    t.add_argument(
        '-e', '--n-epochs', type=int, default=10,
        help="Number of training epochs")
    t.add_argument(
        '--multi-gpu', action='store_true',
        help="Train across all available GPUs. Batches are split across GPUs. Not yet implemented")


    # Prediction subparser
    pp = subparsers.add_parser('predict', help="Predict using SavedModel")
    pp.add_argument('--input-csv', help="Filepath to csv containing scan paths.")
    pp.add_argument('--output-csv', help="Name out output csv filename.")
    ppp = pp.add_argument_group('prediction arguments')
    ppp.add_argument(
        '-m', '--model-dir', required=True, help="Path to directory containing the model.")
    ###
    ppp.add_argument('--output-dir',required= False, help="Name of output directory.",default=None)

    # pval subparser
    pv = subparsers.add_parser('pval', help="Get pvals for each scan in a csv.")
    pv.add_argument('--input-csv', help="Filepath to csv containing scan paths.")
    pv.add_argument('--output-csv', help="Name out output csv filename.")
    pvp = pv.add_argument_group('pval arguments')
    pvp.add_argument(
        '-m', '--model-dir', required=True, help="Path to directory containing the model.")
    ###

    pvp.add_argument('--output-dir',required= False, help="Name of output directory.",default=None)

    return p


def parse_args(args):
    """Return namespace of arguments."""
    parser = create_parser()
    namespace = parser.parse_args(args)
    if namespace.subparser_name is None:
        parser.print_usage()
        parser.exit(1)
    return namespace


def train(params):
    _train(
        model_dir=params['model_dir'],
        csv=params['csv'],
        batch_size=params['batch_size'],
        n_epochs=params['n_epochs'],
        multi_gpu=params['multi_gpu'],
        )

def predict(params):
    _predict(
        model_dir=params['model_dir'],
        input_csv=params['input_csv'],
        output_csv=params['output_csv'],
        output_dir=params['output_dir'],
        )

def pval(params):
    _pval(
        model_dir=params['model_dir'],
        input_csv=params['input_csv'],
        output_csv=params['output_csv'],
        reference_csv=params['reference_csv'],
        output_dir=params['output_dir'],
        )


def main(args=None):
    if args is None:
        namespace = parse_args(sys.argv[1:])
    else:
        namespace = parse_args(args)
    params = vars(namespace)

    if params['debug']:
        setup_exceptionhook()


    if params['subparser_name'] == 'train':
        train(params=params)

    if params['subparser_name'] == 'predict':
        predict(params=params)

    if params['subparser_name'] == 'pval':
        pval(params=params)

    if params['subparser_name'] == 'csv':
        clean_csv(params['input_csv'], params['output_csv'])

if __name__ == '__main__':
    main()