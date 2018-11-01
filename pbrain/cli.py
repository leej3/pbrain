# -*- coding: utf-8 -*-
"""Main command-line interface to pbrain."""

import argparse
import sys

from pbrain.train import train as _train
from pbrain.predict import predict as _predict
from pbrain.pval import pval as _pval
from pbrain.csv_to_pvals import csv_to_pvals as _csv_to_pvals

from pbrain.util import clean_csv, str2bool
from pbrain.util import conform_csv as _conform_csv
from pbrain.util import setup_exceptionhook
import pbrain

STATS_PATH = (Path(pbrain.__file__).parent.parent / 'reference_files' / 'reference_stats')
MODELS_PATH = (Path(pbrain.__file__).parent.parent / 'reference_files' / 'reference_models')
CSV_PATH = (Path(pbrain.__file__).parent.parent / 'reference_files' / 'reference.csv')

def create_parser():
    """Return argument parser for pbrain training interface."""
    p = argparse.ArgumentParser()

    p.add_argument('--debug', action='store_true', dest='debug',
                    help='Do not catch exceptions and show exception '
                    'traceback')


    subparsers = p.add_subparsers(
        dest="subparser_name", title="subcommands",
        description="valid subcommands")

    cp = subparsers.add_parser('clean_csv', help="Create cleaned csv"
     "containing files that can be opened with nibabel")
    c = cp.add_argument_group('clean_csv arguments')
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
        '--model-dir', required=True,
        help="Directory in which to save model checkpoints. If an existing"
             " directory, will resume training from last checkpoint. If not"
             " specified, will use a temporary directory.")

    t = tp.add_argument_group('train arguments')
    t.add_argument(
        '--input-csv', required=True,
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
        '-e', '--n-epochs', type=int, default=5,
        help="Number of training epochs")
    t.add_argument(
        '--multi-gpu', action='store_true',
        help="Train across all available GPUs. Batches are split across GPUs. Not yet implemented")
    t.add_argument('--stats-path',type=str,
        help="Path to statistics files",default=STATS_PATH)

    # Prediction subparser
    pp = subparsers.add_parser('predict', help="Predict using SavedModel")
    pp.add_argument('--input-csv',required=True, help="Filepath to csv containing scan paths.")
    pp.add_argument('--output-csv', help="Name out output csv filename.")
    ppp = pp.add_argument_group('prediction arguments')
    ppp.add_argument(
        '-m', '--model-dir', default=MODELS_PATH, help="Path to directory containing the model.")
    ###
    ppp.add_argument('--output-dir',required= False, help="Name of output directory.",default=None)
    ppp.add_argument('--stats-path',type=str,
        help="Path to statistics files",default=STATS_PATH)
    
    # pval subparser
    pv = subparsers.add_parser('pval', help="Get pvals for each scan in a csv.")
    pv.add_argument('--input-csv',required=True, help="Filepath to csv containing scan paths.")
    pv.add_argument('--output-csv', help="Name out output csv filename.")
    pvp = pv.add_argument_group('pval arguments')
    ###
    pvp.add_argument('--reference-csv',required= False, help="Reference csv containing scores for "
        "the training set.",default=CSV_PATH)
    
    # csv_to_pvals subparser
    c2p = subparsers.add_parser('csv_to_pvals', help="Predict using SavedModel")
    c2p.add_argument('--input-csv', required=True,help="Filepath to csv containing scan paths.")
    c2p.add_argument('--output-csv',required= False, help="Name out output csv filename.",default=None)
    c2pp = c2p.add_argument_group('csv_to_pvals arguments')
    c2pp.add_argument(
        '-m', '--model-dir', default=MODELS_PATH, help="Path to directory containing the model.")
    ###
    c2pp.add_argument('--output-dir',required= False, help="Name of output directory.",default=None)
    c2pp.add_argument('--reference-csv',required= False, help="Reference csv containing scores for "
                    "the training set.",default=CSV_PATH)
    c2pp.add_argument('--clean-input-csv',default=True,type= lambda x: str2bool(x),
                     help="Flag to check that all images in the csv can be loaded into nibabel. Write out a cleaned csv")
    c2pp.add_argument('--target-shape',default=[256,256,256],type= int,nargs=3,
                     help="Length of X,Y,Z dims in number of voxels")
    c2pp.add_argument('--voxel-dims',default=[1.0,1.0,1.0],type= float,nargs=3,
                     help="Length of X,Y,Z dims of voxels in mm")
    c2pp.add_argument('--stats-path',type=str,
                     help="Path to statistics files",default=STATS_PATH)

  # conform_csv subparser
    c2c = subparsers.add_parser('conform_csv', help="Conform all scans in csv as required to be useful input to neural network")
    c2c.add_argument('--input-csv', required= True, help="Filepath to csv containing scan paths.")
    c2c.add_argument('--output-csv',required= True, help="Name out output csv filename.",default=None)
    c2cp = c2c.add_argument_group('conform_csv arguments')
    c2cp.add_argument('--output-shape',default=[256,256,256],type= int,nargs=3,
                     help="Length of X,Y,Z dims in number of voxels")
    c2cp.add_argument('--voxel-dims',default=[1.0,1.0,1.0],type=float,nargs=3,
                     help="Length of X,Y,Z dims of voxels in mm")

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
        input_csv=params['input_csv'],
        batch_size=params['batch_size'],
        n_epochs=params['n_epochs'],
        multi_gpu=params['multi_gpu'],
        stats_path=params['stats_path'],
        )

def predict(params):
    _predict(
        model_dir=params['model_dir'],
        input_csv=params['input_csv'],
        output_csv=params['output_csv'],
        output_dir=params['output_dir'],
        stats_path=params['stats_path'],
        )

def pval(params):
    _pval(
        input_csv=params['input_csv'],
        output_csv=params['output_csv'],
        reference_csv=params['reference_csv'],
        )

def conform_csv(params):
    _conform_csv(
        input_csv=params['input_csv'],
        output_csv=params['output_csv'],
        target_shape=tuple(params['target_shape']),
        voxel_dims=params['voxel_dims'],
        )

def csv_to_pvals(params):
    _csv_to_pvals(
        input_csv=params['input_csv'],
        model_dir=params['model_dir'],
        output_dir=params['output_dir'],
        output_csv=params['output_csv'],
        reference_csv=params['reference_csv'],
        clean_input_csv=bool(params['clean_input_csv']),
        target_shape=tuple(params['target_shape']),
        voxel_dims=params['voxel_dims'],
        stats_path=params['stats_path'],
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

    if params['subparser_name'] == 'csv_to_pvals':
        csv_to_pvals(params=params)

    if params['subparser_name'] == 'conform_csv':
        conform_csv(params=params)

    if params['subparser_name'] == 'clean_csv':
        clean_csv(params['input_csv'], params['output_csv'])


if __name__ == '__main__':
    main()