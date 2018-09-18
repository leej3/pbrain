# -*- coding: utf-8 -*-
TESTS_DATA_PATH=Path('tests/data')
SCANS_CSV= 
from pbrain.util import run_cmd

def test_vae_usage():
    cmd = ("pbrain train %s/  model_checkpoints 10 output_of_testing"
     % (TESTS_DATA_PATH)
    )
    assert False
    pp = run_cmd(cmd)


def test_vae_training():
	cmd = "python pbrain/cli.py train --model-dir model_checkpoints --csv fixed.csv"
	# train(
	# 	model_dir = '',csv,batch_size,n_epochs,multi_gpu)