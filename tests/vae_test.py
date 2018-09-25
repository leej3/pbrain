# -*- coding: utf-8 -*-
TESTS_DATA_PATH=Path('tests/data')
SCANS_CSV= TESTS_DATA_PATH / 'testing_scans.csv'
from pbrain.util import run_cmd, conform_scan
from pathlib import Path
import nibabel as nib
from nilearn import datasets
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

def test_conform_scan():
    expected = (Path(pbrain.__file__).parent.parent.
        joinpath('tests','data','mni_resampled_10_1_10.nii').exists())
    expected = nib.loadsave.load(expected)
    img = datasets.load_mni152_template()
    resampled_img = conform_scan(img=img,d=100,voxel_dims=[10,1,10])
    assert np.allclose(expected.get_data(),resampled_img.get_data())

# CUDA_VISIBLE_DEVICES=7 pbrain predict --input-csv=/data/MLcore/pbrain/fixed.csv --output-csv=prediction_2.csv --model-dir=model_checkpoints