# -*- coding: utf-8 -*-
"""Top-level module imports for pbrain."""

import warnings
# Ignore FutureWarning (from h5py in this case).
warnings.simplefilter(action='ignore', category=FutureWarning)

try:
    import tensorflow
except ImportError:
    raise ImportError(
        "TensorFlow cannot be found. Please re-install pbrain with either"
        " the [cpu] or [gpu] extras, or install TensorFlow separately. Please"
        " see https://www.tensorflow.org/install/ for installation"
        " instructions.")

# from pbrain import train
# from pbrain import models 
# from pbrain import volume
# from pbrain.io import read_csv
# from pbrain.io import read_json
# from pbrain.io import read_mapping
# from pbrain.io import read_volume
# from pbrain.io import save_json

# from pbrain.metrics import dice
# from pbrain.metrics import dice_numpy
# from pbrain.metrics import hamming
# from pbrain.metrics import hamming_numpy
# from pbrain.metrics import streaming_dice
# from pbrain.metrics import streaming_hamming

# from pbrain.models import get_estimator
# from pbrain.models import HighRes3DNet
# from pbrain.models import MeshNet
# from pbrain.models import QuickNAT

# from pbrain.predict import predict

# from pbrain.train import train

# from pbrain.volume import binarize
# from pbrain.volume import change_brightness
# from pbrain.volume import downsample
# from pbrain.volume import flip
# from pbrain.volume import from_blocks
# from pbrain.volume import iterblocks_3d
# from pbrain.volume import itervolumes
# from pbrain.volume import match_histogram
# from pbrain.volume import normalize_zero_one
# from pbrain.volume import reduce_contrast
# from pbrain.volume import replace
# from pbrain.volume import rotate
# from pbrain.volume import salt_and_pepper
# from pbrain.volume import shift
# from pbrain.volume import to_blocks
# from pbrain.volume import zoom
# from pbrain.volume import zscore
# from pbrain.volume import VolumeDataGenerator
