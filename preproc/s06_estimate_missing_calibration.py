import os
import sys
sys.path.append('./')
from collections import defaultdict
import pickle

import torch
import pycolmap
import numpy as np

from preproc import config as _C
from utils import rotation as r



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sequence', default='')
    parser.add_argument('-f', '--frames', nargs='+', type=int)
    args = parser.parse_args()

    if args.sequence != '':
        _C.SEQUENCE_NAME = args.sequence

    calib_pth = os.path.join(_C.PROC_CALIB_DIR, _C.SEQUENCE_NAME, "calib.npz")
    calib = dict(np.load(calib_pth))
    import pdb; pdb.set_trace()
    dense_kp_dir = os.path.join(_C.LANDMARK_RESULTS_DIR, _C.SEQUENCE_NAME)