import os
import sys
sys.path.append('./')

import argparse
import matplotlib.pyplot as plt

from preproc import config as _C
from utils.contact_utils import parse_arduino_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sequence', default='')
    args = parser.parse_args()

    if args.sequence != '':
        _C.SEQUENCE_NAME = args.sequence

    contact_txt_path = os.path.join(_C.RAW_DATA_DIR, _C.SEQUENCE_NAME, "contact", "DATALOG.TXT")

    timestamps, readings = parse_arduino_data(contact_txt_path)
    
    # hand_idxs = [0, 1, 10, 11]
    hand_idxs = [14, 15]
    plt.plot(timestamps / 1e3, readings[:, hand_idxs])
    # import pdb; pdb.set_trace()
    # plt.plot(timestamps[100:] / 1e3, readings[100:, hand_idxs])
    # plt.plot(readings[:, hand_idxs])
    plt.show()

    # check_idxs = [12, 13]
    # check_idxs = [6, 7, 8, 9]
    # check_idxs = [8]
    # check_idxs = [2, 5]
    # for check_idxs in [6, 7, 8, 9]:
    # for check_idxs in range(16):
    #     plt.plot(readings[:, check_idxs])
    #     plt.show()