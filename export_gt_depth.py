# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os

import argparse
import numpy as np
import PIL.Image as pil

from utils import readlines
from kitti_utils import generate_depth_map

DATA = '/media/RAIDONE/radice/datasets/kitti/data/'
TEST_GT_FOLDER = '/media/RAIDONE/radice/datasets/kitti/data/test-velodyne-gt'
TEST_FILE = '/media/RAIDONE/radice/datasets/kitti/eigen_test_files.txt'
OUTPUT_DIR = '/media/RAIDONE/radice/neural-networks-data/evaluation'


def export_gt_depths_kitti():
    '''
        Creo le gt di image_02
        Totale 697 ground truths
    '''
    test_file = open(TEST_FILE, 'r')
    lines = test_file.readlines()

    gt_depths = []
    for line in lines:

        line = line.split()[0]

        line = line.split('/')

        date = line[0]
        seqname = line[1]
        basename = line[4].split('.')[0]

        calib_dir = os.path.join(DATA, date)
        velo_filename = os.path.join(TEST_GT_FOLDER, date, seqname, "velodyne_points/data", "{}.bin".format(basename))
        gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
        gt_depths.append(gt_depth.astype(np.float32))

    output_path = os.path.join(OUTPUT_DIR, "kitti_eigen_gt_depths.npz")

    print("Saving to {}".format(output_path))

    np.savez_compressed(output_path, data=np.array(gt_depths))


if __name__ == "__main__":
    export_gt_depths_kitti()
