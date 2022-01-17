from __future__ import absolute_import, division, print_function

import glob
import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4
EVAL_SPLIT = 'eigen'
GT_DIR = '/media/RAIDONE/radice/neural-networks-data/evaluation'
PREDICTIONS_DIR = '/media/RAIDONE/radice/neural-networks-data/predictions'
TEST_FILE = '/media/RAIDONE/radice/datasets/kitti/eigen_test_files.txt'
# Options list
#"--ext_disp_to_eval", help="optional path to a .npy disparities file to evaluate, ovvero cartella in PREDICTION_DIR"
# "--eval_stereo" help="if set evaluates in stereo mode"
# "--eval_mono", help="if set evaluates in mono mode"
# "--disable_median_scaling", help="if set disables median scaling in evaluation"
# "--pred_depth_scale_factor", help="if set multiplies predictions by this number"

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    path = os.path.join(PREDICTIONS_DIR, opt.ext_disp_to_eval)
    # Load predictions from file
    print("-> Loading predictions from {}".format(path))

    gt_path = os.path.join(GT_DIR, "kitti_eigen_gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    test_file = open(TEST_FILE, 'r')
    lines = test_file.readlines()
    print('-> {} examples'.format(len(lines)))
    for idx, line in enumerate(lines):
        line = line.split()[0]

        line = line.split('/')

        date = line[0]
        seqname = line[1]
        basename = line[4].split('.')[0]

        # può essere '{}_disp.npy'.format(basename) o '{}_depth.npy'.format(basename)
        if os.path.isfile(os.path.join(PREDICTIONS_DIR, opt.ext_disp_to_eval, date, seqname, 'image_02', '{}_disp.npy'.format(basename))):
            # (1, 1, WIDTH, HEIGHT)
            pred_path = os.path.join(PREDICTIONS_DIR, opt.ext_disp_to_eval, date, seqname, 'image_02',
                                     '{}_disp.npy'.format(basename))
            pred_disp = np.load(pred_path)
        else:
            if os.path.isfile(os.path.join(PREDICTIONS_DIR, opt.ext_disp_to_eval, date, seqname, 'image_02',
                                     '{}_depth.npy'.format(basename))):
                pred_path = os.path.join(PREDICTIONS_DIR, opt.ext_disp_to_eval, date, seqname, 'image_02',
                         '{}_depth.npy'.format(basename))
                pred_disp = np.load(pred_path)
            else:
                raise Exception('{} is not a valid path.'.format(os.path.join(PREDICTIONS_DIR, opt.ext_disp_to_eval, date, seqname, 'image_02',
                         '{}'.format(basename))))

        gt_depth = gt_depths[idx]
        gt_height, gt_width = gt_depth.shape[:2]

        # (1, WIDTH, HEIGHT)
        pred_disp = pred_disp[0]
        # (WIDTH, HEIGHT)
        pred_disp = pred_disp[0]

        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        # eval split is eigen
        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor

        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
