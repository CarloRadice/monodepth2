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
MIN_DEPTH = 1e-3
MAX_DEPTH = 80
EVAL_SPLIT = 'eigen'
DATA_PATH = '/media/RAIDONE/radice/datasets/kitti/data'
GT_DIR = '/media/RAIDONE/radice/neural-networks-data/evaluation'
PREDICTIONS_DIR = '/media/RAIDONE/radice/neural-networks-data/predictions'
TEST_FILE = '/home/radice/neuralNetworks/monodepth2/splits/eigen/test_files.txt'
EXT_TEST_FILE = '/media/RAIDONE/radice/datasets/kitti/eigen_test_files.txt'
KITTI_MODELS_DIR = '/media/RAIDONE/radice/neural-networks-data/monodepth2/models'
# Options list
# "--load_weights_folder", help="name of model to load"
# "--ext_disp_to_eval", help="optional path to a .npy disparities file to evaluate, ovvero cartella in PREDICTION_DIR"
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


def model_evaluation(opt):
    model_folder = os.path.join(KITTI_MODELS_DIR, opt.load_weights_folder)

    assert os.path.isdir(model_folder), \
        "Cannot find a folder at {}".format(model_folder)

    print("-> Loading weights from {}".format(model_folder))

    filenames = readlines(TEST_FILE)
    encoder_path = os.path.join(model_folder, "encoder.pth")
    decoder_path = os.path.join(model_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    img_ext = '.png'

    dataset = datasets.KITTIRAWDataset(DATA_PATH, filenames,
                                       encoder_dict['height'], encoder_dict['width'],
                                       [0], 4, is_train=False, img_ext=img_ext)

    dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)

    print('-> Loaded test file')

    encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    pred_disps = []

    print("-> Computing predictions with size {}x{}".format(
        encoder_dict['width'], encoder_dict['height']))

    with torch.no_grad():
        for data in dataloader:
            input_color = data[("color", 0, 0)].cuda()

            if opt.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            output = depth_decoder(encoder(input_color))

            pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            if opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

            pred_disps.append(pred_disp)

    pred_disps = np.concatenate(pred_disps)

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    gt_path = os.path.join(GT_DIR, "kitti_eigen_gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

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

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:
        # Valutazione del modello allenato
        model_evaluation(opt=opt)
    else:
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

        test_file = open(EXT_TEST_FILE, 'r')
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
        
            print('({}/{})'.format(idx + 1, len(lines)))

        mean_errors = np.array(errors).mean(0)

        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
