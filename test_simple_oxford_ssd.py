# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth

# center crop 50% vertically
CROP_AREA = [0, 240, 1280, 720]
MODELS_DIR = '/media/RAIDONE/radice/neural-networks-data/monodepth2/models'

# SOLO PER PROVA
TEST_FILE = '/home/radice/datasets/oxford-radar/test_files.txt'
OUTPUT_DIR = '/home/radice/datasets/oxford-radar/predictions'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use')
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="png")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--resnet",
                        type=int,
                        default=18)
    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """

    # F = []
    # for f in sorted(glob.glob('/media/RAIDONE/radice/datasets/oxford-radar/2019-01-10-14-36-48-radar-oxford-10k-partial/stereo/left/*.png')):
    #     F.append(f + '\n')
    # file = open(TEST_FILE, 'w')
    # file.writelines(F)
    # file.close()
    # exit()

    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_path = os.path.join(MODELS_DIR, args.model_name)
    print("-> Loading model from", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    if args.resnet == 18:
        encoder = networks.ResnetEncoder(18, False)
    elif args.resnet == 50:
        encoder = networks.ResnetEncoder(50, False)
    else:
        raise Exception('Can not find resnet {}'.format(args.resnet))

    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # Changes
    test_file = open(TEST_FILE, 'r')
    lines = test_file.readlines()
    test_file.close()

    model = args.model_name

    output_folder = os.path.join(OUTPUT_DIR, '{}{}'.format('md2-', model))
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    print('-> Save in folder {}{}'.format('md2-', model))
    print("-> Predicting on {:d} test images".format(len(lines)))

    # PREDICTING ON EACH IMAGE IN TURN
    idx = 0
    with torch.no_grad():
        for example in lines:

            # Load image and preprocess
            input_image = pil.open(example.rstrip()).convert('RGB')

            # Crop image
            input_image = input_image.crop(CROP_AREA)

            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(example))[0]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)

            name_dest_npy = os.path.join(output_folder, "{}_disp.npy".format(output_name))
            np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_folder, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved predictions to:".format(
                idx + 1, len(lines)))
            print("   - {}".format(name_dest_im))
            print("   - {}".format(name_dest_npy))
            idx += 1

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
