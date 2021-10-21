from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
from .mono_dataset import MonoDataset


class OXFORDDataset(MonoDataset):
    """
    Super class for different types of OXFORD dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(OXFORDDataset, self).__init__(*args, **kwargs)

    # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
    # To normalize you need to scale the first row by 1 / image_width and the second row
    # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
    # If your principal point is far from the center you might need to disable the horizontal
    # flip augmentation.

        # Cosa ho fatto:
        # prima riga: 983.044006 / 1280, 0, 643.646973/ 1280, 0
        # seconda riga: 0, 983.044006 / 960, 493.378998 / 960, 0
        # self.K = np.array([[0.77, 0, 0.5, 0],
        #                    [0, 1.02, 0.5, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)

        # NO IMAGE CROP
        # self.full_res_shape = (1280, 960)
        # IMAGE CROP
        self.full_res_shape = (1280, 800)
        # seconda riga: 0, 983.044006 / 800, 493.378998 / 800, 0
        self.K = np.array([[0.77, 0, 0.5, 0],
                           [0, 1.23, 0.62, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.side_map = {"l": "left", "r": "right"}

    def check_depth(self):
        """
        Training without ground truth density maps
        """
        return False


    def get_color(self, folder, frame_index, side, do_flip):
        """
        Horizontal flip augmentation.
        """
        color = self.loader(self.get_image_path(folder, frame_index, side))

        # if do_flip:
        #     color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class OXFORDRAWDataset(OXFORDDataset):
    """
    Oxford dataset
    """
    def __init__(self, *args, **kwargs):
        super(OXFORDRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        """
        TO DO
        """
        # frame_index è l'intero nome dell'immagine
        f_str = "{}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(folder, "{}".format(self.side_map[side]), f_str)
        return image_path