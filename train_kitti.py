# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer_kitti import Trainer
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()

# Cartella degli splits /home/radice/neuralNetworks/monodepth2/splits/eigen
# Hard coded options
opts.data_path = '/media/RAIDONE/radice/datasets/kitti/data'
opts.log_dir = '/media/RAIDONE/radice/neural-networks-data/monodepth2/models'
opts.png = True


if __name__ == "__main__":

    trainer = Trainer(opts)
    trainer.train()

    print("-> Done")

    exit()

