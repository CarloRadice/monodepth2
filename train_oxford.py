# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer_oxford import Trainer
from options import MonodepthOptions
import sys


options = MonodepthOptions()
opts = options.parse()
# Hard coded options
opts.data_path = '/media/RAIDONE/radice/datasets/oxford'
opts.log_dir = '/media/RAIDONE/radice/neural-networks-data/monodepth2/log'
opts.png = True
opts.split = "oxford"
opts.dataset = "oxford"
opts.height = 416
opts.width = 640
opts.batch_size = 6

if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()

    print("FATTO")
    sys.exit()
