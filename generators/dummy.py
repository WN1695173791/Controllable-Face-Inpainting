# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch.nn as nn
import torch



class Discriminator(nn.Module):
    """Dummy Discriminator constructor.

    Args:
        dis_cfg (obj): Discriminator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file
    """

    def __init__(self, dummy):
        super(Discriminator, self).__init__()
        self.dummy_layer = torch.nn.Linear(1, 1)
        pass

    def forward(self, data):
        """Dummy discriminator forward.

        Args:
            data (dict):
        """
        return



class Generator(nn.Module):
    r"""Dummy generator.

    Args:
        gen_cfg (obj): Generator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self, dummy):
        super(Generator, self).__init__()
        self.dummy_layer = torch.nn.Linear(1, 1)
        pass

    def forward(self, data):
        r"""Dummy Generator forward.

        Args:
            data (dict):
        """
        return
