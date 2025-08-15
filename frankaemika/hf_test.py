# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the CDF project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------


# 7D panda robot
import numpy as np
import os
import sys
import torch
import math
import time
CUR_PATH = os.path.dirname(os.path.realpath(__file__))
from mlp import MLPRegression
sys.path.append(os.path.join(CUR_PATH,'../../RDF'))
from panda_layer.panda_layer import PandaLayer
import bf_sdf
from nn_cdf import CDF

PI = math.pi
# torch.manual_seed(10)
np.random.seed(10)
# torch.autograd.set_detect_anomaly(True)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cdf = CDF(device)
    print(f'data shape: {cdf.data["x"].shape}, {cdf.data["q"].shape}, {cdf.data["k"].shape}')
