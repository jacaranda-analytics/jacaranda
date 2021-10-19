import pandas as pd
import numpy as np
import os
import torch.nn as nn
import torch
import torch.optim as optim
from torch.nn.modules.loss import MSELoss
from torch.utils.data.sampler import SubsetRandomSampler
import optuna
from optuna.trial import TrialState
import random

import jacaranda.pytorch as jac_torch

# Genereate 50 realisations with 10 covariates
X = np.random.rand(50, 10).astype(dtype=np.float32)
Y = np.random.randint(5, size=(50, 1)).astype(dtype=np.float32)

config = jac_torch.pytorch_config(X, X, n_trials=2, encode_level=0.5)

autoencode = jac_torch.pytorch_autoencoder(
    X,
    config,
)
autoencode.tune()

model = autoencode.model
