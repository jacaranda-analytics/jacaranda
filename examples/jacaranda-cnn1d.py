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

#Genereate 50 realisations with 10 covariates
X = np.random.rand(50, 10).astype(dtype = np.float32)
Y = np.random.randint(5, size=(50, 1)).astype(dtype = np.float32)

#Expand dimensions so that we can use a cnn
X = np.expand_dims(X, axis=1)

config = jac_torch.pytorch_config(X, Y, n_trials=2)

cnn = jac_torch.pytorch_general(
    X, Y, config, define_model=jac_torch.pytorch_cnn1d
)
cnn.tune()

model = cnn.model