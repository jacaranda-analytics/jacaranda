import pandas as pd
import numpy as np
import os
import optuna
from optuna.trial import TrialState
import random
import xgboost as xgb

import jacaranda.xgboost as jac_xgb

# Genereate 50 realisations with 10 covariates
X = np.random.rand(50, 10).astype(dtype=np.float32)
Y = np.random.randint(5, size=(50, 1)).astype(dtype=np.float32)

config = jac_xgb.xgboost_config(X, Y, n_trials=2)

xgboost = jac_xgb.xgboost_general(
    X,
    Y, 
    config,
)
xgboost.tune()

model = xgboost.model
