import numpy as np
import torch
import jacaranda.pytorch as jac_torch

# Genereate 50 realisations with 10 covariates
X = np.random.rand(50, 10).astype(dtype=np.float32)
Y = np.random.randint(5, size=(50, 1)).astype(dtype=np.float32)

# Expand dimensions so that we can use a cnn
X = np.expand_dims(X, axis=1)

def loss_function(x, model, target):
    output = model(x)
    loss = torch.nn.MSELoss()
    return loss(output, target)

def metric_function(x, model, target):
    return loss_function(x, model, target).item()

config = jac_torch.pytorch_config(X, Y, loss = loss_function, metric= metric_function, n_trials=2)

cnn = jac_torch.pytorch_general(X, Y, config, define_model=jac_torch.pytorch_cnn1d)
cnn.tune()

model = cnn.model
