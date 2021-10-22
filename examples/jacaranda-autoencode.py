import numpy as np
import torch
import jacaranda.pytorch as jac_torch

# Genereate 50 realisations with 10 covariates
X = np.random.rand(50, 10).astype(dtype=np.float32)

def loss_function(x, model, target):
    output, _ = model(x)
    loss = torch.nn.MSELoss()
    return loss(output, target)

def metric_function(x, model, target):
    return loss_function(x, model, target).item()

config = jac_torch.pytorch_config(X, X, loss = loss_function, metric= metric_function, n_trials=2)

mlp = jac_torch.pytorch_general(X, X, config, define_model=jac_torch.pytorch_autoencoder)
mlp.tune()

model = mlp.model