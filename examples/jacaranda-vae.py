import numpy as np
import torch
import jacaranda.pytorch as jac_torch
import torch.nn.functional as F

# Genereate 50 realisations with 10 covariates
X = np.random.rand(50, 10).astype(dtype=np.float32)

def loss_function(x, model, target):
    x_reconst, mu, log_var = model(x)
    reconst_loss = F.binary_cross_entropy(x_reconst, x)
    kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconst_loss + kl_div

def metric_function(x, model, target):
    return loss_function(x, model, target).item()

config = jac_torch.pytorch_config(X, X, loss = loss_function, metric= metric_function, n_trials=2)

vae = jac_torch.pytorch_general(X, X, config, define_model=jac_torch.pytorch_vae)
vae.tune()

model = vae.model