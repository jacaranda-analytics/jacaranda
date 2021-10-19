import numpy as np
import jacaranda.pytorch as jac_torch

# Genereate 50 realisations with 10 covariates
X = np.random.rand(50, 10).astype(dtype=np.float32)
Y = np.random.randint(5, size=(50, 1)).astype(dtype=np.float32)

config = jac_torch.pytorch_config(X, Y, n_trials=2)

mlp = jac_torch.pytorch_general(X, Y, config, define_model=jac_torch.pytorch_mlp)
mlp.tune()

model = mlp.model
