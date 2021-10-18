import pandas as pd
import numpy as np
import os 
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import optuna
from optuna.trial import TrialState
import sklearn.metrics
class mlp():
    def __init__(self, X, Y, config):
        self.X = X
        self.Y = Y
        self.config = config
        self.model = "Model has not been tuned yet"
        self.model_paramaters = "Model has not been tuned yet"
        self.model_metric = "Model has not been tuned yet"

    def define_model(self, trial):
        # We optimize the number of layers, hidden units and dropout ratio in each layer.
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layers = []

        in_features = self.config.IN_FEATURES_START
        for i in range(n_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), 50, 100)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            # p = trial.suggest_float("dropout_l{}".format(i), 0, 0.01)
            # layers.append(nn.Dropout(p))

            in_features = out_features
        layers.append(nn.Linear(in_features, self.config.CLASSES))
        # layers.append(nn.LogSoftmax(dim=1)) config.

        return nn.Sequential(*layers)

    def data_loader(self):
    
        dataset = [[self.X[i], self.Y[i]] for i in range(len(self.X))]

        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(self.config.VALIDATION_SPLIT * dataset_size))
        np.random.seed(self.config.SEED)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.BATCHSIZE, sampler=train_sampler
        )
        validation_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.BATCHSIZE, sampler=valid_sampler
        )

        return train_loader, validation_loader

    def objective(self, trial):

        loss_function = self.config.LOSS

        # Generate the model.
        model = self.define_model(trial).to(self.config.DEVICE)

        # Generate the optimizers.
        optimizer_name = trial.suggest_categorical(
            "optimizer", ["Adam", "RMSprop", "SGD"]
        )
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

        # Get the FashionMNIST dataset.
        train_loader, valid_loader = self.data_loader()

        # Training of the model.
        for epoch in range(self.config.EPOCHS):
            model.train()
            for train_data in train_loader:
                data, target = train_data
                # target = target.unsqueeze(1)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_function(output, target)
                loss.backward()
                optimizer.step()

            # Validation of the model.
            model.eval()
            X_val, y_true = train_data
            y_pred =  model(X_val).detach().numpy()
            accuracy = self.config.METRIC(y_true, y_pred)

            trial.report(accuracy, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return accuracy

    def build_model(self,study):
        model = self.define_model(study.best_trial).to(self.config.DEVICE)

        #Data
        train_loader, _ = self.data_loader()

        #Loss
        loss_function=self.config.LOSS

        #Optimiser
        optimizer_name = study.best_trial.params['optimizer']
        lr = study.best_trial.params['lr']
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)


        # Training of the model.
        for _ in range(self.config.EPOCHS):
            model.train()

            for train_data in train_loader:
                data, target = train_data
                # target = target.unsqueeze(1)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_function(output, target)
                loss.backward()
                optimizer.step()
        
        return model


    def tune(self):
        study = optuna.create_study(direction=self.config.OPTIMIZE_DIRECTION)
        study.optimize(self.objective, n_trials=self.config.N_TRIALS, timeout=self.config.TIMEOUT)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial
        self.model_metric = trial.value

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        self.model_paramaters = trial.params

        model = self.build_model(study)
        self.model = model
        
        return model
