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
import os, typing, math

class pytorch_config:
    '''  General configuration class for PyTorch hyper tuning to pass to each of 
    the hyperparamter tuning class.
    '''
    def __init__(
        self,
        X:np.array,
        Y:np.array,
        loss=torch.nn.MSELoss(),
        device: torch.device = torch.device("cpu"),
        metric:typing.Callable = sklearn.metrics.mean_squared_error,
        batchsize: int = 100,
        validation_split: float = 0.2,
        epochs: int = 20,
        seed: float = 123,
        dir:str = os.getcwd(),
        n_trials: int = 75,
        timeout: float = 600,
        optimize_direction: str = "minimize",
        encode_level:float = 0.5,
    ):
        """
        Args:
            X (numpy.array): Independent Covariates

            Y (numpy.array): Dependent Covariates (may be expressed as One hot encode). Note,
            needs to have two-dimensional shape.

            loss ([type], optional): Loss function for pytorch pipelines,
            does not need to be considered for other pipelines). Defaults to torch.nn.MSELoss().

            device (torch.device, optional): Device to run solve for Pytorch pipelines. 
            This does not need to be considered for other pipelines. Defaults to torch.device("cpu").

            batchsize (int, optional): Batch size for Pytorch pipelines Defaults to 100.

            validation_split (float, optional): Validation Split, must be between 0 and 
            1 Defaults to 0.2.

            epochs (int, optional): Epochs for Pytorch pipelines. 
            This does not need to be considered for other pipelines. Defaults to 20.

            seed (float, optional): Random seed used in data loader. Defaults to 123.

            dir (str, optional): Directory to run pipeline. Defaults to os.getcwd().

            n_trials (int, optional): Number of trials to run in the random grid search 
            by Optuna Defaults to 75.

            timeout (float, optional): Timeout for a single trial in Optuna random grid 
            search. Defaults to 600.

            metric (function, optional): Function of metric to determine best model. 
            Defaults to sklearn.metrics.mean_squared_error.

            optimize_direction (str, optional): Direction to optimize (valid inputs are
            'maximize' and 'minimize'). Defaults to 'minimize'.

            encode_level (int, optional): Number of variable to autoencode to determined 
            by a proportion (must be  between 0 and 1) Defaults to 0.5

        """        ''''''
        self.BATCHSIZE = batchsize
        self.VALIDATION_SPLIT = validation_split
        self.EPOCHS = epochs
        self.SEED = seed
        self.DIR = dir
        self.CLASSES = Y.shape[1]
        self.LOSS = loss
        self.DEVICE = device
        self.IN_FEATURES_START = X.shape[-1]
        self.N_TRIALS = n_trials
        self.TIMEOUT = timeout
        self.METRIC = metric
        self.OPTIMIZE_DIRECTION = optimize_direction
        self.ENCODE_LEVEL = math.floor(X.shape[1]/encode_level)


##############################

class pytorch_general():
    """
    Flexible class to tune Pytorch models
    """


    def __init__(self, 
                 X:np.array, 
                 Y:np.array, 
                 config: typing.Type[pytorch_config], 
                 define_model: typing.Type[nn.Module]):
        """
        Args:
            X (np.array): Independent Covariates
            Y (np.array): Dependent Covariates (may be expressed as One hot encode). 
            config (typing.Type[pytorch_config]): config class specific to X,Y
            define_model (typing.Type[nn.Module]):nn.Module to tune (can contain optuna trial aspects)
        """

        self.X = X
        self.Y = Y
        self.config = config
        self.define_model = define_model
        self.model = "Model has not been tuned yet"
        self.model_paramaters = "Model has not been tuned yet"
        self.model_metric = "Model has not been tuned yet"

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
        """Objective function for Optuna tune. 

        Args:
            trial: Optuna trial
        Raises:
            optuna.exceptions.TrialPruned: Optuna inbuilt pruning 

        Returns:
            float: accuracy of current model, defined by config.METRIC
        """
        loss_function = self.config.LOSS

        # Generate the model.
        model = self.define_model(self.config, trial).to(self.config.DEVICE)

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
            y_pred =  model(X_val).detach().np()
            accuracy = self.config.METRIC(y_true, y_pred)

            trial.report(accuracy, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return accuracy

    def build_model(self,study):
        """Rebuilds best model from Optuna search

        Args:
            study ([type]): Best study from Optuna search

        Returns:
            [type]: best model
        """        
        model = self.define_model(self.config, study.best_trial).to(self.config.DEVICE)

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
        """Engages the Optuna search

        Returns:
            [type]: best model
        """        
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



class pytorch_mlp(nn.Module):
    """General class for an mlp, this is passed to pytorch_general to tune an mlp.
    """  
    def __init__(self, config: typing.Type[pytorch_config], trial):
        """[summary]

        Args:
            config (typing.Type[pytorch_config]): config class specific this is handled by pytorch_general
            trial: Optuna trial
        """       
        super(pytorch_mlp,self).__init__()

        self.trial = trial

        n_layers = trial.suggest_int("n_layers", 1, 3)
        layers = []

        in_features = config.IN_FEATURES_START
        for i in range(n_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), 50, 100)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            # p = trial.suggest_float("dropout_l{}".format(i), 0, 0.01)
            # layers.append(nn.Dropout(p))

            in_features = out_features
        layers.append(nn.Linear(in_features, config.CLASSES))
        # layers.append(nn.LogSoftmax(dim=1)) config.

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # forward pass
        x = self.layers(x)
        return x


class pytorch_cnn1d(nn.Module):
    """General class for a cnn, this is passed to pytorch_general to tune an mlp.
    """ 
    def __init__(self,  config: typing.Type[pytorch_config], trial):
        """[summary]

        Args:
            config (typing.Type[pytorch_config]): config class specific this is handled by pytorch_general
            trial: Optuna trial
        """  
        super(pytorch_cnn1d,self).__init__()

        self.trial = trial

        in_features = config.IN_FEATURES_START
        layers1 = []
        layers2 = []
        
        def output(w, k, p=0, s =1):
            return (w - k + 2*p)/s + 1

        # this computes no of features outputted by 2 conv layers
    
        # self.n_conv = int((( ( (self.in_features - 2)/4 ) - 2 )/4 ) * 16)

        num_filters1 = trial.suggest_int("num_filters1",16,64,step=16)
        num_filters2 = trial.suggest_int("num_filters2",16,64,step=16)
        # num_filters2 = 16 
        kernel_size = trial.suggest_int('kernel_size', 2, 7)

        c1 = output(w = in_features, k = kernel_size)  # this is to account for the loss due to conversion to int type
        c2 = output(w = c1, k = kernel_size)
        n_conv = int(c2 * num_filters2)
        

        layers1.append(nn.Conv1d(1, num_filters1, kernel_size, 1))
        layers1.append(nn.BatchNorm1d(num_filters1))
        layers1.append(nn.Conv1d(num_filters1, num_filters2, kernel_size, 1))
        layers1.append(nn.BatchNorm1d(num_filters2))

        #Add in trial range for dropout to determine optimal dropout value
        # self.dp = nn.Dropout(trial.suggest_uniform('dropout_rate',0,1.0))
        layers2.append(nn.Linear(n_conv, 1))
        
        self.layers1 = nn.Sequential(*layers1)
        self.layers2 = nn.Sequential(*layers2)

    def forward(self, x):
        # forward pass
        x = self.layers1(x)
        x = x.flatten(1)
        x = self.layers2(x)
        return x


class pytorch_autoencoder():
    """This is a class that tunes an autoencoder for given data
    """    
    def __init__(self, X:np.array,  config: typing.Type[pytorch_config]):
        """[summary]

        Args:
            X (np.array): Independent Covariates
            config (typing.Type[pytorch_config]):config class specific this is handled by pytorch_general
        """        
        self.X = X
        self.Y = X
        self.config = config
        self.model = "Model has not been tuned yet"
        self.model_paramaters = "Model has not been tuned yet"
        self.model_metric = "Model has not been tuned yet"

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
        """Objective function for Optuna tune. 

        Args:
            trial: Optuna trial
        Raises:
            optuna.exceptions.TrialPruned: Optuna inbuilt pruning 

        Returns:
            float: accuracy of current model, defined by config.METRIC
        """

        loss_function = self.config.LOSS

        # Generate the model.'
        layers = []
        in_features = self.config.IN_FEATURES_START
        out_features = self.config.ENCODE_LEVEL
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        in_features = out_features
        layers.append(nn.Linear(in_features, self.config.CLASSES))
        model = nn.Sequential(*layers).to(self.config.DEVICE)
        
        # model = self.define_model(trial).to(self.config.DEVICE)

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


            model.eval()
            X_val, y_true = train_data
            y_pred =  model(X_val).detach().np()
            accuracy = self.config.METRIC(y_true, y_pred)

            trial.report(accuracy, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return accuracy

    def build_model(self,study):
        """Rebuilds best model from Optuna search

        Args:
            study ([type]): Best study from Optuna search

        Returns:
            [type]: best model
        """ 
        
        class autoencoder_final(nn.Module):
            """General autoencoder class to tune
            """            
            def __init__(self,X,encode_level):
                """[summary]

                Args:
                    X (np.array): Independent Covariates (handled by outside class)
                    encode_level (float): Encode reduction, proportion of variables to reduce to.
                    Must be positive (handled by outside class)
                """                
                super(autoencoder_final,self).__init__()
                self.encoder = nn.Sequential(nn.Linear(X.shape[1],encode_level),nn.ReLU())
                self.decoder = nn.Linear(encode_level,X.shape[1])
            def forward(self,x):
                latent = self.encoder(x)
                decoded = self.decoder(latent)
                decoded = torch.sigmoid(decoded)
                return decoded, latent

        model = autoencoder_final(self.X, self.config.ENCODE_LEVEL).to(self.config.DEVICE)

        #Data
        train_loader, _ = self.data_loader()

        #Loss
        loss_function=nn.MSELoss()

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
                output,_ = model(data)
                loss = loss_function(output, target)
                loss.backward()
                optimizer.step()
        
        return model


    def tune(self):
        """Engages the Optuna search

        Returns:
            [type]: best model
        """ 
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.config.N_TRIALS, timeout=self.config.TIMEOUT)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        model = self.build_model(study)

        self.model_metric = trial.value
        self.model_paramaters = trial.params
        self.model = model

        return model




