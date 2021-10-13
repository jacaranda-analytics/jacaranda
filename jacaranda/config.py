import os, torch, numpy, sklearn



class config:
    def __init__(
        self,
        X:numpy.array,
        Y:numpy.array,
        loss=torch.nn.MSELoss(),
        device: torch.device = torch.device("cpu"),
        metric:function = sklearn.metrics.mean_squared_error,
        batchsize: int = 100,
        validation_split: float = 0.2,
        epochs: int = 20,
        seed: float = 123,
        dir:str = os.getcwd(),
        n_trials: int = 75,
        timeout: float = 600,
    ):
        """

        General configuration class to pass to each of the hyperparamter tuning class.

        Args:
            X (numpy.array): Independent Covariates

            Y (numpy.array): Dependent Covariates (may be expressed as One hot encode)

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

        """        ''''''
        self.BATCHSIZE = batchsize
        self.VALIDATION_SPLIT = validation_split
        self.EPOCHS = epochs
        self.SEED = seed
        self.DIR = dir
        self.CLASSES = len([Y[0]])
        self.LOSS = loss
        self.DEVICE = device
        self.IN_FEATURES_START = X.shape[1]
        self.N_TRIALS = n_trials
        self.TIMEOUT = timeout
        self.METRIC = metric
