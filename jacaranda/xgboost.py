import pandas as pd
import numpy
import xgboost as xgb
import sklearn.metrics
import optuna
import typing, os


class xgboost_config:
    '''  General configuration class for PyTorch hyper tuning to pass to each of 
    the hyperparamter tuning class.
    '''
    def __init__(
        self,
        X:numpy.array,
        Y:numpy.array,
        metric:typing.Callable = sklearn.metrics.mean_squared_error,
        validation_split: float = 0.2,
        seed: float = 123,
        dir:str = os.getcwd(),
        n_trials: int = 75,
        timeout: float = 600,
        optimize_direction: str = "minimize",
        objective = "reg:linear"
    ):
        """
        Args:
            X (numpy.array): Independent Covariates

            Y (numpy.array): Dependent Covariates (may be expressed as One hot encode). Note,
            needs to have two-dimensional shape.

            validation_split (float, optional): Validation Split, must be between 0 and 
            1 Defaults to 0.2.

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

            objective (str, optional): Learning task parameter for xgboost to minimise.
            This also outlines the type of predition type. Defaults to 'reg:linear'.

        """        
        self.VALIDATION_SPLIT = validation_split
        self.SEED = seed
        self.DIR = dir
        self.CLASSES = Y.shape[1]
        self.IN_FEATURES_START = X.shape[-1]
        self.N_TRIALS = n_trials
        self.TIMEOUT = timeout
        self.METRIC = metric
        self.OPTIMIZE_DIRECTION = optimize_direction
        self.OBJECTIVE = objective


class xgboost_general():
    """General XGboost tuning class. 
    """    
    def __init__(self, 
                 X:numpy.array, 
                 Y:numpy.array, 
                 config: typing.Type[xgboost_config], ):
        """
        Args:
            X (numpy.array): Independent Covariates
            Y (numpy.array): Dependent Covariates (may be expressed as One hot encode). 
            config (typing.Type[xgboost_config]): config class specific to X,Y
        """        

        self.X = X
        self.Y = Y
        self.config = config
        self.model = "Model has not been tuned yet"
        self.model_paramaters = "Model has not been tuned yet"
        self.model_metric = "Model has not been tuned yet"

    def data_loader(self):
        # Creating data indices for training and validation splits:
        dataset_size = len(self.X)
        indices = list(range(dataset_size))
        split = int(numpy.floor(self.config.VALIDATION_SPLIT * dataset_size))
        numpy.random.seed(self.config.SEED)
        numpy.random.shuffle(indices)

        valid_y = self.Y[:split]
        dtrain = xgb.DMatrix(self.X[split:], label=self.Y[split:])
        dvalid = xgb.DMatrix(self.X[:split], label=valid_y)
        
        return dtrain, dvalid, valid_y

    def objective(self,trial):
        """Objective function for Optuna tune. 

        Args:
            trial: Optuna trial

        Returns:
            float: accuracy of current model, defined by config.METRIC
        """     
        dtrain, dvalid, valid_y = self.data_loader()

        param = {
            "verbosity": 0,
            "objective": self.config.OBJECTIVE,
            # use exact for small dataset.
            "tree_method": "exact",
            # defines booster, gblinear for linear functions.
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            # L2 regularization weight.
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            # L1 regularization weight.
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            # sampling ratio for training data.
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        }

        if param["booster"] in ["gbtree", "dart"]:
            # maximum depth of the tree, signifies complexity of the tree.
            param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
            # minimum child weight, larger the term more conservative the tree.
            param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            # defines how selective algorithm is.
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

        bst = xgb.train(param, dtrain)
        preds = bst.predict(dvalid)
        pred_labels = numpy.rint(preds)
        accuracy = self.config.METRIC(valid_y, pred_labels)
        return accuracy

    def tune(self):
        """Engages the Optuna search

        Returns:
            [type]: best model
        """   
        study = optuna.create_study(direction=self.config.OPTIMIZE_DIRECTION)
        study.optimize(self.objective, n_trials=self.config.N_TRIALS, timeout=self.config.TIMEOUT)

        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        dtrain, _, _ = self.data_loader()

        model = xgb.train(study.best_trial.params, dtrain)

        self.model_metric = trial.value
        self.model_paramaters = trial.params
        self.model = model

        return model

