import pandas as pd
import numpy as np

import xgboost as xgb
import sklearn.metrics

import optuna


class xgboost_regression():
    def __init__(self, X, Y, config):
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
        split = int(np.floor(self.config.VALIDATION_SPLIT * dataset_size))
        np.random.seed(self.config.SEED)
        np.random.shuffle(indices)

        valid_y = self.Y[:split]
        dtrain = xgb.DMatrix(self.X[split:], label=self.Y[split:])
        dvalid = xgb.DMatrix(self.X[:split], label=valid_y)
        
        return dtrain, dvalid, valid_y

    def objective(self,trial):
        
        dtrain, dvalid, valid_y = self.data_loader()

        param = {
            "verbosity": 0,
            "objective": "reg:linear",
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
        pred_labels = np.rint(preds)
        accuracy = self.config.METRIC(valid_y, pred_labels)
        return accuracy

    def tune(self):
        study = optuna.create_study(direction="minimize")
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

