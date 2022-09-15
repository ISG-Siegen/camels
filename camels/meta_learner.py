from typing import Union, Any, List
import numpy as np
import pandas as pd
from pathlib import Path
from camels.server.server_database_identifier import Learner
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import copy
import os
import pickle


def get_base_algo(learner: Learner) -> Union[RandomForestRegressor, LinearRegression]:
    if learner == Learner.RandomForest:
        return RandomForestRegressor(1000)
    elif learner == Learner.LinearRegression:
        return LinearRegression()
    else:
        print(f"Exception: The algorithm {learner.name} is unsupported.")
        raise Exception


def meta_fit_and_predict(x_train: np.array, y_train: np.array, x_test: np.array, learner: Learner) -> Any:
    # fit meta-learner and predict item with best score
    epm = copy.deepcopy(get_base_algo(learner))
    epm.fit(x_train, y_train)
    epm_predictions = epm.predict(x_test)
    return np.argmin(epm_predictions)


def meta_train_best_model(x_train: np.array, y_train: np.array, learner: Learner,
                          save_dir: Path, save_file: Path) -> None:
    # fit meta-learner on all data
    epm = copy.deepcopy(get_base_algo(learner))
    epm.fit(x_train, y_train)

    # create folder to store models
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(Path(save_dir / save_file), 'wb') as f:
        pickle.dump(epm, f)

    return


def meta_predict(data: pd.DataFrame, save_path: Path) -> List:
    with open(save_path, 'rb') as f:
        model = pickle.load(f)
    return model.predict(data).tolist()
