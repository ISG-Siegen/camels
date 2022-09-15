from typing import Union, Tuple
import pandas as pd
import time

from camels.client.client_database_identifier import Algorithm

from xgboost import XGBRegressor


def get_base_model(algorithm: Algorithm) -> Union[XGBRegressor]:
    if algorithm == Algorithm.XGBoostRegression:
        return XGBRegressor()
    else:
        print("Exception: The algorithm could not be identified.")
        raise Exception


def fit_and_predict(train: pd.DataFrame, test: pd.DataFrame, algorithm: Algorithm) -> Tuple[pd.DataFrame, float, float]:
    print(f"Evaluating sklearn algorithm {algorithm.name}.")
    recommender = get_base_model(algorithm)

    fit_start_time = time.time()
    recommender.fit(train.drop(columns=["rating"]), train["rating"].values.ravel())
    fit_duration = time.time() - fit_start_time
    print(f"Algorithm fitted in {fit_duration} seconds.")
    predict_start_time = time.time()
    prediction = recommender.predict(test.drop(columns=["rating"]))
    predict_duration = time.time() - predict_start_time
    print(f"Algorithm predicted in {predict_duration} seconds.")

    return prediction, fit_duration, predict_duration


def train_best_model(data: pd.DataFrame, algorithm: Algorithm) -> XGBRegressor:
    # train the best recommender based on predictions
    recommender = get_base_model(algorithm)
    if recommender is not None:
        print(f"Training best algorithm {algorithm.name} on supplied data.")
        recommender.fit(data.drop(columns=["rating"]), data["rating"].values.ravel())
        print(f"Done.")
        return recommender
    else:
        print(f"Exception: Algorithm {algorithm.name} could not be built.")
        raise Exception
