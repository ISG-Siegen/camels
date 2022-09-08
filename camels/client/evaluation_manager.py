from typing import List
import numpy as np
import requests
import json
import pandas as pd
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_manager import DataLoader
from camels.database_identifier import Algorithm, Metric, lenskit_algorithms, surprise_algorithms
from camels.client.utils import get_base_model
from lenskit import crossfold
from lenskit.algorithms import Recommender
from surprise import Dataset, Reader

pd.options.mode.chained_assignment = None

config = json.load(open('connection_settings.json'))
SERVER_URL = config["server-ip"]


# stores data and evaluation routines for client evaluation procedures
class Evaluator:

    def __init__(self, loader: DataLoader, algos: List[Algorithm], metrics: List[Metric],
                 override_runs: bool = False, always_perform_runs: bool = False):
        self.loader = loader
        self.algos = algos
        self.metrics = metrics
        self.override_runs = override_runs
        self.always_perform_runs = always_perform_runs

    # evaluates a list of algorithms on a list of metrics for a single data set
    def evaluate(self):
        print("Preprocessing data.")
        self.loader.preprocess()

        print("Checking data status.")
        existing_runs = None

        # checks if runs exist for the data on the given algorithms and metrics
        if self.loader.upload_eval:
            print("Checking metadata run status.")

            algo_ids = [algo.value for algo in self.algos]
            metric_ids = [metric.value for metric in self.metrics]

            try:
                response = requests.post(f"{SERVER_URL}check_data_status",
                                         data={'metadata_id': self.loader.metadata_id, 'algo_ids': json.dumps(algo_ids),
                                               'task_id': self.loader.data_set_type.value,
                                               'metric_ids': json.dumps(metric_ids)})
            except requests.exceptions.ConnectionError:
                print("Connection to the server could not be established!")
                return
            if response.status_code != 200:
                print(f"Error: The server returned with status code {response.status_code}.")
                return

            existing_runs = np.array(json.loads(response.text))
            for run in existing_runs:
                print(f"Server returned: Run for {Algorithm(run[0])} on {Metric(run[1])} exists.")

        print("Evaluating algorithms.")

        for algo in self.algos:
            if existing_runs is not None and len(existing_runs) > 0:
                algo_existing_runs = existing_runs[existing_runs[:, 0] == algo.value]
                algo_metric_combos = np.array([[algo.value, m.value] for m in self.metrics])
                all_combos_exist = np.isin(algo_metric_combos, algo_existing_runs).all()
                if all_combos_exist:
                    print(f"All algorithm and metric combos for {algo.name} already exist.")
                    if self.always_perform_runs or self.override_runs:
                        print("Performing evaluation anyway due to settings.")
                    else:
                        print("Aborting evaluation for algorithm due to settings.")
                        continue

            print(f"Started evaluation for {algo.name}.")
            scores = {metric.name: [] for metric in self.metrics}
            # split data using lenskit cross-fold validation

            for d_train, d_test in crossfold.partition_users(self.loader.data, 1, crossfold.SampleFrac(0.2)):
                d_train.loc[d_train["rating"] == 0, "rating"] = 0.000001
                # choose model
                recommender = get_base_model(algo)
                if algo in lenskit_algorithms:
                    print(f"Evaluating lenskit algorithm.")
                    recommender = Recommender.adapt(recommender)
                    fit_start_time = time.time()
                    recommender.fit(d_train)
                    fit_duration = time.time() - fit_start_time
                    print(f"Algorithm fitted in {fit_duration} seconds.")
                    prediction = recommender.predict(d_test.drop(columns="rating"))
                elif algo in surprise_algorithms:
                    print(f"Evaluating surprise algorithm.")
                    reader = Reader(rating_scale=(self.loader.data["rating"].min(), self.loader.data["rating"].max()))
                    d_train = Dataset.load_from_df(d_train, reader).build_full_trainset()
                    fit_start_time = time.time()
                    recommender.fit(d_train)
                    fit_duration = time.time() - fit_start_time
                    print(f"Algorithm fitted in {fit_duration} seconds.")
                    prediction = [recommender.predict(getattr(row, "user"), getattr(row, "item"))
                                  for row in d_test.itertuples()]
                    prediction = pd.DataFrame(prediction)["est"]
                else:
                    print("The algorithm is unsupported.")
                    return

                # build score dictionary
                for metric in self.metrics:
                    if metric == Metric.Runtime:
                        score = fit_duration
                    elif metric == Metric.NMAE:
                        score = mean_absolute_error(prediction, d_test["rating"]) \
                                / d_test["rating"].mean()
                    elif metric == Metric.NRMSE:
                        score = mean_squared_error(prediction, d_test["rating"], squared=False) \
                                / d_test["rating"].mean()
                    elif metric == Metric.MAE:
                        score = mean_absolute_error(prediction, d_test["rating"])
                    elif metric == Metric.RMSE:
                        score = mean_squared_error(prediction, d_test["rating"], squared=False)
                    else:
                        score = np.Inf

                    print(f"{metric.name} score: {score}.")
                    scores[metric.name].append(score)

            final_scores = {k: np.array(v).mean() for k, v in scores.items()}

            # organize evaluations to database structure
            algo_evaluations = []
            for metric, score in final_scores.items():
                # only upload if override is true or the run does not exist yet
                if self.override_runs or not \
                        any([([algo.value, Metric[metric].value] == entry).all() for entry in existing_runs]):
                    algo_evaluations.append({"MetadataID": [self.loader.metadata_id], "AlgorithmID": [algo.value],
                                             "TaskID": [self.loader.data_set_type.value],
                                             "MetricID": [Metric[metric].value], "Score": [float(score)]})

            if algo_evaluations is not None and len(algo_evaluations) > 0 and self.loader.upload_eval:
                print("Uploading runs.")
                try:
                    response = requests.post(f"{SERVER_URL}save_runs", data={'evals': json.dumps(algo_evaluations)})
                except requests.exceptions.ConnectionError:
                    print("Connection to the server could not be established!")
                    return
                if response.status_code != 200:
                    print(f"Error: The server returned with status code {response.status_code}.")
                    return
                print(f"Server returned: {json.loads(response.text)}")
            else:
                print("Skipping upload.")

        print("Evaluation finished.")
        return
