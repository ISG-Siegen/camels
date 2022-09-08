import requests
import json
from data_manager import DataLoader
from camels.database_identifier import Algorithm, Metric, Task, lenskit_algorithms, surprise_algorithms
from camels.client.utils import get_base_model
from lenskit.algorithms import Recommender
from surprise import Dataset, Reader

config = json.load(open('connection_settings.json'))
SERVER_URL = config["server-ip"]


# stores information about predictions from the server
class ModelManager:

    def __init__(self, loader: DataLoader, metric: Metric, learner: str, task: Task):
        self.loader = loader
        self.metric = metric
        self.learner = learner
        self.task = task
        self.predicted_algo_performance = None
        self.best_model = None

    def predict_algo_performance(self):
        self.loader.preprocess()
        meta_data = self.loader.obtain_metadata()

        print("Predicting with meta learner.")
        try:
            response = requests.post(f"{SERVER_URL}predict_with_meta_learner",
                                     data={'meta_data': json.dumps(meta_data),
                                           'metric_id': json.dumps(self.metric.value),
                                           'learner': json.dumps(self.learner),
                                           'task_id': json.dumps(self.task.value)})
        except requests.exceptions.ConnectionError:
            print("Connection to the server could not be established!")
            return
        if response.status_code != 200:
            print(f"Error: The server returned with status code {response.status_code}.")
            return
        response_pred = json.loads(response.text)

        predicted_algo_performance = {}
        for algo_id, score in enumerate(response_pred[0]):
            predicted_algo_performance[Algorithm(algo_id)] = score

        self.predicted_algo_performance = predicted_algo_performance

        print(f"Predicted best algorithm with {self.learner} meta learner.\n"
              f"Best algorithm for metric {self.metric} on task {self.task} predicted to be "
              f"{min(self.predicted_algo_performance, key=self.predicted_algo_performance.get)}.")

    def return_best_model(self):
        best_algo = min(self.predicted_algo_performance, key=self.predicted_algo_performance.get)

        recommender = get_base_model(best_algo)
        if recommender is not None:
            if best_algo in lenskit_algorithms:
                recommender = Recommender.adapt(recommender)
                print(f"Training best algorithm.")
                recommender.fit(self.loader.data)
                print(f"Done.")
            elif best_algo in surprise_algorithms:
                reader = Reader(rating_scale=(self.loader.data["rating"].min(), self.loader.data["rating"].max()))
                d_train = Dataset.load_from_df(self.loader.data, reader).build_full_trainset()
                print(f"Training best algorithm.")
                recommender.fit(d_train)
                print(f"Done.")
        else:
            print(f"Algorithm {best_algo} could not be built.")

        self.best_model = recommender
