import pandas as pd
import requests
import json
from evaluation_manager import DataLoader, Evaluator
from data_manager import DataManager
from model_manager import ModelManager
from camels.database_identifier import Algorithm, Metric, Task, lenskit_algorithms, surprise_algorithms
from typing import List, Union, Dict

# turn off warnings to avoid cluttering console
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

config = json.load(open('connection_settings.json'))
SERVER_URL = config["server-ip"]

# base folder for the data sets
data_folder = "../../data/"

# example selector for supported extremely large data sets
extremely_large_data_selector = {
    "ml20m": DataManager("ML20M/MLLM", f'{data_folder}ml-20m'),
    "amazon-automotive": DataManager("Amazon", f'{data_folder}amazon-automotive'),
    "amazon-books": DataManager("Amazon", f'{data_folder}amazon-books'),
    "amazon-cell-phones-and-accessories": DataManager("Amazon", f'{data_folder}amazon-cell-phones-and-accessories'),
    "amazon-clothing-shoes-and-jewelry": DataManager("Amazon", f'{data_folder}amazon-clothing-shoes-and-jewelry'),
    "amazon-electronics": DataManager("Amazon", f'{data_folder}amazon-electronics'),
    "amazon-home-and-kitchen": DataManager("Amazon", f'{data_folder}amazon-home-and-kitchen'),
    "amazon-movies-and-tv": DataManager("Amazon", f'{data_folder}amazon-movies-and-tv'),
    "amazon-pet-supplies": DataManager("Amazon", f'{data_folder}amazon-pet-supplies'),
    "amazon-sports-and-outdoors": DataManager("Amazon", f'{data_folder}amazon-sports-and-outdoors'),
    "amazon-tools-and-home-improvement": DataManager("Amazon", f'{data_folder}amazon-tools-and-home-improvement'),
    "amazon-toys-and-games": DataManager("Amazon", f'{data_folder}amazon-toys-and-games'),
}

# example selector for supported large data sets
large_data_selector = {
    "ml10m": DataManager("ML10M", f'{data_folder}ml-10M100K'),
    "amazon-cds-and-vinyl": DataManager("Amazon", f'{data_folder}amazon-cds-and-vinyl'),
    "amazon-grocery-and-gourmet-food": DataManager("Amazon", f'{data_folder}amazon-grocery-and-gourmet-food'),
    "amazon-kindle-store": DataManager("Amazon", f'{data_folder}amazon-kindle-store'),
    "amazon-office-products": DataManager("Amazon", f'{data_folder}amazon-office-products'),
    "amazon-patio-lawn-and-garden": DataManager("Amazon", f'{data_folder}amazon-patio-lawn-and-garden'),
}

# example selector for supported small data sets
small_data_selector = {
    "eachmovie": DataManager("EachMovie", f'{data_folder}eachmovie'),
    "jester3": DataManager("Jester", f'{data_folder}jester3'),
    "jester4": DataManager("Jester", f'{data_folder}jester4'),
    "ml_lm": DataManager("ML20M/MLLM", f'{data_folder}ml-lm'),
    "ml100k": DataManager("ML100K", f'{data_folder}ml-100k'),
    "ml1m": DataManager("ML1M", f'{data_folder}ml-1m'),
    "book_crossing": DataManager("BookCrossing", f'{data_folder}book-crossing'),
    "amazon-all-beauty": DataManager("Amazon", f'{data_folder}amazon-all-beauty'),
    "amazon-appliances": DataManager("Amazon", f'{data_folder}amazon-appliances'),
    "amazon-arts-crafts-and-sewing": DataManager("Amazon", f'{data_folder}amazon-arts-crafts-and-sewing'),
    "amazon-digital-music": DataManager("Amazon", f'{data_folder}amazon-digital-music'),
    "amazon_fashion": DataManager("Amazon", f'{data_folder}amazon-fashion'),
    "amazon-gift-cards": DataManager("Amazon", f'{data_folder}amazon-gift-cards'),
    "amazon-industrial-and-scientific": DataManager("Amazon", f'{data_folder}amazon-industrial-and-scientific'),
    "amazon-luxury-beauty": DataManager("Amazon", f'{data_folder}amazon-luxury-beauty'),
    "amazon-magazine-subscriptions": DataManager("Amazon", f'{data_folder}amazon-magazine-subscriptions'),
    "amazon-musical-instruments": DataManager("Amazon", f'{data_folder}amazon-musical-instruments'),
    "amazon-prime-pantry": DataManager("Amazon", f'{data_folder}amazon-prime-pantry'),
    "amazon-software": DataManager("Amazon", f'{data_folder}amazon-software'),
    "amazon-video-games": DataManager("Amazon", f'{data_folder}amazon-video-games')
}


# calls server function to initially populate database
def c_populate_database():
    print("Populating database on server.")
    try:
        response = requests.post(f"{SERVER_URL}populate_database")
    except requests.exceptions.ConnectionError:
        print("Connection to the server could not be established!")
        return
    if response.status_code != 200:
        print(f"Error: The server returned with status code {response.status_code}.")
        return
    print(f"Server returned: {json.loads(response.text)}")


# starts the evaluation pipeline
def c_evaluate_algorithms(algos: List[Algorithm], metrics: List[Metric], task: Task,
                          data_selector: Union[pd.DataFrame, Dict], override_runs: bool, always_perform_runs: bool):
    """
    @param algos: list of algorithms that should be evaluated
    @param metrics: list of metrics the algorithms should be evaluated for
    @param task: the prediction task
    @param data_selector: either a data selector dictionary or a raw data set to evaluate on
    @param override_runs: should runs in the database be replaced
    @param always_perform_runs: should runs be performed even if they already exist
    """

    if isinstance(data_selector, Dict):
        for ds_name, ds_manager in data_selector.items():
            loader = DataLoader(manager=ds_manager, data_set_name=ds_name, data_set_type=task)
            evaluator = Evaluator(loader, algos, metrics, override_runs, always_perform_runs)
            evaluator.evaluate()
    elif isinstance(data_selector, pd.DataFrame):
        loader = DataLoader(any_data=data_selector, data_set_name="custom", data_set_type=task)
        evaluator = Evaluator(loader, algos, metrics, override_runs, always_perform_runs)
        evaluator.evaluate()


# calls for the server to train the meta learner
def c_train_meta_learner(metrics: List[Metric], tasks: List[Task], learners: List[str]):
    """
    @param metrics: list of metrics the meta-learner(s) should be trained for
    @param tasks: list of tasks the meta-learner(s) should be trained for
    @param learners: list of meta-learner algorithms to train
    """

    metric_ids = [metric.value for metric in metrics]
    task_ids = [task.value for task in tasks]

    print("Training meta learner.")
    try:
        response = requests.post(f"{SERVER_URL}train_meta_learner", data={'metric_ids': json.dumps(metric_ids),
                                                                          'task_ids': json.dumps(task_ids),
                                                                          'learners': json.dumps(learners)})
    except requests.exceptions.ConnectionError:
        print("Connection to the server could not be established!")
        return
    if response.status_code != 200:
        print(f"Error: The server returned with status code {response.status_code}.")
        return
    response_list = json.loads(response.text)
    for eval_cv in response_list:
        print(f"Server returned:\n{pd.read_json(eval_cv)}")


# calls the server to return a prediction for the selected data
def c_predict_with_meta_learner(metric: Metric, learner: str, task: Task, data_selector: Union[pd.DataFrame, Dict]):
    """
    @param metric: the metric for which the meta-learner should predict
    @param learner: the type of meta-learner to use
    @param task: the prediction task to solve for
    @param data_selector: either a data selector dictionary or a raw data set to predict for
    """

    loader = None
    if isinstance(data_selector, Dict):
        for ds_name, ds_manager in data_selector.items():
            loader = DataLoader(manager=ds_manager, data_set_name=ds_name, data_set_type=task, upload_eval=False)
    elif isinstance(data_selector, pd.DataFrame):
        loader = DataLoader(any_data=data_selector, data_set_name="custom", data_set_type=task, upload_eval=False)

    manager = ModelManager(loader, metric, learner, task)
    manager.predict_algo_performance()
    manager.return_best_model()


if __name__ == "__main__":
    # populate the database with tables and administrative entries
    c_populate_database()

    # evaluate algorithms locally and donate metadata sets to the server
    c_evaluate_algorithms(algos=lenskit_algorithms + surprise_algorithms,
                          metrics=[Metric.Runtime, Metric.NMAE, Metric.NRMSE, Metric.MAE, Metric.RMSE],
                          task=Task.ERP, data_selector={"ml100k": DataManager("ML100K", f'{data_folder}ml-100k')},
                          override_runs=False, always_perform_runs=False)

    # train a meta-learner on the server
    c_train_meta_learner(metrics=[Metric.NMAE, Metric.NRMSE], tasks=[Task.ERP], learners=["RandomForestRegression"])

    # predict metadata performance with meta-learner and return best model
    c_predict_with_meta_learner(metric=Metric.NRMSE, learner="RandomForestRegression", task=Task.ERP,
                                data_selector={"ml100k": DataManager("ML100K", f'{data_folder}ml-100k')})
