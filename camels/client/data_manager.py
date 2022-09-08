from data.datasets import MovieLens, ML100K, ML1M, ML10M, BookCrossing, Amazon, EachMovie, Jester
from camels.database_identifier import Task
from collections import Counter
from scipy.stats import skew, kurtosis
import pandas as pd
import requests
import json

config = json.load(open('connection_settings.json'))
SERVER_URL = config["server-ip"]


# stores information about the data sets
class DataManager:

    def __init__(self, set_type: str, folder_path: str):
        self.folder_path = folder_path
        self.set_type = set_type

    def load(self):
        if self.set_type == "EachMovie":
            return EachMovie(self.folder_path).ratings
        elif self.set_type == "Jester":
            return Jester(self.folder_path).ratings
        elif self.set_type == "ML20M/MLLM":
            return MovieLens(self.folder_path).ratings.drop(columns="timestamp")
        elif self.set_type == "ML100K":
            return ML100K(self.folder_path).ratings.drop(columns="timestamp")
        elif self.set_type == "ML1M":
            return ML1M(self.folder_path).ratings.drop(columns="timestamp")
        elif self.set_type == "ML10M":
            return ML10M(self.folder_path).ratings.drop(columns="timestamp")
        elif self.set_type == "BookCrossing":
            return BookCrossing(self.folder_path).ratings
        elif self.set_type == "Amazon":
            return Amazon(self.folder_path).ratings.drop(columns="timestamp")


# controls preprocessing routines and metadata acquisition
class DataLoader:

    def __init__(self, manager: DataManager = None, any_data: pd.DataFrame = None, prune: bool = True,
                 upload_eval: bool = True, data_set_name: str = None, data_set_type: Task = None):
        self.manager = manager
        self.data = any_data
        self.prune = prune
        self.upload_eval = upload_eval
        self.data_set_name = data_set_name
        self.data_set_type = data_set_type
        self.local_hash = None
        self.metadata_id = None

    def obtain_metadata(self):
        # some data stats
        min_rating = self.data["rating"].min()
        max_rating = self.data["rating"].max()
        mean_rating = self.data["rating"].mean()
        user_counter = Counter(self.data["user"])
        item_counter = Counter(self.data["item"])
        num_users = self.data["user"].unique().size
        num_items = self.data["item"].unique().size
        num_instances = len(self.data)

        # create meta data object
        meta_data = {"num_users": [num_users], "num_items": [num_items],
                     "min_rating": [min_rating.tolist()], "max_rating": [max_rating.tolist()],
                     "mean_rating": [mean_rating.tolist()],
                     "normalized_mean_rating": [
                         ((mean_rating - min_rating) / (max_rating - min_rating)).tolist()],
                     "num_instances": [num_instances],
                     "highest_num_ratings_by_single_user": [user_counter.most_common()[0][1]],
                     "lowest_num_ratings_by_single_user": [user_counter.most_common()[-1][1]],
                     "highest_num_ratings_on_single_item": [item_counter.most_common()[0][1]],
                     "lowest_num_ratings_on_single_item": [item_counter.most_common()[-1][1]],
                     "mean_num_ratings_by_user": [num_instances / num_users],
                     "mean_num_ratings_on_item": [num_instances / num_items],
                     "rating_skew": [skew(self.data["rating"])],
                     "rating_kurtosis": [kurtosis(self.data["rating"])],
                     "rating_standard_deviation": [self.data["rating"].std().tolist()],
                     "rating_variance": [self.data["rating"].var().tolist()]}

        return meta_data

    def preprocess(self):
        # use a lenskit loader to read the data or use the supplied data instead
        if self.manager is not None:
            print(f"Reading data set {self.data_set_name}.")
            self.data = self.manager.load()
        elif self.data is not None:
            print("Using the supplied data.")
        else:
            print("No data could be loaded.")
            return

        # check if data has exactly three cols
        data_cols = list(self.data)
        if len(data_cols) != 3:
            print("Data needs to have exactly three columns.")
            return

        # check if data has user, item and rating cols
        required_cols = ['user', 'item', 'rating']
        if not all(col in data_cols for col in required_cols):
            print("Data needs to have the columns: user, item, rating.")
            return

        # prune data
        if self.prune:
            self.data = self.data[~self.data.drop(columns=["rating"]).duplicated(keep="last")]

            # such that every user has at least five rated items and at most 999
            u_cnt = Counter(self.data["user"])
            sig_users = [k for k in u_cnt if ((u_cnt[k] > 4) and (u_cnt[k] < 1000))]
            self.data = self.data[self.data["user"].isin(sig_users)]

            # i_cnt = Counter(self.data["item"])
            # sig_items = [k for k in i_cnt if i_cnt[k] > 4]
            # self.data = self.data[self.data["item"].isin(sig_items)]

        # get metadata only if evaluation should be uploaded to server
        if self.upload_eval:
            # get the dataframe hash to compare against the database
            self.local_hash = int(pd.util.hash_pandas_object(self.data).sum())

            # send hash to server to check if the data set already exists
            print("Checking if hash exists on server.")
            try:
                response = requests.post(f"{SERVER_URL}check_hash", data={'hash': self.local_hash})
            except requests.exceptions.ConnectionError:
                print("Connection to the server could not be established!")
                return
            if response.status_code != 200:
                print(f"Error: The server returned with status code {response.status_code}.")
                return
            response = json.loads(response.text)
            print(f"Server returned: {response[0]}")

            # if the data exists, set metadata id and return
            if response[1]:
                self.metadata_id = response[2]
                return
            # otherwise calculate metadata of data set
            else:
                # new uploads need a data set name
                if self.data_set_name is None:
                    print("You have to provide a data set name to add metadata to the database.")
                    return

                # get meta data
                meta_data = self.obtain_metadata()
                # append with administrative content
                meta_data["Hash"] = [self.local_hash]
                meta_data["MetadataName"] = [self.data_set_name]

                # send meta data to server
                print("Saving metadata on server.")
                try:
                    response = requests.post(f"{SERVER_URL}save_metadata", data={'metadata': json.dumps(meta_data)})
                except requests.exceptions.ConnectionError:
                    print("Connection to the server could not be established!")
                    return
                if response.status_code != 200:
                    print(f"Error: The server returned with status code {response.status_code}.")
                    return
                response = json.loads(response.text)
                print(f"Server returned: {response[0]}")

                # set meta data id on loader
                self.metadata_id = response[1]
                return
