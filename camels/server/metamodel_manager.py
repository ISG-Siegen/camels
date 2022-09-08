import pandas as pd
import numpy as np
import sqlite3 as sl
import json
import pickle
import os
import copy
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut


# trains and saves the meta learner
def train_meta_learner(metric_ids, task_ids, learners):
    task_ids = json.loads(task_ids)
    metric_ids = json.loads(metric_ids)
    learners = json.loads(learners)

    # load required tables
    con = sl.connect('camels.db')
    runs = pd.read_sql_query(f"SELECT * FROM Runs", con)
    metadata = pd.read_sql_query(f"SELECT * FROM Metadata", con)
    config = pd.read_sql_query(f"SELECT * FROM Config", con)
    version_id = config["VersionID"][0]

    # saves the return messages for the client
    final_results = []

    # get the number of metadata features
    metadata_features = len(list(metadata.drop(columns=["MetadataID", "Hash", "MetadataName"])))

    # loop all tasks and metrics
    for task in task_ids:
        for metric in metric_ids:
            regular_metric = -1
            if metric == 1:
                regular_metric = 3
            elif metric == 2:
                regular_metric = 4

            # arrange the metadata and algorithm score
            train_md = pd.DataFrame()
            train_md_regular = pd.DataFrame()
            for metadata_id in runs["MetadataID"].unique():
                eval_runs = runs.loc[(runs["MetadataID"] == metadata_id) &
                                     (runs["TaskID"] == task) &
                                     (runs["MetricID"] == metric)].copy()
                eval_runs_regular = None
                if regular_metric >= 0:
                    eval_runs_regular = runs.loc[(runs["MetadataID"] == metadata_id) &
                                                 (runs["TaskID"] == task) &
                                                 (runs["MetricID"] == regular_metric)].copy()
                metadata_run = metadata.loc[metadata["MetadataID"] == metadata_id].copy()
                metadata_run.drop(columns=["MetadataID", "Hash", "MetadataName"], inplace=True)
                metadata_run_regular = None
                if regular_metric >= 0:
                    metadata_run_regular = metadata.loc[metadata["MetadataID"] == metadata_id].copy()
                    metadata_run_regular.drop(columns=["MetadataID", "Hash", "MetadataName"], inplace=True)
                for run in eval_runs.loc[runs["MetadataID"] == metadata_id].itertuples():
                    metadata_run[f"{run.AlgorithmID}"] = run.Score
                if regular_metric >= 0:
                    for run in eval_runs_regular.loc[runs["MetadataID"] == metadata_id].itertuples():
                        metadata_run_regular[f"{run.AlgorithmID}"] = run.Score
                train_md = pd.concat([train_md, metadata_run])
                if regular_metric >= 0:
                    train_md_regular = pd.concat([train_md_regular, metadata_run_regular])

            # impute missing values
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            train_md[:] = imp.fit_transform(train_md)

            # get and fit the models
            for learner in learners:
                if learner == "RandomForestRegression":
                    base_model = RandomForestRegressor(500)
                    print("Starting evaluation for Random Forest Regressor.")
                elif learner == "LinearRegression":
                    base_model = LinearRegression()
                    print("Starting evaluation for Linear Regressor.")
                else:
                    print(f"The model {learner} is unknown. Skipping.")
                    continue

                # validate the model performance
                print("Evaluating with n-repetition leave-one-out cross-validation.")
                eval_results = []
                n_rep = 50
                for i in range(n_rep):
                    if (i + 1) % 5 == 0:
                        print(f"Evaluation step {i + 1}/{n_rep}...")
                    loo = LeaveOneOut()
                    results_per_model = {k: [[], []] for k in ["VBA", "SBA", "EPM"]}
                    for train_index, test_index in loo.split(train_md):
                        train_eval, test_eval = train_md.iloc[train_index], train_md.iloc[test_index]

                        train_eval_regular, test_eval_regular = None, None
                        if regular_metric >= 0:
                            train_eval_regular, test_eval_regular = train_md_regular.iloc[train_index], \
                                                                    train_md_regular.iloc[test_index]

                        x_train_eval = train_eval.iloc[:, :metadata_features].to_numpy()
                        y_train_eval = train_eval.iloc[:, metadata_features:].to_numpy()

                        x_test_eval = test_eval.iloc[:, :metadata_features].to_numpy()
                        y_test_eval = test_eval.iloc[:, metadata_features:].to_numpy().flatten()

                        y_train_eval_regular, y_test_eval_regular = None, None
                        if regular_metric >= 0:
                            y_train_eval_regular = train_eval_regular.iloc[:, metadata_features:].to_numpy()
                            y_test_eval_regular = test_eval_regular.iloc[:, metadata_features:].to_numpy().flatten()

                        predicted_idx_list = []

                        # get SBA
                        idx_sba = np.argmin(y_train_eval.sum(axis=0))
                        if regular_metric >= 0:
                            idx_sba = np.argmin(y_train_eval_regular.sum(axis=0))
                        predicted_idx_list.append(("SBA", idx_sba))

                        # get VBA
                        idx_vba = np.where(np.min(y_test_eval) == y_test_eval)[0].tolist()
                        if regular_metric >= 0:
                            idx_vba = np.where(np.min(y_test_eval_regular) == y_test_eval_regular)[0].tolist()
                        predicted_idx_list.append(("VBA", idx_vba))

                        # get meta-learner
                        epm = copy.deepcopy(base_model)
                        epm.fit(x_train_eval, y_train_eval)
                        epm_predictions = epm.predict(x_test_eval)
                        idx_ml = np.argmin(epm_predictions)
                        predicted_idx_list.append(("EPM", idx_ml))

                        # get error for models

                        for model_name, idx in predicted_idx_list:
                            if isinstance(idx, list):
                                sel_idx = idx[0]
                            else:
                                sel_idx = idx
                            if regular_metric >= 0:
                                results_per_model[model_name][0].append(y_test_eval_regular[sel_idx])
                            else:
                                results_per_model[model_name][0].append(y_test_eval[sel_idx])
                            results_per_model[model_name][1].append(idx)

                    res_table = []
                    res_columns = ["Selection Method", "Average Error", "Selection Accuracy"]

                    vba_res = results_per_model.pop("VBA")
                    oracle_indices = vba_res[1]
                    res_table.append(("VBA", np.mean(vba_res[0]), 1))
                    # print("VBA", oracle_indices)
                    for model_name, (scores, predicted_indices) in results_per_model.items():
                        # print(model_name, predicted_indices)
                        sel_acc = sum(i in j for i, j in zip(predicted_indices, oracle_indices)) / len(oracle_indices)
                        res_table.append((model_name, np.mean(scores), sel_acc))

                    res_df = pd.DataFrame(res_table, columns=res_columns)
                    eval_results.append(res_df)

                eval_df = pd.concat(eval_results, axis=1)
                mean_res = eval_df.drop(columns=["Selection Method"]).groupby(level=0, axis=1).mean()
                mean_res = pd.concat([mean_res, eval_df["Selection Method"].iloc[:, 0]], axis=1)
                print(mean_res)
                final_results.append(mean_res.to_json())

                # train the final model
                x_t = train_md.iloc[:, :metadata_features]
                y_t = train_md.iloc[:, metadata_features:]
                final_model = copy.deepcopy(base_model)
                final_model.fit(x_t, y_t)

                # create folder to store models
                if not os.path.exists(f"./{version_id}_{learner}/"):
                    os.makedirs(f"./{version_id}_{learner}/")

                # save one model per task per metric
                print(f"Saving model for {learner}.")
                pickle.dump(final_model, open(f"./{version_id}_{learner}/{task}_{metric}", 'wb'))

    # return the result string list
    return final_results


def predict_with_meta_learner(meta_data, metric_id, learner, task_id):
    meta_data = json.loads(meta_data)
    meta_data = pd.DataFrame(meta_data)
    metric_id = json.loads(metric_id)
    learner = json.loads(learner)
    task_id = json.loads(task_id)

    con = sl.connect('camels.db')
    config = pd.read_sql_query(f"SELECT * FROM Config", con)
    version_id = config["VersionID"][0]

    model = pickle.load(open(f"./{version_id}_{learner}/{task_id}_{metric_id}", "rb"))
    prediction = model.predict(meta_data)

    return prediction.tolist()
