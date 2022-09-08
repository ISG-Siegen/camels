import pandas as pd
import sqlite3 as sl
import json


# check if runs exist for the given metadata on algorithms, tasks and metrics
def check_data_status(metadata_id, algo_ids, task_id, metric_ids):
    con = sl.connect('camels.db')
    remote_runs = pd.read_sql_query(f"SELECT * FROM Runs", con)

    # filter runs by parameters
    data_runs = remote_runs.loc[
        (remote_runs["MetadataID"] == int(metadata_id)) &
        (remote_runs["AlgorithmID"].isin(json.loads(algo_ids))) &
        (remote_runs["TaskID"] == int(task_id)) &
        (remote_runs["MetricID"].isin(json.loads(metric_ids)))]

    return data_runs[["AlgorithmID", "MetricID"]].values.tolist()


# write completed runs to database
def save_runs(evaluations):
    con = sl.connect('camels.db')
    remote_runs = pd.read_sql_query(f"SELECT * FROM Runs", con)

    # set the run id
    run_id = -1
    if len(remote_runs) > 0:
        run_id = remote_runs["RunID"].values.max()

    # read the submitted evaluations and write them to the database
    evaluations = json.loads(evaluations)
    evaluation_df = pd.DataFrame()
    for evaluation in evaluations:
        evaluation = pd.DataFrame(evaluation)
        copy_run = remote_runs[(remote_runs["MetadataID"].isin(evaluation["MetadataID"])) &
                               (remote_runs["AlgorithmID"].isin(evaluation["AlgorithmID"])) &
                               (remote_runs["TaskID"].isin(evaluation["TaskID"])) &
                               (remote_runs["MetricID"].isin(evaluation["MetricID"]))]
        if len(copy_run) > 0:
            remote_runs = remote_runs[remote_runs["RunID"] != copy_run["RunID"].values[0]]
            evaluation["RunID"] = copy_run["RunID"].values[0]
            evaluation_df = pd.concat([evaluation_df, evaluation])
        else:
            run_id += 1
            evaluation["RunID"] = run_id
            evaluation_df = pd.concat([evaluation_df, evaluation])

    # update list of runs
    updated_runs = pd.concat([remote_runs, evaluation_df])
    pd.DataFrame(updated_runs).to_sql("Runs", con, if_exists="replace", index=False)

    return "Runs were saved."
