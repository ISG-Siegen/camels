import sqlite3 as sl
import pandas as pd
from camels.database_identifier import Algorithm, Metric, Task
import time


# generates required tables
def generate_tables():
    con = sl.connect('camels.db')
    cur = con.cursor()

    cur.execute("CREATE TABLE IF NOT EXISTS Config("
                "VersionID integer PRIMARY KEY)")

    cur.execute("CREATE TABLE IF NOT EXISTS Runs("
                "RunID integer PRIMARY KEY,"
                "MetadataID integer,"
                "AlgorithmID integer,"
                "TaskID integer,"
                "MetricID integer,"
                "Score real)")

    cur.execute("CREATE TABLE IF NOT EXISTS Metadata("
                "MetadataID integer PRIMARY KEY,"
                "Hash integer UNIQUE,"
                "MetadataName string,"
                "num_users integer,"
                "num_items integer,"
                "min_rating integer,"
                "max_rating integer,"
                "mean_rating real,"
                "normalized_mean_rating real,"
                "num_instances integer,"
                "highest_num_ratings_by_single_user integer,"
                "lowest_num_ratings_by_single_user integer,"
                "highest_num_ratings_on_single_item integer,"
                "lowest_num_ratings_on_single_item integer,"
                "mean_num_ratings_by_user real,"
                "mean_num_ratings_on_item real,"
                "rating_skew real,"
                "rating_kurtosis real,"
                "rating_standard_deviation real,"
                "rating_variance real,"
                "FOREIGN KEY(MetadataID)"
                "REFERENCES Runs(MetadataID))")

    cur.execute("CREATE TABLE IF NOT EXISTS Algorithm("
                "AlgorithmID integer PRIMARY KEY,"
                "AlgorithmName string UNIQUE,"
                "FOREIGN KEY(AlgorithmID)"
                "REFERENCES Runs(AlgorithmID))")

    cur.execute("CREATE TABLE IF NOT EXISTS Task("
                "TaskID integer PRIMARY KEY,"
                "TaskName string UNIQUE,"
                "FOREIGN KEY(TaskID)"
                "REFERENCES Runs(TaskID))")

    cur.execute("CREATE TABLE IF NOT EXISTS Metric("
                "MetricID integer PRIMARY KEY,"
                "MetricName string UNIQUE,"
                "FOREIGN KEY(MetricID)"
                "REFERENCES Runs(MetricID))")

    con.close()
    return


# adds data to a given table
def add_data(data: pd.DataFrame, table_name):
    if table_name == "Runs":
        return "Do not use this function to add run data."

    con = sl.connect('camels.db')
    remote_data = pd.read_sql_query(f"SELECT * FROM {table_name}", con)
    if len(remote_data) > 0:
        return f"The table {table_name} already contains data.", False

    data.to_sql(table_name, con, if_exists="append", index=False)
    con.close()

    return f"Data was written to table {table_name}.", True


# generates database entries based on identifier file
def generate_basic_data():
    config = pd.DataFrame([[time.time_ns()]], columns=["VersionID"])

    msg_cfg, wrt_cfg = add_data(config, "Config")
    print(msg_cfg)

    algorithms = pd.DataFrame([[algorithm.value, algorithm.name] for algorithm in Algorithm],
                              columns=["AlgorithmID", "AlgorithmName"])

    msg_alg, wrt_alg = add_data(algorithms, "Algorithm")
    print(msg_alg)

    tasks = pd.DataFrame([[task.value, task.name] for task in Task],
                         columns=["TaskID", "TaskName"])

    msg_tsk, wrt_tsk = add_data(tasks, "Task")
    print(msg_tsk)

    metrics = pd.DataFrame([[metric.value, metric.name] for metric in Metric],
                           columns=["MetricID", "MetricName"])

    msg_met, wrt_met = add_data(metrics, "Metric")
    print(msg_met)

    if any([wrt_cfg, wrt_alg, wrt_tsk, wrt_met]):
        return "Populated database with basic data."
    else:
        return "Database already populated with basic data."


# shorthand to generate and fill tables
def populate_database():
    generate_tables()
    return generate_basic_data()
