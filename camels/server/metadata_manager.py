import sqlite3 as sl
import pandas as pd
import json


# checks if given hash exists in database
def check_hash(l_hash):
    con = sl.connect('camels.db')
    remote_metadata = pd.read_sql_query(f"SELECT * FROM Metadata", con)

    # check if the metadata already exists and get a new id
    if len(remote_metadata) > 0:
        exist = remote_metadata[remote_metadata["Hash"] == int(l_hash)]
        if len(exist) > 0:
            md_id = exist["MetadataID"].values[0]
            return "Metadata already exists on server.", True, int(md_id)

    return "Metadata does not exist on the server.", False, -1


# writes the metadata object to the database
def write_metadata(meta_data):
    meta_data = json.loads(meta_data)

    con = sl.connect('camels.db')
    remote_metadata = pd.read_sql_query(f"SELECT * FROM Metadata", con)

    # set the metadata id
    meta_data["MetadataID"] = [0]
    if len(remote_metadata) > 0:
        meta_data["MetadataID"] = [remote_metadata["MetadataID"].values.max() + 1]

    # write metadata
    pd.DataFrame(meta_data).to_sql("Metadata", con, if_exists="append", index=False)
    con.close()

    return f"Saved metadata for data set {meta_data['MetadataName'][0]} with ID {meta_data['MetadataID'][0]}.", \
           int(meta_data['MetadataID'][0])
