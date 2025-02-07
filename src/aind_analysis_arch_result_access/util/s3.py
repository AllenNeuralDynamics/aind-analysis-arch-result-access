"""
Util functions for public S3 bucket access
"""

import json
import pickle

import s3fs

# The processed bucket is public
fs = s3fs.S3FileSystem(anon=True)


def get_s3_pkl(s3_path):
    """
    Load a pickled dataframe from an s3 path
    """
    with fs.open(s3_path, "rb") as f:
        df_loaded = pickle.load(f)
    return df_loaded


def get_s3_json(s3_path):
    """
    Load a json file from an s3 path
    """
    with fs.open(s3_path) as f:
        json_loaded = json.load(f)
    return json_loaded
