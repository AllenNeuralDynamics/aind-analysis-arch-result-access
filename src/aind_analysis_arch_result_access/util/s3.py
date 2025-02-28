"""
Util functions for public S3 bucket access
"""

import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import s3fs
from tqdm import tqdm

from aind_analysis_arch_result_access import S3_PATH_ANALYSIS_ROOT

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


def get_s3_latent_variable_batch(ids, max_threads_for_s3=10):
    """Get latent variables from s3 for a batch of ids"""
    with ThreadPoolExecutor(max_workers=max_threads_for_s3) as executor:
        results = list(
            tqdm(
                executor.map(get_s3_latent_variable, ids),
                total=len(ids),
                desc="Get latent variables from s3",
            )
        )
    return [{"_id": _id, "latent_variables": latent} for _id, latent in zip(ids, results)]


def get_s3_latent_variable(id):
    """Get latent variables from s3 for a single id"""
    # -- Rebuild s3 path from id (the job_hash) --
    path = f"{S3_PATH_ANALYSIS_ROOT}/{id}/"

    # -- Try different result json names for back compatibility --
    possible_json_names = ["docDB_mle_fitting.json", "docDB_record.json"]
    for json_name in possible_json_names:
        if fs.exists(f"{path}{json_name}"):
            break
    else:
        print(f"Cannot find latent variables for id {id}")
        return None

    # -- Load the json --
    # Get the full result json from s3
    result_json = get_s3_json(f"{path}{json_name}")

    # Get the latent variables
    latent_variable = result_json["analysis_results"]["fitted_latent_variables"]

    if "q_value" not in latent_variable:
        return latent_variable

    # -- Add RPE to the latent variables, if q_value exists --
    # Notes: RPE = reward - q_value_chosen
    # In the model fitting output, len(choice) = len(reward) = n_trials,
    # but len(q_value) = n_trials + 1, because it includes a final update after the last choice.
    # When computing RPE, we need to use the q_value before the choice on the chosen side.
    choice = np.array(result_json["analysis_results"]["fit_settings"]["fit_choice_history"]).astype(
        int
    )
    reward = np.array(result_json["analysis_results"]["fit_settings"]["fit_reward_history"]).astype(
        int
    )
    q_value_before_choice = np.array(latent_variable["q_value"])[:, :-1]  # Note the :-1 here
    q_value_chosen = q_value_before_choice[choice, np.arange(len(choice))]
    latent_variable["rpe"] = reward - q_value_chosen

    return latent_variable


def get_s3_mle_figure_batch(
    ids, f_names, download_path="./results/mle_figures/", max_threads_for_s3=10
):
    """Download MLE figures from s3 for a batch of ids"""
    os.makedirs(download_path, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max_threads_for_s3) as executor:
        list(
            tqdm(
                executor.map(get_s3_mle_figure, ids, f_names, [download_path] * len(ids)),
                total=len(ids),
                desc="Download figures from s3",
            )
        )


def get_s3_mle_figure(id, f_name, download_path):
    """Download MLE figures from s3 for a single id"""
    file_name_on_s3 = "fitted_session.png"

    if fs.exists(f"{S3_PATH_ANALYSIS_ROOT}/{id}/{file_name_on_s3}"):
        fs.download(f"{S3_PATH_ANALYSIS_ROOT}/{id}/{file_name_on_s3}", f"{download_path}/{f_name}")
