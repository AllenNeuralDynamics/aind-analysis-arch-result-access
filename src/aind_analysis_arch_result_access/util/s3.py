"""
Util functions for public S3 bucket access
"""

import json
import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import s3fs
from tqdm import tqdm

from aind_analysis_arch_result_access import (
    S3_PATH_ANALYSIS_ROOT,
    S3_PATH_BONSAI_ROOT,
)

# The processed bucket is public
fs = s3fs.S3FileSystem(anon=True)

logger = logging.getLogger(__name__)


def get_s3_pkl(s3_path):
    """
    Load a pickled dataframe from an s3 path
    """
    if not fs.exists(s3_path):
        logger.warning(f"Cannot find file at {s3_path}")
        return pd.DataFrame()  # Return an empty DataFrame if the file does not exist
    with fs.open(s3_path, "rb") as f:
        df_loaded = pickle.load(f)
    return df_loaded


def get_s3_json(s3_path):
    """
    Load a json file from an s3 path
    """
    if not fs.exists(s3_path):
        logger.warning(f"Cannot find file at {s3_path}")
        return None
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
        fs.download(
            f"{S3_PATH_ANALYSIS_ROOT}/{id}/{file_name_on_s3}",
            f"{download_path}/{f_name}",
        )


def _build_nwb_name(subject_id, session_date, nwb_suffix):
    """Recover string like 676746_2023-10-06 or 684039_2023-10-25_114737"""
    return subject_id + "_" + session_date + (f"_{nwb_suffix}" if nwb_suffix > 0 else "")


def get_s3_logistic_regression_betas(subject_id, session_date, nwb_suffix, model):
    """Download df_logistic_betas from s3 for a single session"""
    df_logistic = get_s3_pkl(
        f"{S3_PATH_BONSAI_ROOT}/{_build_nwb_name(subject_id, session_date, nwb_suffix)}/"
        f"{_build_nwb_name(subject_id, session_date, nwb_suffix)}"
        f"_df_session_logistic_regression_df_beta_{model}.pkl"
    )
    return df_logistic


def get_s3_logistic_regression_betas_batch(
    subject_ids, session_dates, nwb_suffixs, model, max_threads_for_s3=10
):
    """Get df_logistic_betas from s3 for a batch of sessions"""
    with ThreadPoolExecutor(max_workers=max_threads_for_s3) as executor:
        results = list(
            tqdm(
                executor.map(
                    get_s3_logistic_regression_betas,
                    subject_ids,
                    session_dates,
                    nwb_suffixs,
                    [model] * len(subject_ids),  # Repeat the model for each subject
                ),
                total=len(subject_ids),
                desc="Get logistic regression betas from s3",
            )
        )

    if not results:
        logger.warning("No results found for the provided subject_ids and session_dates.")
        return pd.DataFrame()
    return pd.concat(results).reset_index()


def get_s3_logistic_regression_figure(subject_id, session_date, nwb_suffix, model, download_path):
    """Download logistic regression figure from s3 for a single session"""
    f_name = (
        f"{_build_nwb_name(subject_id, session_date, nwb_suffix)}_logistic_regression_{model}.png"
    )
    fig_full_path = (
        f"{S3_PATH_BONSAI_ROOT}/{_build_nwb_name(subject_id, session_date, nwb_suffix)}/{f_name}"
    )

    if fs.exists(fig_full_path):
        fs.download(fig_full_path, f"{download_path}/{f_name}")
        return True
    else:
        logger.warning(f"Cannot find logistic regression figure at {fig_full_path}")
        return False


def get_s3_logistic_regression_figure_batch(
    subject_ids,
    session_dates,
    nwb_suffixs,
    model,
    download_path="./results/logistic_regression_figures/",
    max_threads_for_s3=10,
):
    """Download logistic regression figures from s3 for a batch of sessions"""
    os.makedirs(download_path, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max_threads_for_s3) as executor:
        list(
            tqdm(
                executor.map(
                    get_s3_logistic_regression_figure,
                    subject_ids,
                    session_dates,
                    nwb_suffixs,
                    [model] * len(subject_ids),
                    [download_path] * len(subject_ids),
                ),
                total=len(subject_ids),
                desc=f"Download logistic regression figures from s3 to {download_path}",
            )
        )
