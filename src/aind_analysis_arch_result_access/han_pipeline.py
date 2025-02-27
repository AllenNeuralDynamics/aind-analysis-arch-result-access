"""
Get results from Han's pipeline
https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-trigger-pipeline
"""

import logging

import numpy as np
import pandas as pd

from aind_analysis_arch_result_access.util.reformat import (
    data_source_mapper,
    trainer_mapper,
)
from aind_analysis_arch_result_access.util.s3 import get_s3_json, get_s3_pkl

from aind_analysis_arch_result_access import S3_PATH_BONSAI_ROOT, S3_PATH_BPOD_ROOT, DFT_ANALYSIS_DB

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_session_table(if_load_bpod=False):
    """
    Load the session table from Han's pipeline and re-build the master table (almost) the same one
    as in the Streamlit app https://foraging-behavior-browser.allenneuraldynamics-test.org/

    params:
        if_load_bpod: bool, default False
            Whether to load old bpod data. If True, it will take a while.
    """
    # --- Load dfs from s3 ---
    logger.info(f"Loading session table from {S3_PATH_BONSAI_ROOT} ...")
    df = get_s3_pkl(f"{S3_PATH_BONSAI_ROOT}/df_sessions.pkl")
    df.rename(columns={"user_name": "trainer", "h2o": "subject_alias"}, inplace=True)

    logger.info(f"Loading mouse PI mapping from {S3_PATH_BONSAI_ROOT} ...")
    df_mouse_pi_mapping = pd.DataFrame(get_s3_json(f"{S3_PATH_BONSAI_ROOT}/mouse_pi_mapping.json"))

    if if_load_bpod:
        logger.info(f"Loading old bpod data from {S3_PATH_BPOD_ROOT} ...")
        df_bpod = get_s3_pkl(f"{S3_PATH_BPOD_ROOT}/df_sessions.pkl")
        df_bpod.rename(columns={"user_name": "trainer", "h2o": "subject_alias"}, inplace=True)
        df = pd.concat([df, df_bpod], axis=0)

    logger.info("Post-hoc processing...")
    # --- Cleaning up ---
    # Remove hierarchical columns
    df.columns = df.columns.get_level_values(1)
    df.sort_values(["session_start_time"], ascending=False, inplace=True)
    df["session_start_time"] = df["session_start_time"].astype(str)  # Turn to string
    df = df.reset_index()

    # Remove invalid session number
    # Remove rows with no session number (effectively only leave the nwb file
    # with the largest finished_trials for now)
    df.dropna(subset=["session"], inplace=True)
    df.drop(df.query("session < 1").index, inplace=True)

    # Remove invalid subject_id
    df = df[(999999 > df["subject_id"].astype(int)) & (df["subject_id"].astype(int) > 300000)]

    # Remove zero finished trials
    df = df[df["finished_trials"] > 0]

    # --- Reformatting ---
    # Handle mouse and user name
    if "bpod_backup_h2o" in df.columns:
        df["subject_alias"] = np.where(
            df["bpod_backup_h2o"].notnull(),
            df["bpod_backup_h2o"],
            df["subject_id"],
        )
        df["trainer"] = np.where(
            df["bpod_backup_user_name"].notnull(),
            df["bpod_backup_user_name"],
            df["trainer"],
        )
    else:
        df["subject_alias"] = df["subject_id"]

    # drop 'bpod_backup_' columns
    df.drop(
        [col for col in df.columns if "bpod_backup_" in col],
        axis=1,
        inplace=True,
    )

    # --- Normalize trainer name ---
    df["trainer"] = df["trainer"].apply(trainer_mapper)

    # Merge in PI name
    df = df.merge(df_mouse_pi_mapping, how="left", on="subject_id")  # Merge in PI name
    df.loc[df["PI"].isnull(), "PI"] = df.loc[
        df["PI"].isnull()
        & (df["trainer"].isin(df["PI"]) | df["trainer"].isin(["Han Hou", "Marton Rozsa"])),
        "trainer",
    ]  # Fill in PI with trainer if PI is missing and the trainer was ever a PI

    # Mapping data source (Room + Hardware etc)
    df[["institute", "rig_type", "room", "hardware", "data_source"]] = df["rig"].apply(
        lambda x: pd.Series(data_source_mapper(x))
    )

    # --- Removing abnormal values ---
    df.loc[
        df["weight_after"] > 100,
        [
            "weight_after",
            "weight_after_ratio",
            "water_in_session_total",
            "water_after_session",
            "water_day_total",
        ],
    ] = np.nan
    df.loc[
        df["water_in_session_manual"] > 100,
        [
            "water_in_session_manual",
            "water_in_session_total",
            "water_after_session",
        ],
    ] = np.nan
    df.loc[
        (df["duration_iti_median"] < 0) | (df["duration_iti_mean"] < 0),
        [
            "duration_iti_median",
            "duration_iti_mean",
            "duration_iti_std",
            "duration_iti_min",
            "duration_iti_max",
        ],
    ] = np.nan
    df.loc[df["invalid_lick_ratio"] < 0, ["invalid_lick_ratio"]] = np.nan

    # --- Adding something else ---
    # add abs(bais) to all terms that have 'bias' in name
    for col in df.columns:
        if "bias" in col:
            df[f"abs({col})"] = np.abs(df[col])

    # weekday
    df.session_date = pd.to_datetime(df.session_date)
    df["weekday"] = df.session_date.dt.dayofweek + 1

    # trial stats
    df["avg_trial_length_in_seconds"] = (
        df["session_run_time_in_min"] / df["total_trials_with_autowater"] * 60
    )

    # last day's total water
    df["water_day_total_last_session"] = df.groupby("subject_id")["water_day_total"].shift(1)
    df["water_after_session_last_session"] = df.groupby("subject_id")["water_after_session"].shift(
        1
    )

    # fill nan for autotrain fields
    filled_values = {
        "curriculum_name": "None",
        "curriculum_version": "None",
        "curriculum_schema_version": "None",
        "current_stage_actual": "None",
        "has_video": False,
        "has_ephys": False,
    }
    df.fillna(filled_values, inplace=True)

    # foraging performance = foraing_eff * finished_rate
    if "foraging_performance" not in df.columns:
        df["foraging_performance"] = df["foraging_eff"] * df["finished_rate"]
        df["foraging_performance_random_seed"] = (
            df["foraging_eff_random_seed"] * df["finished_rate"]
        )

    # Recorder columns so that autotrain info is easier to see
    first_several_cols = [
        "subject_id",
        "session_date",
        "nwb_suffix",
        "session",
        "rig",
        "trainer",
        "PI",
        "curriculum_name",
        "curriculum_version",
        "current_stage_actual",
        "task",
        "notes",
    ]
    new_order = first_several_cols + [col for col in df.columns if col not in first_several_cols]
    df = df[new_order]

    return df

# %%
def get_mle_available_models(subject_id: str, session_date: str):
    """Get the available models for MLE fitting given the subject_id and session_date

    Parameters
    ----------
    subject_id : str,
    session_date : str,

    Returns
    -------
    DataFrame
        A DataFrame with columns: nwb_name, model_alias
    """
    records = DFT_ANALYSIS_DB.retrieve_docdb_records(
        filter_query={
            "subject_id": subject_id,
            "session_date": session_date,
        },
        projection={"_id": 0, "nwb_name": 1,
                    "analysis_results.fit_settings.agent_alias": 1},
    )
    df = pd.json_normalize(records).rename(
        columns={"analysis_results.fit_settings.agent_alias": "model_alias"}
    )

    if df.model_alias.duplicated().any():
        print(
            "WARNING: Duplicated model_alias!\n"
            "         It is possible that there are multiple nwbs for this session.\n"
            "         You should check the time stamp in the nwb names."
        )

    return 

# %%

def get_mle_model_fitting(
    subject_id: str = None,
    session_date: str = None,
    query: dict = None,
    if_get_figures: bool = False,
):
    #%%
    subject_id = "730945"
    session_date = "2024-10-24"
    
    records = DFT_ANALYSIS_DB.retrieve_docdb_records(
        filter_query={
            "subject_id": subject_id,
            "session_date": session_date,
        }
    )
    
    
    #%%
    pass


if __name__ == "__main__":
    df = get_session_table()
    print(df)
    print(df.columns)
