"""
Get results from Han's pipeline
https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-trigger-pipeline
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import numpy as np
import pandas as pd

from aind_analysis_arch_result_access.util.reformat import (
    data_source_mapper,
    trainer_mapper,
)
from aind_analysis_arch_result_access.util.s3 import fs, get_s3_json, get_s3_pkl

from aind_analysis_arch_result_access import S3_PATH_BONSAI_ROOT, S3_PATH_BPOD_ROOT, S3_PATH_ANALYSIS_ROOT, DFT_ANALYSIS_DB

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
def get_mle_model_fitting(
    subject_id: str = None,
    session_date: str = None,
    agent_alias: str = None,
    from_custom_query: dict = None,
    if_include_metrics: bool = True,
    if_include_latent_variables: bool = True,
    paginate_settings: dict = {"paginate": False},
    max_threads_for_s3: int = 10,
):
    """Get the available models for MLE fitting given the subject_id and session_date

    Parameters
    ----------
    subject_id : str, optional
        The subject_id, by default None
    session_date : str, optional
        The session_date, by default None
    agent_alias : str, optional
        The agent_alias, by default None
    from_custom_query : dict, optional
        The custom query, by default None
        If provided, subject_id, session_date, and agent_alias will be ignored.
        Error will be raised if none of the four is provided.
    if_include_metrics : bool, optional
        Whether to include the metrics in the DataFrame, by default True
        If False, only the agent_alias will be included.
    if_include_latent_variables : bool, optional
        Whether to include the latent variables in the DataFrame, by default True
    paginate_settings : dict, optional
        The settings for pagination, by default {"paginate": False}.
        If you see a 503 error, you may need to set paginate to True.
        See aind_data_access_api documentation.
    max_threads_for_s3: int, optional
        The maximum number of parallel threads for getting result from s3, by default 10

    Returns
    -------
    DataFrame
        A DataFrame containing the available models for MLE fitting
    """

    ANALYSIS_NAME = "MLE fitting"
    ANALYSIS_VER = "first version @ 0.10.0"

    # -- Build query --
    filter_query = {
        "analysis_spec.analysis_name": ANALYSIS_NAME,
        "analysis_spec.analysis_ver": ANALYSIS_VER,
    }
    if from_custom_query:
        # If from_custom_query is provided, ignore subject_id, session_date, and agent_alias
        filter_query = filter_query.update(from_custom_query)
    else:
        if subject_id:
            filter_query["subject_id"] = subject_id
        if session_date:
            filter_query["session_date"] = session_date
        if agent_alias:
            filter_query["analysis_results.fit_settings.agent_alias"] = agent_alias
        if not any([subject_id, session_date, agent_alias]):
            raise ValueError(
                "You must provide at least one of subject_id, session_date, "
                "agent_alias, or from_custom_query!"
            )

    projection = {
        "_id": 1,
        "nwb_name": 1,
        "analysis_results.fit_settings.agent_alias": 1,
        "status": 1,
    }
    if if_include_metrics:
        projection.update(
            {
                "analysis_results.log_likelihood": 1,
                "analysis_results.prediction_accuracy": 1,
                "analysis_results.k_model": 1,
                "analysis_results.n_trials": 1,
                "analysis_results.AIC": 1,
                "analysis_results.BIC": 1,
                "analysis_results.LPT": 1,
                "analysis_results.LPT_AIC": 1,
                "analysis_results.LPT_BIC": 1,
                "analysis_results.cross_validation": 1,
                "analysis_results.params": 1,
            }
        )

    # -- Retrieve records --
    logger.info(f"Query: {filter_query}")
    records = DFT_ANALYSIS_DB.retrieve_docdb_records(
        filter_query=filter_query,
        projection=projection,
        **paginate_settings,
    )

    if not records:
        logger.warning(f"No MLE fitting available for {subject_id} on {session_date}")
        return None
    logger.info(f"Found {len(records)} MLE fitting records!")

    # -- Reformat the records --
    # Turn the nested json into a flat DataFrame and rename the columns, except params
    if if_include_metrics:
        params = [record["analysis_results"].pop("params") for record in records]
    df = pd.json_normalize(records)
    df = df.rename(
        columns={
            col: col.replace("analysis_results.", "")
            .replace("cross_validation.", "")
            .replace("fit_settings.", "")
            for col in df.columns
        }
    )

    if if_include_metrics:
        # Put in params as dict
        df["params"] = params
        
        # Compute cross_validation mean and std
        for group in ["test", "fit", "test_bias_only"]:
            df[f"prediction_accuracy_10-CV_{group}"] = df[f"prediction_accuracy_{group}"].apply(
                lambda x: np.mean(x)
            )
            df[f"prediction_accuracy_10-CV_{group}_std"] = df[f"prediction_accuracy_{group}"].apply(
                lambda x: np.std(x)
            )

    if subject_id and session_date and df.agent_alias.duplicated().any():
        # If the user specifies one certain session, and there are
        logger.warning(
            "Duplicated agent_alias!\n"
            "There are multiple nwbs for this session:\n"
            f"{df.nwb_name.unique()}\n"
            "You should check the time stamps to select the one you want."
        )
        
    if if_include_latent_variables and len(df.query("status == 'success'")):
        latents = get_latent_variable_batch(df.query("status == 'success'")._id, 
                                           max_threads_for_s3=max_threads_for_s3)
        df = df.merge(pd.DataFrame(latents), on="_id", how="left")

    return df

def get_latent_variable_batch(_ids, max_threads_for_s3=10):
    with ThreadPoolExecutor(max_workers=max_threads_for_s3) as executor:
        results = list(
            tqdm(
                executor.map(_get_latent_variable, _ids),
                total=len(_ids),
                desc="Get latent variables from s3",
            )
        )
    return [{"_id": _id, "latent_variables": latent} for _id, latent in zip(_ids, results)]

def get_latent_variable(id):
    # -- Rebuild s3 path from id (the job_hash) --
    path = f"{S3_PATH_ANALYSIS_ROOT}/{id}/"
    
    # -- Try different result json names for back compatibility --
    possible_json_names = ["docDB_mle_fitting.json", "docDB_record.json"]
    for json_name in possible_json_names:
        if fs.exists(f"{path}{json_name}"):
            break
    else:
        logger.warning(f"Cannot find latent variables for id {id}")
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
    choice = np.array(
        result_json["analysis_results"]["fit_settings"]["fit_choice_history"]
    ).astype(int)
    reward = np.array(
        result_json["analysis_results"]["fit_settings"]["fit_reward_history"]
    ).astype(int)
    q_value_before_choice = np.array(latent_variable["q_value"])[:, :-1]  # Note the :-1 here
    q_value_chosen = q_value_before_choice[choice, np.arange(len(choice))]
    latent_variable["rpe"] = reward - q_value_chosen
    
    return latent_variable


import time
start = time.time()
df = get_mle_model_fitting(subject_id="730945", 
                           #session_date="2024-10-24", 
                           if_include_metrics=False,
                           if_include_latent_variables=True,
                           max_threads_for_s3=10)

print(time.time() - start)
# %%

if __name__ == "__main__":
    df = get_session_table()
    print(df)
    print(df.columns)
