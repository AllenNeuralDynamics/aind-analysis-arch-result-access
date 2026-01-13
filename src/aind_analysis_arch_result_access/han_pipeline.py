"""
Get results from Han's pipeline
https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-trigger-pipeline
"""

import logging
from datetime import datetime
from typing import Literal

import aind_data_access_api.document_db
import numpy as np
import pandas as pd

from aind_analysis_arch_result_access.util.reformat import (
    curriculum_ver_mapper,
    data_source_mapper,
    trainer_mapper,
)
from aind_analysis_arch_result_access.util.s3 import (
    S3_PATH_BONSAI_ROOT,
    S3_PATH_BPOD_ROOT,
    get_s3_json,
    get_s3_logistic_regression_betas_batch,
    get_s3_logistic_regression_figure_batch,
    get_s3_pkl,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_session_table(if_load_bpod=False, only_recent_n_month=None) -> pd.DataFrame:
    """
    Load the session table from Han's pipeline and re-build the master table (almost) the same one
    as in the Streamlit app https://foraging-behavior-browser.allenneuraldynamics-test.org/

    params:
        if_load_bpod: bool, default False
            Whether to load old bpod data. If True, it will take a while.
        only_recent_n_month: int, optional, default None
            If specified, only sessions from the past N months will be included.
            If None, all sessions will be included.
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

    # Filter sessions by date if requested (early filtering for performance)
    df["session_date"] = pd.to_datetime(df["session_date"])
    if only_recent_n_month is not None:
        # Filter to only recent N months
        cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=only_recent_n_month)
        df = df[df["session_date"] >= cutoff_date]
        logger.info(
            f"Filtered to sessions from {cutoff_date.date()} onwards "
            f"(recent {only_recent_n_month} months). Remaining sessions: {len(df)}"
        )

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

    # Merge in curriculum from Han's autotrain database
    df_autotrain = get_autotrain_table()

    # merge in curriculum from docDB
    df_docDB = get_docDB_table()

    pd_merged = pd.concat([df_autotrain, df_docDB], axis=0, ignore_index=True)

    # Drop curriculum columns retrieved from session json by Han's temporary pipeline
    columns_to_drop = [
        "curriculum_name",
        "curriculum_version",
        "curriculum_schema_version",
        "current_stage_actual",
        "if_overriden_by_trainer",
    ]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Merge curriculum info autotrain database
    df = df.merge(
        pd_merged.query("if_closed_loop == True")[
            [
                "subject_id",
                "session_date",
                "curriculum_name",
                "curriculum_version",
                "curriculum_schema_version",
                "current_stage_suggested",
                "current_stage_actual",
                "session_at_current_stage",
                "decision",
                "next_stage_suggested",
                "if_overriden_by_trainer",
            ]
        ].drop_duplicates(subset=["subject_id", "session_date"], keep="first"),
        on=["subject_id", "session_date"],
        how="left",
    )

    # curriculum version group
    df["curriculum_version_group"] = df["curriculum_version"].map(curriculum_ver_mapper)

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


def get_autotrain_table():
    """
    Load the curriculum data from Han's autotrain database directly from a (duplicated) s3 bucket.
      s3://aind-behavior-data/foraging_nwb_bonsai_processed/foraging_auto_training/df_manager_447_demo.pkl


    """
    s3_path = (
        "s3://aind-behavior-data/foraging_nwb_bonsai_processed/"
        "foraging_auto_training/df_manager_447_demo.pkl"
    )
    df_autotrain = get_s3_pkl(s3_path)
    df_autotrain["session_date"] = pd.to_datetime(df_autotrain["session_date"])

    logger.info("Loaded curriculum data from Han's autotrain.")
    return df_autotrain


def get_docDB_table() -> pd.DataFrame:
    """
    Load the curriculum data from the behavior json in docDB
    """

    logger.info("Loading curriculum data from docDB...")

    docdb_client = aind_data_access_api.document_db.MetadataDbClient(
        host="api.allenneuraldynamics.org", database="metadata_index", collection="data_assets"
    )
    sessions = docdb_client.retrieve_docdb_records(
        filter_query={
            "session.stimulus_epochs.software.name": "dynamic-foraging-task",
            "data_description.data_level": "raw",
            "session.stimulus_epochs": {
                "$elemMatch": {"output_parameters.streamlit": {"$exists": True}}
            },
        },
        projection={
            "session.subject_id": 1,
            "session.session_start_time": 1,
            "session.stimulus_epochs.output_parameters.streamlit": 1,
        },
    )

    df_dict = {
        "subject_id": [],
        "session_date": [],
        "curriculum_name": [],
        "curriculum_version": [],
        "current_stage_actual": [],
        "current_stage_suggested": [],
        "if_overriden_by_trainer": [],
        "next_stage_suggested": [],
        "if_closed_loop": [],  # IMPORTANT: automatically set to True to merge with autotrain
    }

    for session in sessions:
        try:
            curriculum_params = session["session"]["stimulus_epochs"][0]["output_parameters"][
                "streamlit"
            ]
            df_dict["subject_id"].append(session["session"]["subject_id"])
            df_dict["session_date"].append(
                datetime.strptime(session["session"]["session_start_time"][:10], "%Y-%m-%d")
            )
            df_dict["curriculum_name"].append(curriculum_params["curriculum_name"])
            df_dict["curriculum_version"].append(curriculum_params["curriculum_version"])
            df_dict["current_stage_actual"].append(curriculum_params["current_stage_actual"])
            df_dict["current_stage_suggested"].append(curriculum_params["current_stage_suggested"])
            df_dict["if_overriden_by_trainer"].append(curriculum_params["if_overriden_by_trainer"])
            df_dict["next_stage_suggested"].append(curriculum_params["next_stage_suggested"])
            # IMPORTANT: automatically set to True so merge with autotrain table works
            df_dict["if_closed_loop"].append(True)

        except (TypeError, KeyError, IndexError):
            pass

    logger.info(f"Loaded {len(df_dict)} curriculum data from docDB.")
    return pd.DataFrame(df_dict)


def get_mle_model_fitting(
    subject_id: str = None,
    session_date: str = None,
    agent_alias: str = None,
    from_custom_query: dict = None,
    if_include_metrics: bool = True,
    if_include_latent_variables: bool = True,
    if_download_figures: bool = False,
    download_path: str = "./results/mle_figures/",
    paginate_settings: dict = {"paginate": False},
    max_threads_for_s3: int = 10,
) -> pd.DataFrame:
    """DEPRECATED shim.

    The full implementation of `get_mle_model_fitting` has been migrated to
    `aind_analysis_arch_result_access.df_mle_model_fitting.get_mle_model_fitting`.

    This shim preserves the original function name and signature so existing
    code continues to work. It logs a warning and forwards the call to the
    migrated implementation.
    """
    logger.warning(
        "get_mle_model_fitting has been migrated to "
        "aind_analysis_arch_result_access.df_mle_model_fitting.get_mle_model_fitting. "
        "This shim will forward the call to the new implementation."
    )
    logger.warning(
        "Please update your imports to `from aind_analysis_arch_result_access "
        "import get_mle_model_fitting`"
    )

    # Import locally to avoid import-time cycles
    from aind_analysis_arch_result_access.df_mle_model_fitting import (
        get_mle_model_fitting as _migrated_get_mle_model_fitting,
    )

    return _migrated_get_mle_model_fitting(
        subject_id=subject_id,
        session_date=session_date,
        agent_alias=agent_alias,
        from_custom_query=from_custom_query,
        if_include_metrics=if_include_metrics,
        if_include_latent_variables=if_include_latent_variables,
        if_download_figures=if_download_figures,
        download_path=download_path,
        paginate_settings=paginate_settings,
        max_threads_for_s3=max_threads_for_s3,
    )


def get_logistic_regression(
    df_sessions: pd.DataFrame,
    model: Literal["Su2022", "Bari2019", "Miller2021", "Hattori2019"],
    if_download_figures: bool = False,
    download_path: str = "./results/logistic_regression/",
    max_threads_for_s3: int = 10,
) -> pd.DataFrame:
    """Get logistic regression betas from Han's analysis pipeline (the old one with pure s3)
    https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-trigger-pipeline

    Parameters
    ----------
    df_sessions : pd.DataFrame
        A DataFrame containing at least subject_id and session_date columns
    model : Literal["Su2022", "Bari2019", "Miller2021", "Hattori2019"]
        The model to use for logistic regression. See notes here:
        https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models?tab=readme-ov-file#logistic-regression  # noqa: E501
    if_download_figures : bool, optional
        Whether to download the figures from s3, by default False
    download_path : str, optional
        The path to download the figures, by default "./results/logistic_regression/"
    max_threads_for_s3 : int, optional
        The maximum number of parallel threads for getting result from s3, by default 10
    """

    # -- Input validation --
    if set(["subject_id", "session_date"]).issubset(df_sessions.columns) is False:
        raise ValueError("df_sessions must contain subject_id and session_date columns.")
    if model not in ["Su2022", "Bari2019", "Miller2021", "Hattori2019"]:
        raise ValueError(
            "Model must be one of ['Su2022', 'Bari2019', 'Miller2021', 'Hattori2019']."
        )

    # -- Use get nwb_suffix from df_master (the master session table shown on Streamlit) --
    df_master = get_session_table(if_load_bpod=False)
    df_master["session_date"] = df_master["session_date"].astype(str)

    df_to_query = df_sessions[["subject_id", "session_date"]].copy()
    df_to_query["session_date"] = df_to_query["session_date"].astype(str)
    df_to_query = df_to_query.merge(
        df_master[["subject_id", "session_date", "nwb_suffix"]],
        on=["subject_id", "session_date"],
        how="left",
    )

    # -- Get betas --
    sessions_in_han_pipeline = df_to_query["nwb_suffix"].notnull()
    download_setting = dict(
        subject_ids=df_to_query.loc[sessions_in_han_pipeline, "subject_id"],
        session_dates=df_to_query.loc[sessions_in_han_pipeline, "session_date"],
        nwb_suffixs=df_to_query.loc[sessions_in_han_pipeline, "nwb_suffix"].astype(int),
        model=model,
        max_threads_for_s3=max_threads_for_s3,
    )
    df_logistic_regression = get_s3_logistic_regression_betas_batch(
        **download_setting,
    )

    logger.info(
        f"Successfully retrieved logistic regression betas from"
        f" {len(df_logistic_regression)} / {len(df_to_query)} sessions."
    )

    if len(df_logistic_regression) < len(df_to_query):
        logger.warning("Sessions that are missing in han's pipeline: ")
        logger.warning(df_to_query.loc[~sessions_in_han_pipeline].to_string(index=False))

    if len(df_logistic_regression) == 0:
        return pd.DataFrame()

    # -- Merge in fitting metrics (from df_session itself) --
    metrics_columns = [col for col in df_master if model in col and "abs" not in col]
    df_fitting_metrics = df_master[["subject_id", "session_date"] + metrics_columns].set_index(
        ["subject_id", "session_date"]
    )
    df_fitting_metrics.columns = pd.MultiIndex.from_product(
        [df_fitting_metrics.columns, [None]],
        names=["analysis_spec", "analysis_results"],
    )

    df_logistic_regression = df_logistic_regression.merge(
        df_fitting_metrics,
        on=["subject_id", "session_date"],
        how="left",
    )

    # -- Download figures --
    if if_download_figures:
        get_s3_logistic_regression_figure_batch(
            **download_setting,
            download_path=download_path,
        )

    return df_logistic_regression


if __name__ == "__main__":
    df = get_session_table(if_load_bpod=True)
    print(df)
    print(df.columns)
