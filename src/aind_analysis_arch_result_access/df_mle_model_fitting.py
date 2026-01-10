"""
Migrated get_mle_model_fitting implementation.
This module contains the MLE model fitting function moved out of han_pipeline.py
so callers can import it directly from the package.
"""

import logging

import numpy as np
import pandas as pd
from aind_data_access_api.document_db import MetadataDbClient
from scipy.stats import entropy

from aind_analysis_arch_result_access.util.s3 import (
    S3_PATH_ANALYSIS_ROOT,
    get_s3_latent_variable_batch,
    get_s3_mle_figure_batch,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# New collection
analysis_docDB_dft = MetadataDbClient(
    host="api.allenneuraldynamics.org",
    database="analysis",
    collection="dynamic-foraging-model-fitting",
)


def _add_qvalue_spread(latents):
    """
    For a list of latents, compute the uniform ratio of q_values for each.
    Returns a list of uniform ratios (np.nan if q_value is missing).
    """
    num_bins = 100
    max_entropy = np.log2(num_bins)
    for latent in latents:
        if latent is None or latent.get("latent_variables") is None:
            latent["qvalue_spread"] = np.nan
            continue
        q_vals = latent["latent_variables"].get("q_value", None)
        if q_vals is None:
            latent["qvalue_spread"] = np.nan
            continue
        hist, _ = np.histogram(q_vals, bins=num_bins, range=(0, 1))
        prob = hist / np.sum(hist) if np.sum(hist) > 0 else np.zeros_like(hist)
        prob = prob[prob > 0]
        if len(prob) == 0:
            latent["qvalue_spread"] = np.nan
            continue
        uniform_ratio = entropy(prob, base=2) / max_entropy
        latent["qvalue_spread"] = uniform_ratio
    return latents


def build_query_new_format(
    from_custom_query=None, subject_id=None, session_date=None, agent_alias=None
):
    """Build query for new AIND Analysis Framework format."""
    filter_query = {
        "processing.data_processes.code.parameters.analysis_name": "MLE fitting",
        "processing.data_processes.code.parameters.analysis_tag": "aind-analysis-framework v0.1",
    }

    # If custom query is provided, use it exclusively
    if from_custom_query:
        filter_query.update(from_custom_query)
        return filter_query

    # Ensure at least one of the parameters is provided
    if not any([subject_id, session_date, agent_alias]):
        raise ValueError(
            "You must provide at least one of subject_id, session_date, "
            "agent_alias, or from_custom_query!"
        )

    # Build a dictionary with only provided keys
    standard_query = {
        "processing.data_processes.output_parameters.subject_id": subject_id,
        "processing.data_processes.output_parameters.session_date": session_date,
        "processing.data_processes.output_parameters.fit_settings.agent_alias": agent_alias,
    }
    # Update filter_query only with non-None values
    filter_query.update({k: v for k, v in standard_query.items() if v is not None})
    return filter_query


def build_query_old_format(
    from_custom_query=None, subject_id=None, session_date=None, agent_alias=None
):
    """Build query for old format (backward compatibility)."""
    filter_query = {
        "analysis_spec.analysis_name": "MLE fitting",
        "analysis_spec.analysis_ver": "first version @ 0.10.0",
    }

    # If custom query is provided, use it exclusively
    if from_custom_query:
        filter_query.update(from_custom_query)
        return filter_query

    # Ensure at least one of the parameters is provided
    if not any([subject_id, session_date, agent_alias]):
        raise ValueError(
            "You must provide at least one of subject_id, session_date, "
            "agent_alias, or from_custom_query!"
        )

    # Build a dictionary with only provided keys
    standard_query = {
        "subject_id": subject_id,
        "session_date": session_date,
        "analysis_results.fit_settings.agent_alias": agent_alias,
    }
    # Update filter_query only with non-None values
    filter_query.update({k: v for k, v in standard_query.items() if v is not None})
    return filter_query


def build_query(from_custom_query=None, subject_id=None, session_date=None, agent_alias=None):
    """Build query for MLE fitting (legacy wrapper for backward compatibility)."""
    return build_query_old_format(from_custom_query, subject_id, session_date, agent_alias)


def _build_projection(if_include_metrics: bool, is_new_format: bool = False) -> dict:
    """Build projection dict for database query with field aliasing.

    Uses MongoDB's field aliasing ("new_name": "$path") to directly project
    fields with their final flattened names, avoiding post-processing.

    Parameters
    ----------
    if_include_metrics : bool
        Whether to include metric fields
    is_new_format : bool
        If True, build for AIND Analysis Framework; else Han's prototype
    """
    if is_new_format:
        p = "processing.data_processes.output_parameters"  # Base path
        fr = f"{p}.fitting_results"  # Fitting results path
        base_projection = {
            "_id": 1,
            "nwb_name": f"${p}.nwb_name",
            "analysis_time": "$processing.data_processes.end_date_time",
            "subject_id": f"${p}.subject_id",
            "session_date": f"${p}.session_date",
            "status": f"${p}.additional_info",
            "agent_alias": f"${fr}.fit_settings.agent_alias",
            "n_trials": f"${fr}.n_trials",
            "S3_location": "$location",
            "CO_asset_id": "$processing.data_processes.code.input_data.url",
        }
    else:
        fr = "analysis_results"  # Fitting results path (reuse variable name)
        base_projection = {
            "_id": 1,
            "nwb_name": 1,
            "analysis_time": "$analysis_datetime",
            "subject_id": 1,
            "session_date": 1,
            "status": 1,
            "agent_alias": f"${fr}.fit_settings.agent_alias",
            "n_trials": f"${fr}.n_trials",
        }

    if if_include_metrics:
        # Metric fields (same structure for both formats, just different base path)
        metric_fields = [
            "log_likelihood",
            "prediction_accuracy",
            "k_model",
            "AIC",
            "BIC",
            "LPT",
            "LPT_AIC",
            "LPT_BIC",
            "params",
        ]
        cv_fields = [
            "prediction_accuracy_test",
            "prediction_accuracy_fit",
            "prediction_accuracy_test_bias_only",
        ]

        base_projection.update({field: f"${fr}.{field}" for field in metric_fields})
        base_projection.update({field: f"${fr}.cross_validation.{field}" for field in cv_fields})

    return base_projection


def _try_retrieve_records(
    query_builder,
    format_name: str,
    if_include_metrics: bool,
    subject_id,
    session_date,
    agent_alias,
    from_custom_query,
    paginate_settings,
):
    """Try to retrieve records from database.

    Parameters
    ----------
    query_builder : callable
        Function to build query (build_query_new_format or build_query_old_format)
    format_name : str
        Name of format ('AIND Analysis Framework' or 'Han's prototype analysis pipeline')
    if_include_metrics : bool
        Whether to include metrics in projection
    subject_id, session_date, agent_alias, from_custom_query
        Query parameters
    paginate_settings : dict
        Pagination settings

    Returns
    -------
    list
        Records, or empty list if none found
    """
    filter_query = query_builder(from_custom_query, subject_id, session_date, agent_alias)
    is_new = format_name == "AIND Analysis Framework"
    projection = _build_projection(if_include_metrics, is_new_format=is_new)

    print(f"Querying {format_name}: {filter_query}")
    records = analysis_docDB_dft.retrieve_docdb_records(
        filter_query=filter_query,
        projection=projection,
        **paginate_settings,
    )

    # Strip out the [] for new records (because they were from processing.data_processed[0])
    if is_new:
        for i, record in enumerate(records):
            records[i] = {
                k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in record.items()
            }

    return records


def get_mle_model_fitting(
    subject_id: str = None,
    session_date: str = None,
    agent_alias: str = None,
    from_custom_query: dict = None,
    only_recent_version: bool = True,
    if_include_metrics: bool = True,
    if_include_latent_variables: bool = True,
    if_download_figures: bool = False,
    download_path: str = "./results/mle_figures/",
    paginate_settings: dict = {"paginate": False},
    max_threads_for_s3: int = 10,
) -> pd.DataFrame:
    """Get MLE model fitting results from the analysis database.

    Retrieves MLE (Maximum Likelihood Estimation) model fitting results for dynamic
    foraging behavioral data. The function queries the analysis database, processes
    the results, and optionally includes latent variables and downloads visualization
    figures from S3.

    This implementation is migrated from `han_pipeline.py` to allow direct imports
    without loading the entire pipeline module.

    Parameters
    ----------
    subject_id : str, optional
        The subject identifier (e.g., animal ID). At least one of subject_id,
        session_date, agent_alias, or from_custom_query must be provided.
    session_date : str, optional
        The session date in string format. At least one of subject_id, session_date,
        agent_alias, or from_custom_query must be provided.
    agent_alias : str, optional
        The model agent alias/name used for fitting. At least one of subject_id,
        session_date, agent_alias, or from_custom_query must be provided.
    from_custom_query : dict, optional
        A custom MongoDB query dictionary that overrides all other query parameters.
        If provided, subject_id, session_date, and agent_alias are ignored.
    only_recent_version : bool, default=True
        If True, keeps only the most recent version when multiple records have
        the same nwb_name and agent_alias.
    if_include_metrics : bool, default=True
        If True, includes model metrics such as log_likelihood, prediction_accuracy,
        AIC, BIC, LPT scores, cross-validation results, and fitted parameters in
        the returned DataFrame.
    if_include_latent_variables : bool, default=True
        If True, retrieves and merges latent variables (e.g., q_values) from S3
        into the DataFrame. Also computes qvalue_spread (uniformity measure).
    if_download_figures : bool, default=False
        If True, downloads visualization figures from S3 to the local filesystem.
    download_path : str, default="./results/mle_figures/"
        The local directory path where figures will be saved if if_download_figures
        is True.
    paginate_settings : dict, default={"paginate": False}
        Settings for database pagination. Pass {"paginate": True} along with
        pagination parameters for large queries.
    max_threads_for_s3 : int, default=10
        Maximum number of parallel threads to use when downloading latent variables
        and figures from S3.

    Returns
    -------
    pd.DataFrame or None
        A DataFrame containing MLE fitting results with the following columns:

        Always included:
            - _id : Analysis record ID
            - nwb_name : NWB file name
            - agent_alias : Model agent name
            - status : Fitting status ('success' or 'failed')
            - subject_id : Subject identifier
            - session_date : Session date
            - n_trials : Number of trials in the session

        If if_include_metrics=True, also includes:
            - log_likelihood : Model log-likelihood
            - prediction_accuracy : Prediction accuracy on training data
            - k_model : Number of model parameters
            - AIC, BIC : Information criteria
            - LPT, LPT_AIC, LPT_BIC : Local prediction transfer scores
            - prediction_accuracy_test/fit/test_bias_only : Cross-validation arrays
            - prediction_accuracy_10-CV_test/fit/test_bias_only : CV means
            - prediction_accuracy_10-CV_test/fit/test_bias_only_std : CV stds
            - params : Dict of fitted model parameters

        If if_include_latent_variables=True, also includes:
            - latent_variables : Dict containing latent variable arrays (e.g., q_value)
            - qvalue_spread : Uniformity ratio of q-values (0-1 scale)

        Returns None if no records are found.

    Raises
    ------
    ValueError
        If none of subject_id, session_date, agent_alias, or from_custom_query
        are provided.

    Notes
    -----
    - The function queries the 'dynamic-foraging-model-fitting' collection in the
      analysis database with analysis_name='MLE fitting' and
      analysis_ver='first version @ 0.10.0'.
    - If multiple NWB files exist for the same session (duplicated agent_alias),
      a warning is printed suggesting to check timestamps.
    - Only successful fits (status='success') will have latent variables retrieved.
    - The qvalue_spread metric measures the uniformity of q-value distributions
      using normalized entropy (0=concentrated, 1=uniform).

    Examples
    --------
    Get all MLE fitting results for a specific subject:

    >>> df = get_mle_model_fitting(subject_id="12345")

    Get results for a specific session with metrics only:

    >>> df = get_mle_model_fitting(
    ...     subject_id="12345",
    ...     session_date="2025-01-15",
    ...     if_include_latent_variables=False
    ... )

    Get results for a specific model agent and download figures:

    >>> df = get_mle_model_fitting(
    ...     subject_id="730945",
    ...     agent_alias="QLearning_L2F1_CK1_softmax",
    ...     if_download_figures=True,
    ...     download_path="./my_figures/"
    ... )

    Use a custom query to retrieve specific records:

    >>> custom_query = {"subject_id": {"$in": ["12345", "67890"]}}
    >>> df = get_mle_model_fitting(from_custom_query=custom_query)
    """

    # -- Fetch from both AIND Analysis Framework and Han's prototype analysis pipeline --
    records_new = _try_retrieve_records(
        build_query_new_format,
        "AIND Analysis Framework",
        if_include_metrics,
        subject_id,
        session_date,
        agent_alias,
        from_custom_query,
        paginate_settings,
    )

    records_old = _try_retrieve_records(
        build_query_old_format,
        "Han's prototype analysis pipeline",
        if_include_metrics,
        subject_id,
        session_date,
        agent_alias,
        from_custom_query,
        paginate_settings,
    )

    # Create DataFrames from records
    if records_new:
        df_new = pd.DataFrame(records_new)
        df_new["pipeline_source"] = "aind analysis framework"
        print(f"Found {len(df_new)} records in AIND Analysis Framework")
    else:
        df_new = pd.DataFrame()
        print("No records in AIND Analysis Framework")

    if records_old:
        df_old = pd.DataFrame(records_old)
        df_old["pipeline_source"] = "han's analysis pipeline"
        print(f"Found {len(df_old)} records in Han's prototype analysis pipeline")
    else:
        df_old = pd.DataFrame()
        print("No records in Han's prototype analysis pipeline")

    # Concatenate both DataFrames
    if df_new.empty and df_old.empty:
        print(f"No MLE fitting available for {subject_id} on {session_date}")
        return None

    df = pd.concat([df_new, df_old], ignore_index=True)
    print(f"Total: {len(df)} MLE fitting records!")

    # -- Filter for successful fittings early --
    df_success = df.query("status == 'success'")
    df_failed = df.query("status != 'success'")
    print(
        f"--- After filtering for successful fittings: {len(df_success)} records "
        f"({len(df_failed)} skipped) ---"
    )

    # Use only successful fittings for further processing
    df = df_success

    # -- Only keep the recent version if requested --
    if only_recent_version and len(df) > 0:
        # Sort by analysis_time in descending order (most recent first) and keep
        # the first occurrence
        df = (
            df.sort_values("analysis_time", ascending=False)
            .drop_duplicates(subset=["nwb_name", "agent_alias"], keep="first")
            .reset_index(drop=True)
        )
        n_aind_framework = sum(df.pipeline_source == "aind analysis framework")
        n_han_pipeline = sum(df.pipeline_source == "han's analysis pipeline")
        print(f"--- After filtering for most recent versions: {len(df)} records  ---")
        print(f"    AIND Analysis Framework: {n_aind_framework}")
        print(f"    Han's prototype analysis pipeline: {n_han_pipeline}")

    # Add S3_location for old pipeline records (new pipeline already has it from database)
    if "S3_location" not in df.columns:
        df["S3_location"] = df["_id"].apply(lambda id: f"{S3_PATH_ANALYSIS_ROOT}/{id}")
    else:
        # Fill missing S3_location values for old pipeline records
        df.loc[df["S3_location"].isna(), "S3_location"] = df.loc[
            df["S3_location"].isna(), "_id"
        ].apply(lambda id: f"{S3_PATH_ANALYSIS_ROOT}/{id}")

    # If the user specifies one certain session, and there are still multiple nwbs for this session,
    # we warn the user to check nwb time stamps.
    if subject_id and session_date and df.agent_alias.duplicated().any():
        print(
            "WARNING: Duplicated records for the same session and agent_alias!\n"
            "         You should check the nwb_name, n_trials, or pipeline_source\n"
            "         to select the ones you want."
        )

    # -- Get latent variables --
    if if_include_latent_variables and len(df):
        s3_root_list = df["S3_location"].tolist()
        latents = get_s3_latent_variable_batch(
            df._id, s3_root_list=s3_root_list, max_threads_for_s3=max_threads_for_s3
        )
        latents = _add_qvalue_spread(latents)
        df = df.merge(pd.DataFrame(latents), on="_id", how="left")

    # -- Download figures --
    if if_download_figures and len(df):
        s3_root_list = df["S3_location"].tolist()
        f_names = (
            df["nwb_name"].map(lambda x: x.replace(".nwb", "") if x.endswith(".nwb") else x)
            + "_"
            + df.agent_alias
            + "_"
            + df._id.map(lambda x: x[:10])
            + ".png"
        )
        get_s3_mle_figure_batch(
            ids=df._id,
            f_names=f_names,
            s3_root_list=s3_root_list,
            download_path=download_path,
            max_threads_for_s3=max_threads_for_s3,
        )

    return df


if __name__ == "__main__":
    # Old pipeline
    # df = get_mle_model_fitting(subject_id="730945", session_date="2024-10-24",
    # if_download_figures=True)
    # print(df)

    # New pipeline
    df = get_mle_model_fitting(
        subject_id="778869",
        session_date="2025-07-26",
        if_download_figures=True,
        only_recent_version=True,
    )
    print(df)
