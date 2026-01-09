"""
Migrated get_mle_model_fitting implementation.
This module contains the MLE model fitting function moved out of han_pipeline.py
so callers can import it directly from the package.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import entropy

from aind_data_access_api.document_db import MetadataDbClient
from aind_analysis_arch_result_access.util.s3 import (
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


def build_query(from_custom_query=None, subject_id=None, session_date=None, agent_alias=None):
    """Build query for MLE fitting (copied from han_pipeline to avoid circular imports)."""
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
    ...     agent_alias="RL_model_v2",
    ...     if_download_figures=True,
    ...     download_path="./my_figures/"
    ... )
    
    Use a custom query to retrieve specific records:
    
    >>> custom_query = {"subject_id": {"$in": ["12345", "67890"]}}
    >>> df = get_mle_model_fitting(from_custom_query=custom_query)
    """

    # -- Build query --
    filter_query = build_query(from_custom_query, subject_id, session_date, agent_alias)

    projection = {
        "_id": 1,
        "nwb_name": 1,
        "analysis_results.fit_settings.agent_alias": 1,
        "status": 1,
        "subject_id": 1,
        "session_date": 1,
        "analysis_results.n_trials": 1,
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
    print(f"Query: {filter_query}")
    records = analysis_docDB_dft.retrieve_docdb_records(
        filter_query=filter_query,
        projection=projection,
        **paginate_settings,
    )

    if not records:
        print(f"No MLE fitting available for {subject_id} on {session_date}")
        return None

    print(f"Found {len(records)} MLE fitting records!")

    # -- Reformat the records --
    # Turn the nested json into a flat DataFrame and rename the columns, except params
    if if_include_metrics:
        params = [
            record["analysis_results"].pop("params") if record["status"] == "success" else None
            for record in records
        ]
    df = pd.json_normalize(records)
    df = df.rename(
        columns={
            col: col.replace("analysis_results.", "")
            .replace("cross_validation.", "")
            .replace("fit_settings.", "")
            for col in df.columns
        }
    )

    # If the user specifies one certain session, and there are multiple nwbs for this session,
    # we warn the user to check nwb time stamps.
    if subject_id and session_date and df.agent_alias.duplicated().any():
        print(
            "Duplicated agent_alias!\n"
            "There are multiple nwbs for this session:\n"
            f"{df.nwb_name.unique()}\n"
            "You should check the time stamps to select the one you want."
        )

    # -- Some post-processing of metrics --
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

    # -- Get latent variables --
    df_success = df.query("status == 'success'")
    print(f"Found {len(df_success)} successful MLE fitting!")
    if not len(df_success):
        return df

    if if_include_latent_variables:
        latents = get_s3_latent_variable_batch(
            df_success._id, max_threads_for_s3=max_threads_for_s3
        )
        latents = _add_qvalue_spread(latents)
        df = df.merge(pd.DataFrame(latents), on="_id", how="left")

    # -- Download figures --
    if if_download_figures:
        f_names = (
            df.nwb_name.map(lambda x: x.replace(".nwb", ""))
            + "_"
            + df.agent_alias
            + "_"
            + df._id.map(lambda x: x[:10])
            + ".png"
        )  # Build the file names
        get_s3_mle_figure_batch(
            ids=df_success._id,
            f_names=f_names,
            download_path=download_path,
            max_threads_for_s3=max_threads_for_s3,
        )

    return df
