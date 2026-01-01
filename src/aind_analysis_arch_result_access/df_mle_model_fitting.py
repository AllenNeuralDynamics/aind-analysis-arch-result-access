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

from aind_analysis_arch_result_access import analysis_docDB_dft
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
    """Get MLE fitting from Han's analysis pipeline (migrated implementation).

    This implementation is copied from `han_pipeline.py` and kept here so callers
    can import it directly from the package without depending on the large
    `han_pipeline` module.
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
