"""Test get_streamlit_master_table.py"""

import unittest

import pandas as pd

from aind_analysis_arch_result_access.han_pipeline import (
    get_logistic_regression,
    get_mle_model_fitting,
    get_session_table,
)


class TestGetMasterSessionTable(unittest.TestCase):
    """Get Han's pipeline master session table."""

    def test_get_session_table(self):
        """Test get session table for a specific subject and session date."""

        df = get_session_table(if_load_bpod=False)
        self.assertIsNotNone(df)
        print(df.head())
        print(df.columns)

        df_bpod = get_session_table(if_load_bpod=True)
        self.assertIsNotNone(df)
        self.assertGreater(len(df_bpod), len(df))
        print(df_bpod.head())

    def test_get_recent_sessions(self):
        """Test get session table for sessions from the last 6 months."""

        # Get sessions from the last 6 months using the parameter
        months = 6
        df = get_session_table(if_load_bpod=False, only_recent_n_month=months)
        self.assertIsNotNone(df)

        # Calculate date 6 months ago using the same method as the pipeline
        cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=months)

        # Verify no session is earlier than 6 months ago
        self.assertTrue(
            (df["session_date"] >= cutoff_date).all(),
            f"Found sessions older than {cutoff_date.date()}. "
            f"Earliest session: {df['session_date'].min()}",
        )

        # Verify we have some recent sessions
        self.assertGreater(len(df), 0)
        print(f"Found {len(df)} sessions in the last {months} months")
        print(f"Cutoff date: {cutoff_date.date()}")
        print(f"Earliest session: {df['session_date'].min()}")
        print(f"Latest session: {df['session_date'].max()}")
        print(df.head())


class TestGetMLEModelFitting(unittest.TestCase):
    """Get MLE model fitting results"""

    def test_get_mle_model_fitting(self):
        """Test get MLE model fitting results for a specific subject and session date."""

        df = get_mle_model_fitting(
            subject_id="730945",
            session_date="2024-10-24",
            if_include_metrics=True,
            if_include_latent_variables=True,
            if_download_figures=True,
            max_threads_for_s3=10,
        )

        self.assertIsNotNone(df)
        print(df.head())
        print(df.columns)


class TestGetLogisticRegression(unittest.TestCase):
    """Get logistic regression results"""

    def test_get_logistic_regression_valid_and_invalid(self):
        """Test get logistic regression results for a specific subject and session date."""

        # -- Test with a valid and invalid session id
        df_sessions = pd.DataFrame(
            {
                "subject_id": ["mouse not exists", "769253"],
                "session_date": ["2025-03-12", "2025-03-12"],
            }
        )
        df = get_logistic_regression(
            df_sessions=df_sessions,
            model="Su2022",
            if_download_figures=False,
        )
        self.assertEqual(len(df), 1)
        print(df.head())

    def test_get_logistic_regression_all_invalid(self):
        """Test get logistic regression results where all session ids are invalid."""

        # -- Test with a valid and invalid session id
        df_sessions = pd.DataFrame(
            {
                "subject_id": ["mouse not exists"],
                "session_date": ["2025-03-12"],
            }
        )
        df = get_logistic_regression(
            df_sessions=df_sessions,
            model="Su2022",
            if_download_figures=False,
        )
        self.assertEqual(len(df), 0)

    def test_invalid_model(self):
        """Test get logistic regression results with an invalid model."""

        # -- Test with a valid and invalid session id
        df_sessions = pd.DataFrame(
            {
                "subject_id": ["769253"],
                "session_date": ["2025-03-12"],
            }
        )
        with self.assertRaises(ValueError):
            get_logistic_regression(
                df_sessions=df_sessions,
                model="invalid_model",
                if_download_figures=False,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
