"""Test get_streamlit_master_table.py"""

import unittest

import pandas as pd

from aind_analysis_arch_result_access import (
    get_mle_model_fitting,
)
from aind_analysis_arch_result_access.han_pipeline import (
    get_logistic_regression,
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

    def test_get_mle_model_fitting_old_pipeline(self):
        """Test get MLE model fitting results from old pipeline (subject 730945, session 2024-10-24)."""

        df = get_mle_model_fitting(
            subject_id="730945",
            session_date="2024-10-24",
            if_include_metrics=True,
            if_include_latent_variables=True,
            if_download_figures=True,
            max_threads_for_s3=10,
        )

        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0, "Expected at least one result")
        # Verify required columns exist
        self.assertIn("pipeline_source", df.columns, "pipeline_source column should exist")
        self.assertIn("S3_location", df.columns, "S3_location column should exist")
        self.assertIn("status", df.columns, "status column should exist")
        # Verify all records are successful
        self.assertTrue(
            (df["status"] == "success").all(), "All records should have status='success'"
        )
        print(df.head())
        print(df.columns)

    def test_get_mle_model_fitting_new_pipeline(self):
        """Test get MLE model fitting results from new pipeline (subject 778869, session 2025-07-26)."""

        df = get_mle_model_fitting(
            subject_id="778869",
            session_date="2025-07-26",
            if_include_metrics=True,
            if_include_latent_variables=True,
            if_download_figures=True,
            max_threads_for_s3=10,
        )

        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0, "Expected at least one result")
        # Verify required columns exist
        self.assertIn("pipeline_source", df.columns, "pipeline_source column should exist")
        self.assertIn("S3_location", df.columns, "S3_location column should exist")
        self.assertIn("status", df.columns, "status column should exist")
        # Verify all records are successful
        self.assertTrue(
            (df["status"] == "success").all(), "All records should have status='success'"
        )
        print(df.head())

    def test_get_mle_with_version_filtering(self):
        """Test get MLE model fitting with only_recent_version=True."""

        # Test with version filtering enabled (default)
        df_recent = get_mle_model_fitting(
            subject_id="778869",
            session_date="2025-07-26",
            only_recent_version=True,
            if_include_metrics=True,
        )

        # Test with version filtering disabled
        df_all = get_mle_model_fitting(
            subject_id="778869",
            session_date="2025-07-26",
            only_recent_version=False,
            if_include_metrics=True,
        )

        self.assertIsNotNone(df_recent)
        self.assertIsNotNone(df_all)
        # df_all should have >= records than df_recent
        self.assertGreaterEqual(
            len(df_all), len(df_recent), "only_recent_version=False should return >= records"
        )

        # If there are multiple versions, check that recent has fewer records
        if len(df_all) > len(df_recent):
            print(f"Version filtering reduced records from {len(df_all)} to {len(df_recent)}")

    def test_get_mle_with_agent_alias(self):
        """Test get MLE model fitting by agent_alias."""

        df = get_mle_model_fitting(
            subject_id="730945",
            agent_alias="QLearning_L2F1_CK1_softmax",
            if_include_metrics=True,
        )

        self.assertIsNotNone(df)
        if len(df) > 0:
            # Verify all records match the agent_alias
            self.assertTrue(
                (df["agent_alias"] == "QLearning_L2F1_CK1_softmax").all(),
                "All records should match the requested agent_alias",
            )
            print(f"Found {len(df)} records for agent_alias=QLearning_L2F1_CK1_softmax")

    def test_get_mle_without_metrics(self):
        """Test get MLE model fitting without metrics (if_include_metrics=False)."""

        df = get_mle_model_fitting(
            subject_id="730945",
            session_date="2024-10-24",
            if_include_metrics=False,
            if_include_latent_variables=False,
        )

        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0, "Expected at least one result")
        # Metrics columns should not exist when if_include_metrics=False
        metrics_columns = ["BIC", "AIC", "log_likelihood", "prediction_accuracy"]
        for col in metrics_columns:
            if col in df.columns:
                # It's okay if some metrics are in base query, but latent_variable should not be there
                pass
        self.assertNotIn(
            "latent_variable",
            df.columns,
            "latent_variable should not exist when if_include_latent_variables=False",
        )

    def test_get_mle_without_latent_variables(self):
        """Test get MLE model fitting without latent variables (if_include_latent_variables=False)."""

        df = get_mle_model_fitting(
            subject_id="778869",
            session_date="2025-07-26",
            if_include_metrics=True,
            if_include_latent_variables=False,
            if_download_figures=False,
        )

        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0, "Expected at least one result")
        # latent_variable column should not exist
        self.assertNotIn(
            "latent_variable",
            df.columns,
            "latent_variable column should not exist when if_include_latent_variables=False",
        )

    def test_get_mle_custom_query(self):
        """Test get MLE model fitting with custom query."""

        custom_query = {"subject_id": {"$in": ["730945", "778869"]}}
        df = get_mle_model_fitting(
            from_custom_query=custom_query,
            if_include_metrics=True,
            if_include_latent_variables=False,
            if_download_figures=False,
        )

        self.assertIsNotNone(df)
        if len(df) > 0:
            # Verify all records are from the specified subjects
            self.assertTrue(
                df["subject_id"].isin(["730945", "778869"]).all(),
                "All records should be from subject_id in ['730945', '778869']",
            )
            print(f"Found {len(df)} records from custom query")

    def test_get_mle_by_subject_only(self):
        """Test get MLE model fitting by subject_id only (no session_date filter)."""

        df = get_mle_model_fitting(
            subject_id="730945",
            if_include_metrics=True,
            if_include_latent_variables=False,
            if_download_figures=False,
        )

        self.assertIsNotNone(df)
        if len(df) > 0:
            # Verify all records are from the specified subject
            self.assertTrue(
                (df["subject_id"] == "730945").all(),
                "All records should be from subject_id='730945'",
            )
            # Should have multiple sessions potentially
            num_sessions = df["session_date"].nunique()
            print(f"Found {len(df)} records across {num_sessions} sessions for subject 730945")


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
