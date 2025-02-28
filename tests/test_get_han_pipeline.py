"""Test get_streamlit_master_table.py"""

import unittest

from aind_analysis_arch_result_access.han_pipeline import (
    get_mle_model_fitting,
    get_session_table,
)


class TestGetMasterSessionTable(unittest.TestCase):
    """Get Han's pipeline master session table."""

    def test_get_session_table(self):
        """Example of how to test the truth of a statement."""

        df = get_session_table(if_load_bpod=False)
        self.assertIsNotNone(df)
        print(df.head())
        print(df.columns)

        df_bpod = get_session_table(if_load_bpod=True)
        self.assertIsNotNone(df)
        self.assertGreater(len(df_bpod), len(df))
        print(df_bpod.head())


class TestGetMLEModelFitting(unittest.TestCase):
    """Get MLE model fitting results"""

    def test_get_mle_model_fitting(self):
        """Example of how to test the truth of a statement."""

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


if __name__ == "__main__":
    unittest.main()
