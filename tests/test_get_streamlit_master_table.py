"""Example test template."""

import unittest

from aind_analysis_arch_result_access.han_pipeline import get_session_table


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


if __name__ == "__main__":
    unittest.main()
