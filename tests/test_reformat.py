"""Test reformat"""

import unittest

from aind_analysis_arch_result_access.util import reformat


class TestReformat(unittest.TestCase):
    """Test util functions for reformatting data"""

    def test_split_nwb_name(self):
        """Test split_nwb_name"""

        test_cases = {
            "721403_2024-08-09_08-39-12.nwb": ("721403", "2024-08-09", 83912),
            "685641_2023-10-04.nwb": ("685641", "2023-10-04", 0),
            "behavior_754280_2024-11-14_11-06-24.nwb": ("754280", "2024-11-14", 110624),
            "behavior_1_2024-08-05_15-48-54": ("1", "2024-08-05", 154854),
        }

        for nwb_name, expected in test_cases.items():
            with self.subTest(nwb_name=nwb_name):
                self.assertEqual(reformat.split_nwb_name(nwb_name), expected)


if __name__ == "__main__":
    unittest.main()
