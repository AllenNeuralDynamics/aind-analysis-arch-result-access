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

    def test_split_nwb_name_invalid(self):
        """Test split_nwb_name with invalid input"""
        invalid_cases = [
            "invalid_name.nwb",
            "no_pattern_here",
            "",
        ]
        for nwb_name in invalid_cases:
            with self.subTest(nwb_name=nwb_name):
                result = reformat.split_nwb_name(nwb_name)
                self.assertEqual(result, (None, None, 0))

    def test_trainer_mapper(self):
        """Test trainer_mapper for normalizing trainer names"""
        test_cases = {
            "Avalon": "Avalon Amaya",
            "Ella": "Ella Hilton",
            "Katrina": "Katrina Nguyen",
            "Lucas": "Lucas Kinsey",
            "Travis": "Travis Ramirez",
            "Xinxin": "Xinxin Yin",
            "the ghost": "Xinxin Yin",
            "Bowen": "Bowen Tan",
            "Henry Loeffer": "Henry Loeffler",
            "margaret lee": "Margaret Lee",
            "Madseline Tom": "Madeline Tom",
            "Unknown Trainer": "Unknown Trainer",  # No mapping, returns as-is
        }
        for input_name, expected in test_cases.items():
            with self.subTest(input_name=input_name):
                self.assertEqual(reformat.trainer_mapper(input_name), expected)

    def test_data_source_mapper(self):
        """Test data_source_mapper for extracting rig information"""
        test_cases = {
            "bpod_rig": ("Janelia", "training", "NA", "bpod", "Janelia_training_NA_bpod"),
            "AIND_bpod_rig": ("AIND", "training", "347", "bpod", "AIND_training_347_bpod"),
            "Ephys-Han": ("AIND", "ephys", "321", "bonsai", "AIND_ephys_321_bonsai"),
            "447_rig": ("AIND", "training", "447", "bonsai", "AIND_training_447_bonsai"),
            "446_rig": ("AIND", "training", "446", "bonsai", "AIND_training_446_bonsai"),
            "323_rig": ("AIND", "training", "323", "bonsai", "AIND_training_323_bonsai"),
            "322_rig": ("AIND", "training", "322", "bonsai", "AIND_training_322_bonsai"),
            "ephys_rig": ("AIND", "ephys", "323", "bonsai", "AIND_ephys_323_bonsai"),
            "default_rig": ("AIND", "training", "447", "bonsai", "AIND_training_447_bonsai"),
        }
        for rig_name, expected in test_cases.items():
            with self.subTest(rig_name=rig_name):
                self.assertEqual(reformat.data_source_mapper(rig_name), expected)

    def test_curriculum_ver_mapper(self):
        """Test curriculum_ver_mapper for grouping curriculum versions"""
        test_cases = {
            "1.0": "v1",
            "1.0.1": "v1",
            "2.0": "v2",
            "2.1": "v2",
            "2.2": "v2",
            "2.3": "v3",
            "2.3.1": "v3",
            None: None,
            123: None,  # Non-string input
        }
        for version, expected in test_cases.items():
            with self.subTest(version=version):
                self.assertEqual(reformat.curriculum_ver_mapper(version), expected)


if __name__ == "__main__":
    unittest.main()
