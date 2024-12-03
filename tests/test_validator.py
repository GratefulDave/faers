"""Unit tests for FAERS data validation functionality."""
import unittest
import pandas as pd
import numpy as np
from src.faers_processor.services.validator import DataValidator

class TestDataValidator(unittest.TestCase):
    """Test cases for FAERS data validation."""
    
    def setUp(self):
        """Set up test environment."""
        self.validator = DataValidator()

    def test_validate_demographics(self):
        """Test demographics data validation."""
        # Valid data
        valid_df = pd.DataFrame({
            'primaryid': [1, 2],
            'i_f_code': ['I', 'I'],
            'event_dt': ['20220101', '20220102'],
            'sex': ['M', 'F'],
            'age': [45, 32],
            'age_cod': ['YR', 'YR']
        })
        result = self.validator.validate_demographics(valid_df)
        self.assertTrue(result.valid)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.warnings), 0)

        # Missing required columns
        invalid_df = pd.DataFrame({
            'primaryid': [1],
            'some_column': ['value']
        })
        result = self.validator.validate_demographics(invalid_df)
        self.assertFalse(result.valid)
        self.assertTrue(any('Missing required columns' in err for err in result.errors))

        # Invalid values
        invalid_values_df = pd.DataFrame({
            'primaryid': [1, 2],
            'i_f_code': ['I', 'I'],
            'event_dt': ['20220101', 'invalid_date'],
            'sex': ['X', 'Y'],  # Invalid sex values
            'age': [45, 32],
            'age_cod': ['XX', 'YY']  # Invalid age codes
        })
        result = self.validator.validate_demographics(invalid_values_df)
        self.assertTrue(result.valid)  # Still valid as these are warnings
        self.assertTrue(any('Invalid sex values' in warn for warn in result.warnings))
        self.assertTrue(any('Invalid age codes' in warn for warn in result.warnings))

    def test_validate_drug_info(self):
        """Test drug information validation."""
        # Valid data
        valid_df = pd.DataFrame({
            'primaryid': [1, 2],
            'drug_seq': [1, 1],
            'drugname': ['Drug A', 'Drug B'],
            'role_cod': ['PS', 'SS'],
            'route': ['ORAL', 'INTRAVENOUS']
        })
        result = self.validator.validate_drug_info(valid_df)
        self.assertTrue(result.valid)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.warnings), 0)

        # Missing required columns
        invalid_df = pd.DataFrame({
            'primaryid': [1],
            'drugname': ['Drug A']
        })
        result = self.validator.validate_drug_info(invalid_df)
        self.assertFalse(result.valid)
        self.assertTrue(any('Missing required columns' in err for err in result.errors))

        # Invalid values
        invalid_values_df = pd.DataFrame({
            'primaryid': [1, 2],
            'drug_seq': [1, 1],
            'drugname': ['Drug A', 'Drug B'],
            'role_cod': ['XX', 'YY'],  # Invalid role codes
            'route': ['INVALID', 'ROUTE']  # Invalid routes
        })
        result = self.validator.validate_drug_info(invalid_values_df)
        self.assertTrue(result.valid)  # Still valid as these are warnings
        self.assertTrue(any('Invalid role codes' in warn for warn in result.warnings))
        self.assertTrue(any('Invalid routes' in warn for warn in result.warnings))

    def test_validate_reactions(self):
        """Test reaction data validation."""
        # Valid data
        valid_df = pd.DataFrame({
            'primaryid': [1, 2],
            'pt': ['Reaction A', 'Reaction B']
        })
        result = self.validator.validate_reactions(valid_df)
        self.assertTrue(result.valid)
        self.assertEqual(len(result.errors), 0)

        # Missing required columns
        invalid_df = pd.DataFrame({
            'primaryid': [1]
        })
        result = self.validator.validate_reactions(invalid_df)
        self.assertFalse(result.valid)
        self.assertTrue(any('Missing required columns' in err for err in result.errors))

    def test_validate_outcomes(self):
        """Test outcome data validation."""
        # Valid data
        valid_df = pd.DataFrame({
            'primaryid': [1, 2],
            'outc_cod': ['DE', 'HO']
        })
        result = self.validator.validate_outcomes(valid_df)
        self.assertTrue(result.valid)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.warnings), 0)

        # Invalid outcome codes
        invalid_df = pd.DataFrame({
            'primaryid': [1, 2],
            'outc_cod': ['XX', 'YY']  # Invalid outcome codes
        })
        result = self.validator.validate_outcomes(invalid_df)
        self.assertTrue(result.valid)  # Still valid as these are warnings
        self.assertTrue(any('Invalid outcome codes' in warn for warn in result.warnings))

    def test_validate_therapies(self):
        """Test therapy data validation."""
        # Valid data
        valid_df = pd.DataFrame({
            'primaryid': [1, 2],
            'drug_seq': [1, 1],
            'start_dt': ['20220101', '20220102'],
            'end_dt': ['20220201', '20220202']
        })
        result = self.validator.validate_therapies(valid_df)
        self.assertTrue(result.valid)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.warnings), 0)

        # Invalid dates
        invalid_df = pd.DataFrame({
            'primaryid': [1, 2],
            'drug_seq': [1, 1],
            'start_dt': ['invalid', 'date'],
            'end_dt': ['also', 'invalid']
        })
        result = self.validator.validate_therapies(invalid_df)
        self.assertTrue(result.valid)  # Still valid as these are warnings
        self.assertTrue(any('Invalid dates' in warn for warn in result.warnings))

    def test_validate_data(self):
        """Test generic data validation."""
        # Valid demographics data
        demo_df = pd.DataFrame({
            'primaryid': [1],
            'i_f_code': ['I'],
            'event_dt': ['20220101'],
            'sex': ['M']
        })
        result = self.validator.validate_data(demo_df, 'demo')
        self.assertTrue(result.valid)

        # Unknown data type
        result = self.validator.validate_data(demo_df, 'unknown')
        self.assertFalse(result.valid)
        self.assertTrue(any('Unknown data type' in err for err in result.errors))

if __name__ == '__main__':
    unittest.main()
