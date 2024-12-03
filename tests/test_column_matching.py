"""Unit tests for FAERS column matching functionality."""
import unittest
import pandas as pd
from pathlib import Path
from src.faers_processor.services.standardizer import DataStandardizer

class TestColumnMatching(unittest.TestCase):
    """Test cases for column name matching and standardization."""
    
    def setUp(self):
        """Set up test environment."""
        self.external_dir = Path("tests/data/external")
        self.output_dir = Path("tests/data/output")
        self.standardizer = DataStandardizer(self.external_dir, self.output_dir)

    def test_get_column_case_insensitive(self):
        """Test case-insensitive column name retrieval."""
        # Test data
        df = pd.DataFrame({
            'DRUGNAME': [1],
            'sex': [2],
            'Age_Cod': [3]
        })

        # Test exact match
        self.assertEqual(
            self.standardizer._get_column_case_insensitive(df, 'DRUGNAME'),
            'DRUGNAME'
        )

        # Test lowercase match
        self.assertEqual(
            self.standardizer._get_column_case_insensitive(df, 'sex'),
            'sex'
        )

        # Test mixed case match
        self.assertEqual(
            self.standardizer._get_column_case_insensitive(df, 'age_cod'),
            'Age_Cod'
        )

        # Test non-existent column
        self.assertIsNone(
            self.standardizer._get_column_case_insensitive(df, 'nonexistent')
        )

    def test_standardize_demographics_columns(self):
        """Test demographics column standardization."""
        # Test data with various column name cases
        df = pd.DataFrame({
            'I_F_COD': ['I'],
            'GNDR_COD': ['M'],
            'age': [45],
            'Age_Cod': ['YR'],
            'EVENT_DT': ['20220101'],
            'reporter_country': ['US']
        })

        # Standardize
        result = self.standardizer.standardize_demographics(df)

        # Check standardized column names
        expected_columns = {
            'i_f_code', 'sex', 'age', 'age_cod',
            'event_dt', 'country'
        }
        self.assertEqual(set(result.columns), expected_columns)

        # Check values preserved
        self.assertEqual(result['i_f_code'].iloc[0], 'I')
        self.assertEqual(result['sex'].iloc[0], 'M')
        self.assertEqual(result['age'].iloc[0], 45)

    def test_standardize_drug_info_columns(self):
        """Test drug information column standardization."""
        # Test data with various column name cases
        df = pd.DataFrame({
            'DRUGNAME': ['Drug A'],
            'DRUG_SEQ': [1],
            'ROLE_COD': ['PS'],
            'route': ['ORAL'],
            'DOSE_AMT': ['100'],
            'dose_unit': ['MG']
        })

        # Standardize
        result = self.standardizer.standardize_drug_info(df)

        # Check standardized column names
        expected_columns = {
            'drugname', 'drug_seq', 'role_cod', 'route',
            'dose_amt', 'dose_unit', 'prod_ai'
        }
        self.assertEqual(set(result.columns), expected_columns)

        # Check values preserved
        self.assertEqual(result['drugname'].iloc[0], 'Drug A')
        self.assertEqual(result['route'].iloc[0], 'ORAL')

    def test_missing_column_handling(self):
        """Test handling of missing required columns."""
        # Test data missing required columns
        df = pd.DataFrame({
            'some_column': [1]
        })

        # Test demographics
        demo_result = self.standardizer.standardize_demographics(df)
        self.assertIn('i_f_code', demo_result.columns)
        self.assertEqual(demo_result['i_f_code'].iloc[0], 'I')

        # Test drug info
        drug_result = self.standardizer.standardize_drug_info(df)
        self.assertIn('drugname', drug_result.columns)
        self.assertTrue(pd.isna(drug_result['drugname'].iloc[0]))

if __name__ == '__main__':
    unittest.main()
