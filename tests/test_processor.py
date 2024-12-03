"""Unit tests for FAERS data processor."""
import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
import os

from src.faers_processor.services.processor import (
    FAERSProcessor,
    QuarterSummary,
    TableSummary,
    FAERSProcessingSummary
)
from src.faers_processor.services.standardizer import DataStandardizer
from src.faers_processor.services.validator import DataValidator

class TestFAERSProcessor(unittest.TestCase):
    """Test cases for FAERS data processor."""
    
    def setUp(self):
        """Set up test environment."""
        self.standardizer = DataStandardizer()
        self.processor = FAERSProcessor(self.standardizer)
        self.test_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
        
    def create_test_files(self, quarter_name: str):
        """Create test FAERS files for a quarter."""
        quarter_dir = self.test_dir / quarter_name
        quarter_dir.mkdir(parents=True)
        
        # Create test data
        demo_data = pd.DataFrame({
            'primaryid': ['1', '2'],
            'i_f_code': ['I', 'I'],
            'event_dt': ['20220101', '20220102'],
            'sex': ['M', 'F']
        })
        
        drug_data = pd.DataFrame({
            'primaryid': ['1', '1', '2'],
            'drug_seq': ['1', '2', '1'],
            'drugname': ['Drug A', 'Drug B', 'Drug C']
        })
        
        # Save test files
        demo_data.to_csv(quarter_dir / 'DEMO22Q1.txt', index=False)
        drug_data.to_csv(quarter_dir / 'DRUG22Q1.txt', index=False)
        
        return quarter_dir
        
    def test_process_quarter(self):
        """Test processing of a single quarter."""
        # Create test quarter
        quarter_dir = self.create_test_files('2022q1')
        
        # Process quarter with default settings
        summary = self.processor.process_quarter(quarter_dir, parallel=False, max_workers=None)
        
        # Verify summary
        self.assertIsNotNone(summary)
        self.assertEqual(summary.quarter, '2022q1')
        self.assertEqual(summary.demo_summary.total_rows, 2)
        self.assertEqual(summary.drug_summary.total_rows, 3)
        
        # Test parallel processing
        summary = self.processor.process_quarter(quarter_dir, parallel=True, max_workers=2)
        self.assertIsNotNone(summary)
        self.assertEqual(summary.quarter, '2022q1')

    def test_process_all(self):
        """Test processing of multiple quarters."""
        # Create test quarters
        self.create_test_files('2022q1')
        self.create_test_files('2022q2')
        
        output_dir = self.test_dir / 'output'
        output_dir.mkdir()
        
        # Test sequential processing
        result = self.processor.process_all(self.test_dir, output_dir, parallel=False)
        self.assertEqual(len(result['summaries']), 2)
        
        # Test parallel processing
        result = self.processor.process_all(self.test_dir, output_dir, parallel=True, max_workers=2)
        self.assertEqual(len(result['summaries']), 2)
        
        # Verify report generation
        report_files = list(output_dir.glob('faers_processing_report_*.md'))
        self.assertEqual(len(report_files), 2)

    def test_error_handling(self):
        """Test error handling during processing."""
        # Create quarter with invalid data
        invalid_quarter = self.test_dir / '2022q1'
        invalid_quarter.mkdir(parents=True)
        pd.DataFrame({'invalid': ['data']}).to_csv(invalid_quarter / 'DEMO22Q1.txt', index=False)
        
        # Test sequential processing error handling
        summary = self.processor.process_quarter(invalid_quarter, parallel=False)
        self.assertIsNotNone(summary)
        self.assertTrue(any('Missing required columns' in err 
                          for err in summary.demo_summary.parsing_errors))
        
        # Test parallel processing error handling
        summary = self.processor.process_quarter(invalid_quarter, parallel=True, max_workers=2)
        self.assertIsNotNone(summary)
        self.assertTrue(any('Missing required columns' in err 
                          for err in summary.demo_summary.parsing_errors))
        
        # Test process_all error handling
        output_dir = self.test_dir / 'output'
        output_dir.mkdir()
        
        # Create a mix of valid and invalid quarters
        valid_quarter = self.create_test_files('2022q2')
        
        # Test sequential processing with mixed quarters
        result = self.processor.process_all(self.test_dir, output_dir, parallel=False)
        self.assertEqual(len(result['summaries']), 2)
        
        # Test parallel processing with mixed quarters
        result = self.processor.process_all(self.test_dir, output_dir, parallel=True, max_workers=2)
        self.assertEqual(len(result['summaries']), 2)
        
        # Verify error reporting in processing summary
        report_files = list(output_dir.glob('faers_processing_report_*.md'))
        self.assertGreater(len(report_files), 0)
        
        # Test with invalid max_workers
        with self.assertRaises(ValueError):
            self.processor.process_quarter(invalid_quarter, parallel=True, max_workers=0)
            
        # Test with non-existent directory
        non_existent = self.test_dir / 'non_existent'
        summary = self.processor.process_quarter(non_existent, parallel=False)
        self.assertIsNone(summary)

    def test_parallel_processing(self):
        """Test parallel processing of quarters."""
        # Create multiple test quarters
        quarters = ['2022q1', '2022q2', '2022q3', '2022q4']
        for q in quarters:
            self.create_test_files(q)
            
        output_dir = self.test_dir / 'output'
        output_dir.mkdir()
        
        # Test with different numbers of workers
        worker_counts = [1, 2, 4]
        for workers in worker_counts:
            result = self.processor.process_all(
                self.test_dir, 
                output_dir, 
                parallel=True, 
                max_workers=workers
            )
            self.assertEqual(len(result['summaries']), len(quarters))
            
        # Test parallel processing at quarter level
        quarter_dir = self.test_dir / '2022q1'
        for workers in worker_counts:
            summary = self.processor.process_quarter(
                quarter_dir,
                parallel=True,
                max_workers=workers
            )
            self.assertIsNotNone(summary)
            self.assertEqual(summary.quarter, '2022q1')
            
        # Test parallel processing with large number of workers
        result = self.processor.process_all(
            self.test_dir,
            output_dir,
            parallel=True,
            max_workers=10  # More workers than quarters
        )
        self.assertEqual(len(result['summaries']), len(quarters))
        
        # Verify processing times are recorded
        for summary in result['summaries']:
            self.assertGreater(summary.processing_time, 0)

    def test_summary_generation(self):
        """Test generation of processing summary."""
        # Create and process test data
        quarter_dir = self.create_test_files('2022q1')
        summary = self.processor.process_quarter(quarter_dir)
        
        # Add to processing summary
        self.processor.processing_summary.add_quarter_summary('2022q1', summary)
        
        # Generate report
        output_dir = self.test_dir / 'output'
        output_dir.mkdir()
        report = self.processor.processing_summary.generate_markdown_report(output_dir)
        
        # Verify report content
        self.assertIn('# FAERS Processing Summary Report', report)
        self.assertIn('## Quarter: 2022q1', report)
        self.assertIn('Total Rows:', report)
        
    def test_invalid_quarter_handling(self):
        """Test handling of invalid quarter directories."""
        # Create empty directory
        empty_dir = self.test_dir / 'empty'
        empty_dir.mkdir()
        
        # Try to process
        result = self.processor.process_all(empty_dir, self.test_dir / 'output')
        
        # Verify error handling
        self.assertEqual(len(result['summaries']), 0)
        
    def test_file_reading(self):
        """Test file reading and cleaning."""
        # Create test file with messy data
        df = pd.DataFrame({
            'primaryid': ['1', '2'],
            'messy_column': ['  value  ', ' other value '],
            'empty_column': ['', np.nan]
        })
        
        file_path = self.test_dir / 'test.txt'
        df.to_csv(file_path, index=False)
        
        # Read and clean
        cleaned_df = self.processor._read_and_clean_file(file_path)
        
        # Verify cleaning
        self.assertIsNotNone(cleaned_df)
        self.assertEqual(cleaned_df['messy_column'].iloc[0], 'value')
        self.assertEqual(cleaned_df['empty_column'].iloc[0], '')

    def test_process_dataset(self):
        """Test processing of individual datasets."""
        # Test demographics dataset
        demo_df = pd.DataFrame({
            'primaryid': ['1', '2'],
            'i_f_code': ['I', 'I'],
            'event_dt': ['20220101', '20220102'],
            'sex': ['M', 'F'],
            'age': ['45', '32'],
            'age_cod': ['YR', 'YR']
        })
        
        df, summary = self.processor._process_dataset(demo_df, 'demo', '2022q1', 'DEMO22Q1.txt')
        
        # Verify processing results
        self.assertEqual(len(df), 2)
        self.assertEqual(summary.total_rows, 2)
        self.assertEqual(summary.processed_rows, 2)
        self.assertEqual(len(summary.parsing_errors), 0)
        
        # Test drug dataset with messy data
        drug_df = pd.DataFrame({
            'primaryid': ['1', '2'],
            'drug_seq': ['1', '2'],
            'drugname': ['  Drug A  ', ' Drug B '],
            'route': ['  ORAL  ', ' INTRAVENOUS ']
        })
        
        df, summary = self.processor._process_dataset(drug_df, 'drug', '2022q1', 'DRUG22Q1.txt')
        
        # Verify data cleaning
        self.assertEqual(df['drugname'].iloc[0], 'Drug A')
        self.assertEqual(df['route'].iloc[1], 'INTRAVENOUS')
        
    def test_process_dataset_with_errors(self):
        """Test processing of datasets with various errors."""
        # Test missing required columns
        invalid_demo = pd.DataFrame({
            'some_column': ['value1', 'value2']
        })
        
        df, summary = self.processor._process_dataset(invalid_demo, 'demo', '2022q1', 'DEMO22Q1.txt')
        
        # Verify error handling
        self.assertTrue(any('Missing required columns' in err for err in summary.parsing_errors))
        self.assertGreater(len(summary.data_errors), 0)
        
        # Test invalid data format
        invalid_dates = pd.DataFrame({
            'primaryid': ['1', '2'],
            'i_f_code': ['I', 'I'],
            'event_dt': ['invalid', 'date'],
            'sex': ['M', 'F']
        })
        
        df, summary = self.processor._process_dataset(invalid_dates, 'demo', '2022q1', 'DEMO22Q1.txt')
        
        # Verify date validation
        self.assertTrue(any('Invalid dates' in warn for warn in summary.parsing_errors))
        
    def test_clean_dataframe(self):
        """Test DataFrame cleaning functionality."""
        # Create DataFrame with various data issues
        messy_df = pd.DataFrame({
            'MIXED_CASE': ['Value', 'OTHER'],
            'whitespace_col': ['  extra spaces  ', ' trailing space '],
            'empty_strings': ['', '   ', '\t'],
            'normal_col': ['1', '2']
        })
        
        cleaned_df = self.processor._clean_dataframe(messy_df)
        
        # Verify cleaning results
        self.assertTrue(all(col.islower() for col in cleaned_df.columns))
        self.assertEqual(cleaned_df['whitespace_col'].iloc[0], 'extra spaces')
        self.assertEqual(cleaned_df['empty_strings'].iloc[0], '')
        
    def test_handle_dataset_errors(self):
        """Test error handling for different error types."""
        summary = TableSummary()
        
        # Test ValueError handling
        error = ValueError("Missing required columns: primaryid")
        self.processor._handle_dataset_errors(error, 'demo', '2022q1', 'DEMO22Q1.txt', summary)
        self.assertEqual(summary.data_errors['missing_columns'], 1)
        
        # Test EmptyDataError handling
        error = pd.errors.EmptyDataError("Empty file")
        self.processor._handle_dataset_errors(error, 'demo', '2022q1', 'DEMO22Q1.txt', summary)
        self.assertEqual(summary.data_errors['empty_file'], 1)
        
        # Test unknown error handling
        error = Exception("Unknown error")
        self.processor._handle_dataset_errors(error, 'demo', '2022q1', 'DEMO22Q1.txt', summary)
        self.assertEqual(summary.data_errors['unknown_error'], 1)
        
    def test_process_multiple_files(self):
        """Test processing of multiple files in a quarter."""
        # Create quarter with multiple files
        quarter_dir = self.test_dir / '2022q1'
        quarter_dir.mkdir(parents=True)
        
        # Create files with different data types
        files_data = {
            'DEMO22Q1.txt': pd.DataFrame({
                'primaryid': ['1', '2'],
                'i_f_code': ['I', 'I'],
                'event_dt': ['20220101', '20220102'],
                'sex': ['M', 'F']
            }),
            'DRUG22Q1.txt': pd.DataFrame({
                'primaryid': ['1', '2'],
                'drug_seq': ['1', '1'],
                'drugname': ['Drug A', 'Drug B']
            }),
            'REAC22Q1.txt': pd.DataFrame({
                'primaryid': ['1', '2'],
                'pt': ['Reaction A', 'Reaction B']
            })
        }
        
        for filename, df in files_data.items():
            df.to_csv(quarter_dir / filename, index=False)
            
        # Process quarter
        summary = self.processor.process_quarter(quarter_dir)
        
        # Verify processing results
        self.assertIsNotNone(summary)
        self.assertEqual(summary.demo_summary.total_rows, 2)
        self.assertEqual(summary.drug_summary.total_rows, 2)
        self.assertEqual(summary.reac_summary.total_rows, 2)
        
    def test_parallel_processing_error_handling(self):
        """Test error handling in parallel processing."""
        # Create quarters with both valid and invalid data
        valid_quarter = self.create_test_files('2022q1')
        
        invalid_quarter = self.test_dir / '2022q2'
        invalid_quarter.mkdir(parents=True)
        pd.DataFrame({'invalid': ['data']}).to_csv(invalid_quarter / 'DEMO22Q2.txt', index=False)
        
        output_dir = self.test_dir / 'output'
        output_dir.mkdir()
        
        # Process with parallel=True
        result = self.processor.process_all(self.test_dir, output_dir, parallel=True, max_workers=2)
        
        # Verify error handling
        self.assertEqual(len(result['summaries']), 1)  # Only valid quarter processed successfully

if __name__ == '__main__':
    unittest.main()
