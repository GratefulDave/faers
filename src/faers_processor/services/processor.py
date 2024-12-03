"""Service for processing FAERS data files."""
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import io
import os
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
from datetime import datetime
import sys
import chardet

from .standardizer import DataStandardizer


class ColoredFormatter(logging.Formatter):
    """Custom formatter adding colors to log levels."""
    
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.INFO: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%y-%m-%d %H:%M:%S')
        return formatter.format(record)

def setup_logging():
    """Setup logging with colors and proper formatting."""
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler with colored formatting
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)
    
    return logger

@dataclass
class TableSummary:
    """Summary statistics for a single FAERS table."""
    total_rows: int = 0
    processed_rows: int = 0
    invalid_dates: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    missing_columns: Dict[str, str] = field(default_factory=dict)  # column -> default value
    parsing_errors: List[str] = field(default_factory=list)
    data_errors: Dict[str, int] = field(default_factory=lambda: defaultdict(int))  # error type -> count
    processing_time: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate percentage of successfully processed rows."""
        return (self.processed_rows / self.total_rows * 100) if self.total_rows > 0 else 0.0

    def add_missing_column(self, col: str, default_value: str):
        """Track missing column and its default value."""
        self.missing_columns[col] = default_value
        logging.warning(f"Required column '{col}' not found, adding with default value: {default_value}")

    def add_invalid_date(self, field: str, count: int):
        """Track invalid dates by field."""
        self.invalid_dates[field] = count
        logging.warning(f"{count}/{self.total_rows} rows ({count/self.total_rows*100:.1f}%) had invalid dates in {field}")

    def add_data_error(self, error_type: str):
        """Track data validation errors."""
        self.data_errors[error_type] += 1
        
    def add_parsing_error(self, error: str):
        """Track parsing errors."""
        self.parsing_errors.append(error)
        logging.error(f"Parsing error: {error}")


@dataclass
class QuarterSummary:
    """Summary statistics for a FAERS quarter."""
    quarter: str
    demo_summary: TableSummary = field(default_factory=TableSummary)
    drug_summary: TableSummary = field(default_factory=TableSummary)
    reac_summary: TableSummary = field(default_factory=TableSummary)
    outc_summary: TableSummary = field(default_factory=TableSummary)
    rpsr_summary: TableSummary = field(default_factory=TableSummary)
    ther_summary: TableSummary = field(default_factory=TableSummary)
    indi_summary: TableSummary = field(default_factory=TableSummary)
    processing_time: float = 0.0

    def log_summary(self):
        """Log processing summary for this quarter."""
        logging.info(f"\nProcessing Summary for Quarter {self.quarter}:")
        for data_type in ['demo', 'drug', 'reac', 'outc', 'rpsr', 'ther', 'indi']:
            summary = getattr(self, f"{data_type}_summary")
            if summary.total_rows > 0:
                logging.info(f"\n{data_type.upper()} Summary:")
                logging.info(f"Total Rows: {summary.total_rows}")
                logging.info(f"Processed Rows: {summary.processed_rows}")
                if summary.missing_columns:
                    logging.info("Missing Columns:")
                    for col, default in summary.missing_columns.items():
                        logging.info(f"  - {col} (default: {default})")
                if summary.invalid_dates:
                    logging.info("Invalid Dates:")
                    for field, count in summary.invalid_dates.items():
                        pct = (count/summary.total_rows*100) if summary.total_rows > 0 else 0
                        logging.info(f"  - {field}: {count}/{summary.total_rows} rows ({pct:.1f}%)")
                if summary.parsing_errors:
                    logging.info("Parsing Errors:")
                    for error in summary.parsing_errors:
                        logging.info(f"  - {error}")


class FAERSProcessingSummary:
    """Tracks and generates summary reports for FAERS data processing."""

    def __init__(self):
        self.quarter_summaries: Dict[str, QuarterSummary] = {}

    def add_quarter_summary(self, quarter: str, summary: QuarterSummary):
        """Add summary for a processed quarter."""
        self.quarter_summaries[quarter] = summary

    def generate_markdown_report(self) -> str:
        """Generate a detailed markdown report of all processing results."""
        report = ["# FAERS Processing Summary Report\n"]
        
        # Sort quarters for consistent reporting
        sorted_quarters = sorted(self.quarter_summaries.keys())
        
        for quarter in sorted_quarters:
            summary = self.quarter_summaries[quarter]
            report.append(f"\n## Quarter: {quarter}")
            report.append(f"Processing Time: {summary.processing_time:.2f} seconds\n")
            
            # Process each file type
            for data_type in ['demo', 'drug', 'reac', 'outc', 'rpsr', 'ther', 'indi']:
                table_summary = getattr(summary, f"{data_type}_summary")
                if table_summary.total_rows > 0:
                    report.append(f"\n### {data_type.upper()} File")
                    
                    # Basic statistics
                    report.append("#### Processing Statistics")
                    report.append(f"- Total Rows: {table_summary.total_rows:,}")
                    report.append(f"- Processed Rows: {table_summary.processed_rows:,}")
                    report.append(f"- Success Rate: {(table_summary.processed_rows/table_summary.total_rows*100):.1f}%")
                    report.append(f"- Processing Time: {table_summary.processing_time:.2f} seconds")
                    
                    # Missing Columns
                    if table_summary.missing_columns:
                        report.append("\n#### Missing Columns")
                        report.append("| Column | Default Value |")
                        report.append("|--------|---------------|")
                        for col, default in table_summary.missing_columns.items():
                            report.append(f"| {col} | {default} |")
                    
                    # Invalid Dates
                    if table_summary.invalid_dates:
                        report.append("\n#### Invalid Dates")
                        report.append("| Field | Invalid Count | Percentage |")
                        report.append("|-------|---------------|------------|")
                        for field, count in table_summary.invalid_dates.items():
                            pct = (count/table_summary.total_rows*100)
                            report.append(f"| {field} | {count:,}/{table_summary.total_rows:,} | {pct:.1f}% |")
                    
                    # Data Errors
                    if table_summary.data_errors:
                        report.append("\n#### Data Validation Errors")
                        report.append("| Error Type | Count |")
                        report.append("|------------|--------|")
                        for error_type, count in table_summary.data_errors.items():
                            report.append(f"| {error_type} | {count:,} |")
                    
                    # Parsing Errors
                    if table_summary.parsing_errors:
                        report.append("\n#### Parsing Errors")
                        for error in table_summary.parsing_errors:
                            report.append(f"- {error}")
                            
        return "\n".join(report)

    def save_report(self, output_dir: Path):
        """Save the processing report to a markdown file."""
        report = self.generate_markdown_report()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"faers_processing_report_{timestamp}.md"
        
        report_path.write_text(report)
        logging.info(f"Saved processing report to: {report_path}")


class FAERSProcessor:
    """Processor for FAERS data files."""

    def __init__(self, standardizer: DataStandardizer, use_parallel: bool = False):
        """Initialize the FAERS processor.

        Args:
            standardizer: DataStandardizer instance for data cleaning
            use_parallel: Whether to use parallel processing
        """
        self.standardizer = standardizer
        self.use_parallel = use_parallel
        self.logger = setup_logging()

    def process_all(self, input_dir: Path, output_dir: Path, max_workers: int = None) -> None:
        """Process all quarters in the input directory and save merged results in output_dir."""
        import time
        from datetime import datetime

        # Convert to absolute paths and resolve any symlinks
        input_dir = input_dir.resolve()
        output_dir = output_dir.resolve()

        self.logger.info(f"Using absolute paths:")
        self.logger.info(f"Input directory: {input_dir}")
        self.logger.info(f"Output directory: {output_dir}")

        # Initialize results tracking
        results = {
            'total_quarters': 0,
            'success': [],
            'failed': [],
            'skipped': [],
            'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'max_workers': max_workers
        }
        start_time = time.time()

        # Ensure output directory exists and is clean
        if output_dir.exists():
            self.logger.info(f"Cleaning output directory: {output_dir}")
            for file in output_dir.glob('*.txt'):
                file.unlink()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all quarter directories
        quarter_dirs = [d for d in input_dir.iterdir() if d.is_dir() and re.match(r'\d{4}Q[1-4]', d.name, re.IGNORECASE)]
        total_quarters = len(quarter_dirs)
        results['total_quarters'] = total_quarters

        if not quarter_dirs:
            self.logger.error(f"No quarter directories found in {input_dir}")
            return

        self.logger.info(f"Found {total_quarters} quarters to process")

        # Initialize dictionaries to store DataFrames for each data type
        all_data = {
            'demo': [],
            'drug': [],
            'reac': [],
            'outc': [],
            'rpsr': [],
            'ther': [],
            'indi': []
        }

        # Create progress bar for total quarters
        pbar = tqdm(total=total_quarters, desc="Processing FAERS quarters", 
                   unit="quarter", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} quarters "
                   "[{elapsed}<{remaining}, {rate_fmt}]")

        if self.use_parallel:
            if max_workers is None:
                max_workers = max(1, multiprocessing.cpu_count() - 1)

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Process all quarters with futures
                futures = {
                    executor.submit(self.process_quarter, quarter_dir): quarter_dir 
                    for quarter_dir in quarter_dirs
                }

                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    quarter_dir = futures[future]
                    quarter = quarter_dir.name
                    try:
                        results_dict = future.result()

                        # Add quarter column and append to all_data
                        for data_type, df in results_dict.items():
                            if not df.empty:
                                df['quarter'] = quarter
                                all_data[data_type].append(df)

                        results['success'].append(quarter)
                        pbar.update(1)
                        pbar.set_postfix({"quarter": quarter, "status": "success"}, refresh=True)

                    except Exception as e:
                        results['failed'].append((quarter, str(e)))
                        self.logger.error(f"Error processing {quarter}: {str(e)}")
                        pbar.update(1)
                        pbar.set_postfix({"quarter": quarter, "status": "failed"}, refresh=True)
        else:
            # Sequential processing
            for quarter_dir in quarter_dirs:
                quarter = quarter_dir.name
                try:
                    results_dict = self.process_quarter(quarter_dir)

                    # Add quarter column and append to all_data
                    for data_type, df in results_dict.items():
                        if not df.empty:
                            df['quarter'] = quarter
                            all_data[data_type].append(df)

                    results['success'].append(quarter)
                    pbar.update(1)
                    pbar.set_postfix({"quarter": quarter, "status": "success"}, refresh=True)

                except Exception as e:
                    results['failed'].append((quarter, str(e)))
                    self.logger.error(f"Error processing {quarter}: {str(e)}")
                    pbar.update(1)
                    pbar.set_postfix({"quarter": quarter, "status": "failed"}, refresh=True)

        pbar.close()

        # Merge and save all data types
        for data_type, dfs in all_data.items():
            if dfs:
                try:
                    self.logger.info(f"Merging {len(dfs)} quarters for {data_type}")
                    merged_df = pd.concat(dfs, ignore_index=True)

                    # Convert numeric columns
                    if 'primaryid' in merged_df.columns:
                        merged_df['primaryid'] = pd.to_numeric(merged_df['primaryid'], errors='coerce')
                    if data_type == 'demo':
                        if 'caseid' in merged_df.columns:
                            merged_df['caseid'] = pd.to_numeric(merged_df['caseid'], errors='coerce')
                        if 'age' in merged_df.columns:
                            merged_df['age'] = pd.to_numeric(merged_df['age'], errors='coerce')
                    elif data_type == 'drug' and 'drug_seq' in merged_df.columns:
                        merged_df['drug_seq'] = pd.to_numeric(merged_df['drug_seq'], errors='coerce')

                    # Sort by primaryid and quarter
                    merged_df = merged_df.sort_values(['primaryid', 'quarter'])

                    # Save merged file
                    output_file = output_dir.resolve() / f'{data_type}.txt'
                    self.logger.info(f"Saving {data_type} to: {output_file}")
                    merged_df.to_csv(output_file, sep='$', index=False, encoding='utf-8')
                    self.logger.info(f"Successfully saved {data_type} to {output_file}")
                    self.logger.info(f"Shape: {merged_df.shape}")
                except Exception as e:
                    self.logger.error(f"Error merging {data_type}: {str(e)}")

        # Record end time and duration
        results['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results['duration'] = time.time() - start_time

        # Generate summary report
        self.generate_summary_report(output_dir, results)

        # Print final summary
        print(f"\nProcessing completed in {results['duration']:.2f} seconds")
        print(f"See detailed report at: {output_dir}/processing_report.md")

    def generate_summary_report(self, output_dir: Path, results: Dict) -> None:
        """Generate a markdown summary report of processing results.

        Args:
            output_dir: Directory where processed files are saved
            results: Dictionary containing processing results
        """
        report_path = output_dir / "processing_report.md"

        summary = FAERSProcessingSummary()

        for quarter in results['success']:
            summary.add_quarter_summary(quarter, QuarterSummary(quarter))

        summary.save_report(output_dir)

        self.logger.info(f"Generated processing report: {report_path}")

    def _find_ascii_directory(self, quarter_dir: Path) -> Optional[Path]:
        """Find the ASCII directory - it's always ASCII or ascii.
        
        NOTE: DO NOT MODIFY THIS METHOD - case insensitive directory finding is working as intended.
        """
        # Log current directory for debugging
        self.logger.info(f"Searching for ASCII directory in: {quarter_dir}")
        
        # Use rglob to find any variation of 'ascii' directory
        for item in quarter_dir.rglob('[aA][sS][cC][iI][iI]'):
            if item.is_dir():
                self.logger.info(f"Found ASCII directory: {item}")
                return item
                
        self.logger.warning(f"No ASCII directory found in {quarter_dir} or its subdirectories")
        return None

    def process_quarter(self, quarter_dir: Path) -> Dict[str, pd.DataFrame]:
        """Process a single quarter directory."""
        quarter_dir = self._normalize_quarter_path(quarter_dir)
        ascii_dir = self._find_ascii_directory(quarter_dir)
        
        if not ascii_dir:
            raise ValueError(f"No ASCII directory found in {quarter_dir}")
            
        self.logger.info(f"Processing quarter directory: {quarter_dir}")
        self.logger.info(f"Using ASCII directory: {ascii_dir}")
        
        # Initialize results dictionary for all FAERS file types
        results = {
            'demo': pd.DataFrame(),      # Demographics
            'drug': pd.DataFrame(),      # Drug information
            'reac': pd.DataFrame(),      # Reactions
            'outc': pd.DataFrame(),      # Outcomes
            'rpsr': pd.DataFrame(),      # Report sources
            'ther': pd.DataFrame(),      # Therapy information
            'indi': pd.DataFrame()       # Indications
        }
        
        # Create quarter summary
        quarter_summary = QuarterSummary(quarter=quarter_dir.name)
        start_time = time.time()
        
        # Process each file type
        file_types = {
            'DEMO': ('demo', ['demo', 'demog']),
            'DRUG': ('drug', ['drug']),
            'REAC': ('reac', ['reac', 'reaction']),
            'OUTC': ('outc', ['outc', 'outcome']),
            'RPSR': ('rpsr', ['rpsr', 'source']),
            'THER': ('ther', ['ther', 'therapy']),
            'INDI': ('indi', ['indi', 'indic'])
        }
        
        for base_name, (data_type, patterns) in file_types.items():
            try:
                # Find matching files case-insensitively
                matched_files = []
                # Use rglob to find all .txt files (case insensitive)
                for file in ascii_dir.rglob('*.[tT][xX][tT]'):
                    file_name = file.name.lower()
                    if any(pat.lower() in file_name for pat in patterns):
                        matched_files.append(file)
                        self.logger.info(f"Found {data_type} file: {file}")
                        break  # Take the first matching file
                
                if not matched_files:
                    self.logger.warning(f"No {data_type} files found in {ascii_dir}")
                    continue
                    
                # Process the file
                file_path = matched_files[0]
                self.logger.info(f"Processing {data_type} file: {file_path}")
                
                table_summary = getattr(quarter_summary, f"{data_type}_summary")
                df = self.process_file(file_path, data_type, table_summary)
                
                if not df.empty:
                    results[data_type] = df
                    self.logger.info(f"Successfully processed {data_type} file. Shape: {df.shape}")
                else:
                    self.logger.warning(f"Empty DataFrame returned for {data_type}")
                    
            except Exception as e:
                error_msg = f"Error processing {data_type}: {str(e)}"
                self.logger.error(error_msg)
                table_summary = getattr(quarter_summary, f"{data_type}_summary")
                table_summary.add_parsing_error(error_msg)
        
        # Update quarter processing time and log summary
        quarter_summary.processing_time = time.time() - start_time
        quarter_summary.log_summary()
        
        # Add quarter summary to processing summary
        if not hasattr(self, 'processing_summary'):
            self.processing_summary = FAERSProcessingSummary()
        self.processing_summary.add_quarter_summary(quarter_dir.name, quarter_summary)
        
        return results

    def process_file(self, file_path: Path, data_type: str, table_summary: TableSummary) -> pd.DataFrame:
        """Process a single FAERS file with enhanced error tracking.
        
        NOTE: DO NOT MODIFY FILE FINDING LOGIC - case insensitive file matching is working as intended.
        """
        start_time = time.time()
        
        try:
            # Read the file with error handling like R's read.delim
            try:
                df = pd.read_csv(
                    file_path,
                    delimiter='$',
                    encoding='latin1',
                    on_bad_lines='skip',  # Skip bad lines like R does
                    low_memory=False,  # Avoid mixed type inference warnings
                    dtype=str,  # Read all columns as string initially like R
                    quoting=3,  # QUOTE_NONE like R's quote=""
                    skip_blank_lines=True  # Skip blank lines like R
                )
                
                if df.empty:
                    error_msg = f"Empty DataFrame after reading {file_path}"
                    self.logger.error(error_msg)
                    table_summary.add_parsing_error(error_msg)
                    return df
                    
                self.logger.info(f"Successfully read {len(df):,} rows from {file_path}")
                table_summary.total_rows = len(df)
                
                # Process DataFrame through common processing first
                df = self._process_dataframe(df, data_type, table_summary)
                if df.empty:
                    return df
                
            except pd.errors.ParserError as e:
                error_msg = f"Parser error in {file_path}: {str(e)}"
                self.logger.error(error_msg)
                table_summary.add_parsing_error(error_msg)
                return pd.DataFrame()
                
            except Exception as e:
                error_msg = f"Error reading {file_path}: {str(e)}"
                self.logger.error(error_msg)
                table_summary.add_parsing_error(error_msg)
                return pd.DataFrame()
            
            # Process based on data type
            try:
                if data_type == 'demo':
                    df = self.standardizer.standardize_demographics(df)
                elif data_type == 'drug':
                    df = self.standardizer.standardize_drugs(df)
                elif data_type == 'reac':
                    df = self.standardizer.standardize_reactions(df)
                elif data_type == 'outc':
                    df = self.standardizer.standardize_outcomes(df)
                elif data_type == 'rpsr':
                    df = self.standardizer.standardize_sources(df)
                elif data_type == 'ther':
                    df = self.standardizer.standardize_therapies(df)
                elif data_type == 'indi':
                    df = self.standardizer.standardize_indications(df)
                
                if df is None or df.empty:
                    error_msg = f"Empty DataFrame after standardizing {data_type}"
                    self.logger.error(error_msg)
                    table_summary.add_parsing_error(error_msg)
                    return pd.DataFrame()
                
                table_summary.processed_rows = len(df)
                table_summary.processing_time = time.time() - start_time
                
                self.logger.info(f"Successfully processed {data_type} file. Input rows: {table_summary.total_rows:,}, Output rows: {table_summary.processed_rows:,}")
                if table_summary.total_rows != table_summary.processed_rows:
                    self.logger.warning(f"Row count changed during processing. {table_summary.total_rows - table_summary.processed_rows:,} rows were filtered out")
                
                return df
                
            except Exception as e:
                error_msg = f"Error standardizing {data_type}: {str(e)}"
                self.logger.error(error_msg)
                table_summary.add_parsing_error(error_msg)
                return pd.DataFrame()
            
        except Exception as e:
            error_msg = f"Unexpected error processing {data_type}: {str(e)}"
            self.logger.error(error_msg)
            table_summary.add_parsing_error(error_msg)
            return pd.DataFrame()

    def _process_dataframe(self, df: pd.DataFrame, data_type: str, table_summary: TableSummary) -> pd.DataFrame:
        """Process a DataFrame based on its type."""
        if df.empty:
            return df

        try:
            # Common processing for all types
            df = df.fillna('')  # Replace NaN with empty string
            table_summary.total_rows = len(df)
            
            # Type-specific processing
            if data_type == 'demo':
                # Check required columns
                if 'i_f_code' not in df.columns:
                    table_summary.add_missing_column('i_f_code', 'I')
                    df['i_f_code'] = 'I'
                    self.logger.warning("Required column 'i_f_code' not found, adding with default value: I")
                    
                if 'sex' not in df.columns:
                    table_summary.add_missing_column('sex', '<NA>')
                    df['sex'] = '<NA>'
                    self.logger.warning("Required column 'sex' not found, adding with default value: <NA>")
                
                # Validate dates
                date_fields = ['event_dt', 'fda_dt', 'rept_dt']
                for field in date_fields:
                    if field in df.columns:
                        invalid_dates = df[~df[field].str.match(r'^\d{8}$', na=True)].shape[0]
                        if invalid_dates > 0:
                            table_summary.add_invalid_date(field, invalid_dates)
                            self.logger.warning(f"{invalid_dates}/{len(df)} rows ({invalid_dates/len(df)*100:.1f}%) had invalid dates in {field}")
                
                # Check country standardization
                if 'country' not in df.columns:
                    self.logger.warning("Country column not found, skipping country standardization")
            
            elif data_type == 'drug':
                # Check for required drug fields
                required_fields = ['drugname', 'prod_ai', 'route', 'dose_amt', 'dose_unit', 'dose_form']
                for field in required_fields:
                    if field not in df.columns:
                        table_summary.add_missing_column(field, '<NA>')
                        df[field] = '<NA>'
                        self.logger.warning(f"Required column '{field}' not found, adding with default value: <NA>")
                
            elif data_type == 'reac':
                if 'pt' not in df.columns:
                    table_summary.add_missing_column('pt', '<NA>')
                    df['pt'] = '<NA>'
                    self.logger.warning("Required column 'pt' not found, adding with default value: <NA>")
                    
            elif data_type == 'outc':
                if 'outc_cod' not in df.columns:
                    table_summary.add_missing_column('outc_cod', '<NA>')
                    df['outc_cod'] = '<NA>'
                    self.logger.warning("Required column 'outc_cod' not found, adding with default value: <NA>")
                    
            elif data_type == 'rpsr':
                if 'rpsr_cod' not in df.columns:
                    table_summary.add_missing_column('rpsr_cod', '<NA>')
                    df['rpsr_cod'] = '<NA>'
                    self.logger.warning("Required column 'rpsr_cod' not found, adding with default value: <NA>")
                    
            elif data_type == 'ther':
                # Check for required therapy fields
                required_fields = ['dsg_drug_seq', 'start_dt', 'end_dt']
                for field in required_fields:
                    if field not in df.columns:
                        table_summary.add_missing_column(field, '<NA>')
                        df[field] = '<NA>'
                        self.logger.warning(f"Required column '{field}' not found, adding with default value: <NA>")
                        
                # Validate dates
                date_fields = ['start_dt', 'end_dt']
                for field in date_fields:
                    if field in df.columns:
                        invalid_dates = df[~df[field].str.match(r'^\d{8}$', na=True)].shape[0]
                        if invalid_dates > 0:
                            table_summary.add_invalid_date(field, invalid_dates)
                            self.logger.warning(f"{invalid_dates}/{len(df)} rows ({invalid_dates/len(df)*100:.1f}%) had invalid dates in {field}")
                            
            elif data_type == 'indi':
                if 'indi_pt' not in df.columns:
                    table_summary.add_missing_column('indi_pt', '<NA>')
                    df['indi_pt'] = '<NA>'
                    self.logger.warning("Required column 'indi_pt' not found, adding with default value: <NA>")
            
            # Call standardizer for final processing
            df = self.standardizer.standardize_data(df, data_type)
            
        except Exception as e:
            error_msg = f"Error processing DataFrame: {str(e)}"
            table_summary.add_parsing_error(error_msg)
            self.logger.error(error_msg)
            return pd.DataFrame()
            
        return df

    def _fix_known_data_issues(self, file_path: Path) -> str:
        """Fix known data formatting issues in specific FAERS files.

        Args:
            file_path: Path to the file being processed

        Returns:
            Fixed content with proper line breaks
        """
        # Known problematic files and their fixes - adjusted line numbers to be 0-based
        known_issues = {
            "DRUG11Q2.TXT": {
                "line": 322966,
                "pattern": "$$$$$$7475791",
                "expected_fields": 13
            },
            "DRUG11Q3.TXT": {
                "line": 247895,
                "pattern": "$$$$$$7652730",
                "expected_fields": 13
            },
            "DRUG11Q4.TXT": {
                "line": 446737,
                "pattern": "021487$7941354",
                "expected_fields": 13
            },
            "DEMO12Q1.TXT": {
                "line": 105916,
                "pattern": None,
                "expected_fields": 24
            }
        }

        filename = os.path.basename(file_path).upper()
        if filename in known_issues:
            issue = known_issues[filename]
            lines = []
            with open(file_path, 'r') as f:
                for line in f:
                    lines.append(line.strip())
            # Check surrounding lines for the issue
            problem_line = issue["line"]
            for offset in [-1, 0, 1]:  # Check the line before, the line itself, and the line after
                check_line = problem_line + offset
                if 0 <= check_line < len(lines):
                    current_line = lines[check_line]
                    fields = current_line.split("$")

                    # If we have too many fields
                    if len(fields) > issue["expected_fields"]:
                        if issue["pattern"]:
                            # Try to split at the known pattern
                            if issue["pattern"] in current_line:
                                parts = current_line.split(issue["pattern"])
                                if len(parts) == 2:
                                    # Create two properly formatted lines
                                    first_line = parts[0] + "$" * (issue["expected_fields"] - 1)
                                    second_line = "$".join([""] * (issue["expected_fields"] - 1)) + parts[1]
                                    lines[check_line] = first_line
                                    lines.insert(check_line + 1, second_line)
                                    break
                        else:
                            # Generic handling for field count issues
                            new_lines = []
                            current_fields = []
                            for field in fields:
                                current_fields.append(field)
                                if len(current_fields) == issue["expected_fields"]:
                                    new_lines.append("$".join(current_fields))
                                    current_fields = []

                            if current_fields:  # Handle any remaining fields
                                while len(current_fields) < issue["expected_fields"]:
                                    current_fields.append("")
                                new_lines.append("$".join(current_fields))

                            # Replace the problematic line with fixed lines
                            lines[check_line:check_line+1] = new_lines
                            break

            # Rejoin the lines
            content = "\n".join(lines)

        return content

    def _normalize_quarter_path(self, quarter_dir: Path) -> Path:
        """Normalize quarter directory path to handle case sensitivity.

        Args:
            quarter_dir: Original quarter directory path

        Returns:
            Normalized path
        """
        # Extract year and quarter
        match = re.match(r'(\d{4})([qQ][1-4])', quarter_dir.name)
        if not match:
            return quarter_dir

        year, quarter = match.groups()
        normalized_name = f"{year}Q{quarter[-1]}"  # Convert to YYYYQ# format

        # Create new path with normalized name
        return quarter_dir.parent / normalized_name

    def unify_data(self, files_list: List[str], namekey: Dict[str, str], 
                   column_subset: List[str], duplicated_cols_x: List[str], 
                   duplicated_cols_y: List[str]) -> pd.DataFrame:
        """Exact implementation of R's unify_data function.
        
        Args:
            files_list: List of file paths to process
            namekey: Dictionary mapping original column names to new names
            column_subset: List of columns to keep
            duplicated_cols_x: List of target columns for duplicate handling
            duplicated_cols_y: List of source columns for duplicate handling
            
        Returns:
            DataFrame with unified data
        """
        result_df = None
        
        for i, f in enumerate(files_list):
            # Match R's gsub(".TXT", ".rds", f, ignore.case = TRUE)
            name = re.sub(r'\.TXT$', '.rds', f, flags=re.IGNORECASE)
            self.logger.info(f"Processing {name}")
            
            # Read RDS file
            try:
                x = pd.read_pickle(name)
            except Exception as e:
                self.logger.error(f"Error reading {name}: {e}")
                continue
                
            # Drop columns with NA names (x <- x[!is.na(names(x))])
            x = x.loc[:, x.columns.notna()]
            
            # Extract quarter (substr(name, nchar(name) - 7, nchar(name) - 4))
            quart = name[-8:-4]
            
            # Add quarter column (x <- setDT(x)[, quarter := quart])
            x['quarter'] = quart
            
            # Rename columns using namekey (names(x) <- namekey[names(x)])
            x = x.rename(columns=lambda col: namekey.get(col, col))
            
            # Combine DataFrames (rbindlist with fill=TRUE equivalent)
            if result_df is not None:
                result_df = pd.concat([result_df, x], ignore_index=True, sort=False)
            else:
                result_df = x
        
        if result_df is None:
            return pd.DataFrame()
            
        # Handle duplicated columns
        if duplicated_cols_x and duplicated_cols_y:
            for x_col, y_col in zip(duplicated_cols_x, duplicated_cols_y):
                if pd.isna(x_col) or pd.isna(y_col):
                    continue
                # Match R's: y[is.na(get(duplicated_cols_x[n])), (duplicated_cols_x[n]) := get(duplicated_cols_y[n])]
                result_df.loc[result_df[x_col].isna(), x_col] = result_df[y_col]
        
        # Get removed columns (setdiff(colnames(y),column_subset))
        removed_cols = set(result_df.columns) - set(column_subset)
        
        # Keep only subset columns (y <- y[, ..cols])
        result_df = result_df[column_subset]
        
        # Convert empty strings to NA (y[y == ""] <- NA)
        result_df = result_df.replace(r'^\s*$', pd.NA, regex=True)
        
        # Remove duplicates (y <- y %>% distinct())
        result_df = result_df.drop_duplicates()
        
        # Print removed columns message
        removed_cols_msg = "The following columns were lost in the cleaning: " + "; ".join(removed_cols)
        self.logger.info(removed_cols_msg)
        print(removed_cols_msg)
        
        return result_df

    def find_faers_files(self, input_dir: Path) -> List[Path]:
        """Exact match to R's list.files with pattern=".TXT"."""
        faers_list = []
        for file in input_dir.rglob('*.[tT][xX][tT]'):
            # Match R's: faers_list[!grepl("STAT|SIZE",faers_list)]
            if not re.search(r'STAT|SIZE', str(file), re.IGNORECASE):
                faers_list.append(file)
        return faers_list

    def get_project_paths(self) -> Dict[str, Path]:
        """Get standardized project paths matching exact project structure."""
        project_root = Path("/Users/davidandrews/Documents/Projects/DiAna")
        return {
            "raw": project_root / "data" / "raw",
            "clean": project_root / "data" / "clean",
            "external": project_root / "external_data",
            "dictionary": project_root / "external_data" / "DiAna_dictionary",
            "manual_fixes": project_root / "external_data" / "manual_fixes",
            "meddra": project_root / "external_data" / "meddra",
            "meddra_ascii": project_root / "external_data" / "meddra" / "MedAscii",
            "scripts": project_root / "src" / "faers_processor" / "services"  # Python equivalent of R-scripts
        }

    def save_faers_list(self, faers_list: List[Path]) -> None:
        """Save faers_list to CSV in the same format as R's write.csv2."""
        paths = self.get_project_paths()
        faers_df = pd.DataFrame({'x': [str(f) for f in faers_list]})
        
        # Create clean directory if it doesn't exist
        paths["clean"].mkdir(parents=True, exist_ok=True)
        
        # Save using semicolon separator like R's write.csv2
        output_path = paths["clean"] / "faers_list.csv"
        faers_df.to_csv(output_path, sep=';', index=False)
        self.logger.info(f"Saved faers_list to {output_path}")

    def process_drug_datasets(self) -> None:
        """Process DRUG datasets exactly as in the R implementation.
        
        Creates two datasets:
        1. DRUG: General information about drugs and suspect degree
        2. DRUG_INFO: Details about doses, formulations, dechallenge, and routes
        
        Both datasets maintain primary (primaryid) and secondary (drug_seq) keys.
        """
        paths = self.get_project_paths()
        self.logger.info("Processing DRUG datasets")
        
        # Verify directory structure
        if not paths["raw"].exists():
            raise ValueError(f"Raw data directory not found at {paths['raw']}")
        if not paths["clean"].exists():
            paths["clean"].mkdir(parents=True)
            
        # Find DRUG files (str_detect(faers_list, regex("drug", ignore_case = T)))
        drug_files = []
        for quarter_dir in paths["raw"].iterdir():
            if quarter_dir.is_dir():
                ascii_dir = quarter_dir / "ascii"
                if ascii_dir.exists():
                    for file in ascii_dir.glob("*.[tT][xX][tT]"):
                        if re.search(r'drug', file.name, re.IGNORECASE):
                            drug_files.append(file)
        
        if not drug_files:
            raise ValueError("No DRUG files found in the ascii directories")
            
        # Define common namekey mapping for both datasets
        namekey = {
            "ISR": "primaryid",
            "DRUG_SEQ": "drug_seq",
            "ROLE_COD": "role_cod",
            "DRUGNAME": "drugname",
            "VAL_VBM": "val_vbm",
            "ROUTE": "route",
            "DOSE_VBM": "dose_vbm",
            "DECHAL": "dechal",
            "RECHAL": "rechal",
            "LOT_NUM": "lot_num",
            "NDA_NUM": "nda_num",
            "EXP_DT": "exp_dt"
        }
        
        try:
            # Process DRUG dataset (general information)
            self.logger.info("Processing DRUG dataset")
            drug_df = self.unify_data(
                files_list=drug_files,
                namekey=namekey,
                column_subset=[
                    "primaryid", "drug_seq", "role_cod", "drugname", "prod_ai"
                ],
                duplicated_cols_x=None,  # NA in R
                duplicated_cols_y=None   # NA in R
            )
            
            # Save DRUG dataset
            drug_output = paths["clean"] / "DRUG.rds"
            drug_df.to_pickle(drug_output)
            self.logger.info(f"Saved DRUG dataset to {drug_output}")
            self.logger.info(f"DRUG shape: {drug_df.shape}")
            
            # Process DRUG_INFO dataset (detailed information)
            self.logger.info("Processing DRUG_INFO dataset")
            drug_info_df = self.unify_data(
                files_list=drug_files,
                namekey=namekey,
                column_subset=[
                    "primaryid", "drug_seq", "val_vbm", "nda_num", "lot_num",
                    "route", "dose_form", "dose_freq", "exp_dt",
                    "dose_vbm", "cum_dose_unit", "cum_dose_chr", "dose_amt",
                    "dose_unit", "dechal", "rechal"
                ],
                duplicated_cols_x=["lot_num"],
                duplicated_cols_y=["lot_nbr"]
            )
            
            # Save DRUG_INFO dataset
            drug_info_output = paths["clean"] / "DRUG_INFO.rds"
            drug_info_df.to_pickle(drug_info_output)
            self.logger.info(f"Saved DRUG_INFO dataset to {drug_info_output}")
            self.logger.info(f"DRUG_INFO shape: {drug_info_df.shape}")
            
            # Log summary of both datasets
            self.logger.info("DRUG processing complete:")
            self.logger.info(f"DRUG columns: {', '.join(drug_df.columns)}")
            self.logger.info(f"DRUG_INFO columns: {', '.join(drug_info_df.columns)}")
            
        except Exception as e:
            self.logger.error(f"Error processing DRUG datasets: {str(e)}")
            raise

    def process_demo_dataset(self, input_dir: Path, output_dir: Path) -> None:
        """Process DEMO dataset exactly as in the R implementation.
        
        Specific steps:
        1. Excludes IMAGE, CONFID, and DEATH_DT variables
        2. Derives sex from sex and gndr_cod
        3. Combines rept_dt and " rept_dt" for reporter date
        """
        self.logger.info("Processing DEMO dataset")
        
        # Find DEMO files (str_detect(faers_list, regex("demo", ignore_case = T)))
        demo_files = [f for f in self.find_faers_files(input_dir) 
                     if re.search(r'demo', str(f), re.IGNORECASE)]
        
        if not demo_files:
            raise ValueError("No DEMO files found")
            
        # Define exact column mappings as in R
        namekey = {
            "ISR": "primaryid",
            "CASE": "caseid",
            "FOLL_SEQ": "caseversion",
            "I_F_COD": "i_f_cod",
            "EVENT_DT": "event_dt",
            "MFR_DT": "mfr_dt",
            "FDA_DT": "fda_dt",
            "REPT_COD": "rept_cod",
            "MFR_NUM": "mfr_num",
            "MFR_SNDR": "mfr_sndr",
            "AGE": "age",
            "AGE_COD": "age_cod",
            "GNDR_COD": "sex",
            "E_SUB": "e_sub",
            "WT": "wt",
            "WT_COD": "wt_cod",
            "REPT_DT": "rept_dt",
            "OCCP_COD": "occp_cod",
            "TO_MFR": "to_mfr",
            "REPORTER_COUNTRY": "reporter_country",
            "quarter": "quarter",
            "i_f_code": "i_f_cod"
        }
        
        # Define exact column subset as in R
        column_subset = [
            "primaryid", "caseid", "caseversion", "i_f_cod", "sex", "age",
            "age_cod", "age_grp", "wt", "wt_cod", "reporter_country",
            "occr_country", "event_dt", "rept_dt", "mfr_dt", "init_fda_dt",
            "fda_dt", "rept_cod", "occp_cod", "mfr_num", "mfr_sndr", "to_mfr",
            "e_sub", "quarter", "auth_num", "lit_ref"
        ]
        
        # Define duplicated columns exactly as in R
        duplicated_cols_x = ["rept_dt", "sex"]
        duplicated_cols_y = [" rept_dt", "gndr_cod"]
        
        try:
            # Process DEMO files
            demo_df = self.unify_data(
                files_list=demo_files,
                namekey=namekey,
                column_subset=column_subset,
                duplicated_cols_x=duplicated_cols_x,
                duplicated_cols_y=duplicated_cols_y
            )
            
            # Explicitly verify excluded columns are not present
            excluded_cols = ["IMAGE", "CONFID", "DEATH_DT"]
            for col in excluded_cols:
                if col in demo_df.columns:
                    self.logger.warning(f"Excluded column {col} found in dataset - removing")
                    demo_df = demo_df.drop(columns=[col])
            
            # Save processed DEMO dataset
            output_path = output_dir / "DEMO.rds"
            demo_df.to_pickle(output_path)
            self.logger.info(f"Saved processed DEMO dataset to {output_path}")
            self.logger.info(f"DEMO shape: {demo_df.shape}")
            
            # Log summary statistics
            self.logger.info(f"DEMO columns: {', '.join(demo_df.columns)}")
            
        except Exception as e:
            self.logger.error(f"Error processing DEMO dataset: {str(e)}")
            raise

    def process_indi_dataset(self) -> None:
        """Process INDI dataset exactly as in the R implementation.
        
        Specific steps:
        1. Process INDI files
        2. Remove rows with no indication specified (NA indi_pt)
        3. Save to RDS/pickle format
        """
        paths = self.get_project_paths()
        self.logger.info("Processing INDI dataset")
        
        # Find INDI files (str_detect(faers_list, regex("indi", ignore_case = T)))
        indi_files = []
        for quarter_dir in paths["raw"].iterdir():
            if quarter_dir.is_dir():
                ascii_dir = quarter_dir / "ascii"
                if ascii_dir.exists():
                    for file in ascii_dir.glob("*.[tT][xX][tT]"):
                        if re.search(r'indi', file.name, re.IGNORECASE):
                            indi_files.append(file)
        
        if not indi_files:
            raise ValueError("No INDI files found in the ascii directories")
            
        try:
            # Process INDI dataset with exact R parameters
            indi_df = self.unify_data(
                files_list=indi_files,
                namekey={
                    "ISR": "primaryid",
                    "DRUG_SEQ": "drug_seq",
                    "indi_drug_seq": "drug_seq",
                    "INDI_PT": "indi_pt"
                },
                column_subset=[
                    "primaryid",
                    "drug_seq",
                    "indi_pt"
                ],
                duplicated_cols_x=None,  # NA in R
                duplicated_cols_y=None   # NA in R
            )
            
            # Remove rows with NA indi_pt (INDΙ <- INDΙ[!is.na(indi_pt)])
            indi_df = indi_df.dropna(subset=['indi_pt'])
            
            # Save processed INDI dataset
            output_path = paths["clean"] / "INDI.rds"
            indi_df.to_pickle(output_path)
            self.logger.info(f"Saved processed INDI dataset to {output_path}")
            self.logger.info(f"INDI shape: {indi_df.shape}")
            self.logger.info(f"INDI columns: {', '.join(indi_df.columns)}")
            
        except Exception as e:
            self.logger.error(f"Error processing INDI dataset: {str(e)}")
            raise

    def process_outc_dataset(self) -> None:
        """Process OUTC dataset exactly as in the R implementation.
        
        Specific steps:
        1. Process OUTC files
        2. Remove rows with no outcome specified (NA outc_cod)
        3. Save to RDS/pickle format
        """
        paths = self.get_project_paths()
        self.logger.info("Processing OUTC dataset")
        
        # Find OUTC files (str_detect(faers_list, regex("outc", ignore_case = T)))
        outc_files = []
        for quarter_dir in paths["raw"].iterdir():
            if quarter_dir.is_dir():
                ascii_dir = quarter_dir / "ascii"
                if ascii_dir.exists():
                    for file in ascii_dir.glob("*.[tT][xX][tT]"):
                        if re.search(r'outc', file.name, re.IGNORECASE):
                            outc_files.append(file)
        
        if not outc_files:
            raise ValueError("No OUTC files found in the ascii directories")
            
        try:
            # Process OUTC dataset with exact R parameters
            outc_df = self.unify_data(
                files_list=outc_files,
                namekey={
                    "ISR": "primaryid",
                    "OUTC_COD": "outc_cod"
                },
                column_subset=[
                    "primaryid",
                    "outc_cod"
                ],
                duplicated_cols_x=["outc_cod"],  # Matches R's c("outc_cod")
                duplicated_cols_y=["outc_code"]  # Matches R's c("outc_code")
            )
            
            # Remove rows with NA outc_cod (OUTC <- OUTC[!is.na(outc_cod)])
            outc_df = outc_df.dropna(subset=['outc_cod'])
            
            # Save processed OUTC dataset
            output_path = paths["clean"] / "OUTC.rds"
            outc_df.to_pickle(output_path)
            self.logger.info(f"Saved processed OUTC dataset to {output_path}")
            self.logger.info(f"OUTC shape: {outc_df.shape}")
            self.logger.info(f"OUTC columns: {', '.join(outc_df.columns)}")
            
        except Exception as e:
            self.logger.error(f"Error processing OUTC dataset: {str(e)}")
            raise

    def process_reac_dataset(self) -> None:
        """Process REAC dataset exactly as in the R implementation.
        
        Specific steps:
        1. Process REAC files
        2. Remove rows with no reaction specified (NA pt)
        3. Save to RDS/pickle format
        """
        paths = self.get_project_paths()
        self.logger.info("Processing REAC dataset")
        
        # Find REAC files (str_detect(faers_list, regex("reac", ignore_case = T)))
        reac_files = []
        for quarter_dir in paths["raw"].iterdir():
            if quarter_dir.is_dir():
                ascii_dir = quarter_dir / "ascii"
                if ascii_dir.exists():
                    for file in ascii_dir.glob("*.[tT][xX][tT]"):
                        if re.search(r'reac', file.name, re.IGNORECASE):
                            reac_files.append(file)
        
        if not reac_files:
            raise ValueError("No REAC files found in the ascii directories")
            
        try:
            # Process REAC dataset with exact R parameters
            reac_df = self.unify_data(
                files_list=reac_files,
                namekey={
                    "ISR": "primaryid",
                    "PT": "pt"
                },
                column_subset=[
                    "primaryid",
                    "pt",
                    "drug_rec_act"
                ],
                duplicated_cols_x=None,  # NA in R
                duplicated_cols_y=None   # NA in R
            )
            
            # Remove rows with NA pt (REAC <- REAC[!is.na(pt)])
            reac_df = reac_df.dropna(subset=['pt'])
            
            # Save processed REAC dataset
            output_path = paths["clean"] / "REAC.rds"
            reac_df.to_pickle(output_path)
            self.logger.info(f"Saved processed REAC dataset to {output_path}")
            self.logger.info(f"REAC shape: {reac_df.shape}")
            self.logger.info(f"REAC columns: {', '.join(reac_df.columns)}")
            
        except Exception as e:
            self.logger.error(f"Error processing REAC dataset: {str(e)}")
            raise

    def process_rpsr_dataset(self) -> None:
        """Process RPSR dataset exactly as in the R implementation.
        
        Specific steps:
        1. Process RPSR files
        2. Save to RDS/pickle format without any filtering
        """
        paths = self.get_project_paths()
        self.logger.info("Processing RPSR dataset")
        
        # Find RPSR files (str_detect(faers_list, regex("rpsr", ignore_case = T)))
        rpsr_files = []
        for quarter_dir in paths["raw"].iterdir():
            if quarter_dir.is_dir():
                ascii_dir = quarter_dir / "ascii"
                if ascii_dir.exists():
                    for file in ascii_dir.glob("*.[tT][xX][tT]"):
                        if re.search(r'rpsr', file.name, re.IGNORECASE):
                            rpsr_files.append(file)
        
        if not rpsr_files:
            raise ValueError("No RPSR files found in the ascii directories")
            
        try:
            # Process RPSR dataset with exact R parameters
            rpsr_df = self.unify_data(
                files_list=rpsr_files,
                namekey={
                    "ISR": "primaryid",
                    "RPSR_COD": "rpsr_cod"
                },
                column_subset=[
                    "primaryid",
                    "rpsr_cod"
                ],
                duplicated_cols_x=None,  # NA in R
                duplicated_cols_y=None   # NA in R
            )
            
            # Save processed RPSR dataset (no filtering needed)
            output_path = paths["clean"] / "RPSR.rds"
            rpsr_df.to_pickle(output_path)
            self.logger.info(f"Saved processed RPSR dataset to {output_path}")
            self.logger.info(f"RPSR shape: {rpsr_df.shape}")
            self.logger.info(f"RPSR columns: {', '.join(rpsr_df.columns)}")
            
        except Exception as e:
            self.logger.error(f"Error processing RPSR dataset: {str(e)}")
            raise

    def process_ther_dataset(self) -> None:
        """Process THER dataset exactly as in the R implementation.
        
        Specific steps:
        1. Process THER files
        2. Save to RDS/pickle format without any filtering
        """
        paths = self.get_project_paths()
        self.logger.info("Processing THER dataset")
        
        # Find THER files (str_detect(faers_list, regex("ther", ignore_case = T)))
        ther_files = []
        for quarter_dir in paths["raw"].iterdir():
            if quarter_dir.is_dir():
                ascii_dir = quarter_dir / "ascii"
                if ascii_dir.exists():
                    for file in ascii_dir.glob("*.[tT][xX][tT]"):
                        if re.search(r'ther', file.name, re.IGNORECASE):
                            ther_files.append(file)
        
        if not ther_files:
            raise ValueError("No THER files found in the ascii directories")
            
        try:
            # Process THER dataset with exact R parameters
            ther_df = self.unify_data(
                files_list=ther_files,
                namekey={
                    "ISR": "primaryid",
                    "dsg_drug_seq": "drug_seq",
                    "DRUG_SEQ": "drug_seq",
                    "START_DT": "start_dt",
                    "END_DT": "end_dt",
                    "DUR": "dur",
                    "DUR_COD": "dur_cod"
                },
                column_subset=[
                    "primaryid",
                    "drug_seq",
                    "start_dt",
                    "end_dt",
                    "dur",
                    "dur_cod"
                ],
                duplicated_cols_x=None,  # NA in R
                duplicated_cols_y=None   # NA in R
            )
            
            # Save processed THER dataset (no filtering needed)
            output_path = paths["clean"] / "THER.rds"
            ther_df.to_pickle(output_path)
            self.logger.info(f"Saved processed THER dataset to {output_path}")
            self.logger.info(f"THER shape: {ther_df.shape}")
            self.logger.info(f"THER columns: {', '.join(ther_df.columns)}")
            
        except Exception as e:
            self.logger.error(f"Error processing THER dataset: {str(e)}")
            raise

    def standardize_meddra_terms(self) -> None:
        """Standardize MedDRA terms exactly as in R implementation.
        
        Standardizes:
        1. REAC.rds: 'pt' and 'drug_rec_act' fields
        2. INDI.rds: 'indi_pt' field
        """
        paths = self.get_project_paths()
        self.logger.info("Starting MedDRA standardization")
        
        try:
            standardizer = MedDRAStandardizer(paths["external"])
            
            # Standardize REAC dataset
            reac_file = paths["clean"] / "REAC.rds"
            reac_df = standardizer.standardize_pt(reac_file, "pt")
            reac_df = standardizer.standardize_pt(reac_file, "drug_rec_act")
            reac_df.to_pickle(reac_file)
            self.logger.info("Completed REAC standardization")
            
            # Standardize INDI dataset
            indi_file = paths["clean"] / "INDI.rds"
            indi_df = standardizer.standardize_pt(indi_file, "indi_pt")
            indi_df.to_pickle(indi_file)
            self.logger.info("Completed INDI standardization")
            
        except Exception as e:
            self.logger.error(f"Error in MedDRA standardization: {str(e)}")
            raise

    def correct_problematic_file(self, file_path: Path, old_line: str) -> None:
        """Exact match to R's correct_problematic_file function.
        
        Args:
            file_path: Path to file to correct
            old_line: Problematic line pattern to fix
        """
        self.logger.info(f"Correcting problematic file: {file_path}")
        try:
            with open(file_path, 'r', encoding='latin1') as f:
                lines = f.readlines()
            
            # Match R's gsub and strsplit logic exactly
            fixed_lines = []
            for line in lines:
                if old_line in line:
                    # Replace old_line with separator version
                    new_line = re.sub(r'([0-9]+)$', r'SePaRaToR\1', old_line)
                    line = line.replace(old_line, new_line)
                    # Split on separator
                    fixed_lines.extend(line.split('SePaRaToR'))
                else:
                    fixed_lines.append(line)
            
            with open(file_path, 'w', encoding='latin1') as f:
                f.writelines(fixed_lines)
                
            self.logger.info(f"Successfully corrected {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error correcting {file_path}: {str(e)}")
            raise

    def store_to_rds(self, file_path: Path, output_dir: Path) -> None:
        """Exact match to R's store_to_rds function with enhanced error handling.
        
        Args:
            file_path: Path to input TXT file
            output_dir: Directory to save RDS file
        """
        try:
            # Create output name (gsub(".TXT",".rds",f, ignore.case = T))
            rds_name = re.sub(r'\.TXT$', '.rds', str(file_path), flags=re.IGNORECASE)
            rds_path = output_dir / Path(rds_name).name
            
            self.logger.info(f"Converting {file_path} to {rds_path}")
            
            # Read header (readLines(file(f),n=1))
            with open(file_path, 'r', encoding='latin1') as f:
                header = f.readline().strip()
            column_names = header.split("$")
            
            # Read data (read.table(f,skip=1,sep="$", comment.char = "",quote=""))
            df = pd.read_csv(
                file_path, 
                sep='$',
                skiprows=1,
                names=column_names,
                quoting=3,  # QUOTE_NONE
                comment=None,
                encoding='latin1',
                dtype=str,
                on_bad_lines='warn'
            )
            
            # Save as pickle (equivalent to saveRDS)
            df.to_pickle(rds_path)
            self.logger.info(f"Successfully saved {rds_path}")
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            raise

    def preprocess_faers_files(self, input_dir: Path, output_dir: Path) -> None:
        """Preprocess FAERS files exactly like the R implementation.
        
        This combines all the R preprocessing steps while maintaining our enhanced features.
        """
        self.logger.info("Starting FAERS preprocessing")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all FAERS files
        faers_list = self.find_faers_files(input_dir)
        if not faers_list:
            raise ValueError(f"No FAERS files found in {input_dir}")
            
        # Save faers_list
        faers_df = pd.DataFrame({'x': [str(f) for f in faers_list]})
        
        # Create clean directory if it doesn't exist
        paths = self.get_project_paths()
        paths["clean"].mkdir(parents=True, exist_ok=True)
        
        # Save using semicolon separator like R's write.csv2
        output_path = paths["clean"] / "faers_list.csv"
        faers_df.to_csv(output_path, sep=';', index=False)
        self.logger.info(f"Saved faers_list to {output_path}")

        # Correct known problematic files
        problem_files = {
            "Raw_FAERS_QD//aers_ascii_2011q2/ascii/DRUG11Q2.txt": "\\$\\$\\$\\$\\$\\$7475791",
            "Raw_FAERS_QD//aers_ascii_2011q3/ascii/DRUG11Q3.txt": "\\$\\$\\$\\$\\$\\$7652730",
            "Raw_FAERS_QD//aers_ascii_2011q4/ascii/DRUG11Q4.txt": "021487\\$7941354"
        }
        
        for file_pattern, old_line in problem_files.items():
            # Find matching file in our directory structure
            pattern = Path(file_pattern).name
            matching_files = list(input_dir.rglob(pattern))
            if matching_files:
                self.correct_problematic_file(matching_files[0], old_line)
        
        # Process all files to RDS format with progress bar
        with tqdm(total=len(faers_list), desc="Converting to RDS", unit="file") as pbar:
            for file_path in faers_list:
                try:
                    self.store_to_rds(file_path, output_dir)
                    pbar.update(1)
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path}: {str(e)}")
                    continue
