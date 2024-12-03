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
from .deduplicator import FAERSDeduplicator


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

    def __init__(self, standardizer: DataStandardizer):
        """Initialize the FAERS processor."""
        self.standardizer = standardizer
        self.deduplicator = FAERSDeduplicator()
        self.logger = logging.getLogger(__name__)
        # Set current date in YYYYMMDD format
        self.current_date = datetime.now().strftime("%Y%m%d")

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
        """Find the ASCII directory - case insensitive search."""
        self.logger.info(f"Searching for ASCII directory in: {quarter_dir}")
        
        if not quarter_dir.exists():
            self.logger.error(f"Quarter directory does not exist: {quarter_dir}")
            return None

        # First check if the provided path itself is the ASCII directory
        if quarter_dir.name.lower() == 'ascii':
            self.logger.info(f"Input path is the ASCII directory: {quarter_dir}")
            return quarter_dir

        # Check immediate children
        self.logger.info("Checking immediate children for ASCII directory")
        for item in quarter_dir.iterdir():
            if item.is_dir():
                self.logger.debug(f"Checking directory: {item.name}")
                if item.name.lower() == 'ascii':
                    self.logger.info(f"Found ASCII directory: {item}")
                    return item

        # If not found, check one level deeper
        self.logger.info("Checking one level deeper for ASCII directory")
        for item in quarter_dir.iterdir():
            if item.is_dir():
                if item.name.lower() == 'ascii':
                    self.logger.info(f"Found ASCII directory: {item}")
                    return item
                self.logger.debug(f"Checking subdirectories of: {item.name}")
                for subitem in item.iterdir():
                    if subitem.is_dir() and subitem.name.lower() == 'ascii':
                        self.logger.info(f"Found ASCII directory: {subitem}")
                        return subitem

        self.logger.error(f"No ASCII directory found in {quarter_dir} or its subdirectories")
        self.logger.error(f"Available directories at {quarter_dir}:")
        for item in quarter_dir.iterdir():
            if item.is_dir():
                self.logger.error(f"  - {item.name}")
        return None

    def _find_data_file(self, directory: Path, patterns: List[str]) -> Optional[Path]:
        """Find a data file matching any of the patterns case-insensitively."""
        if not directory.exists():
            return None
            
        # Convert patterns to lowercase for comparison
        patterns = [p.lower() for p in patterns]
        
        # Search for .txt files case-insensitively
        for file in directory.glob('*'):
            if file.is_file() and file.suffix.lower() == '.txt':
                file_name = file.stem.lower()
                if any(pattern in file_name for pattern in patterns):
                    return file
        
        return None

    def process_quarter(self, quarter_dir: Path, parallel: bool = True, max_workers: Optional[int] = None, chunk_size: int = 50000) -> Dict[str, pd.DataFrame]:
        """Process a single quarter directory."""
        quarter_dir = self._normalize_quarter_path(quarter_dir)
        ascii_dir = self._find_ascii_directory(quarter_dir)
        quarter_name = quarter_dir.name
        
        if not ascii_dir:
            self.logger.error(f"({quarter_name}) No ASCII directory found in {quarter_dir}")
            raise ValueError(f"No ASCII directory found in {quarter_dir}")
            
        self.logger.info(f"\nProcessing quarter: {quarter_name}")
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
        quarter_summary = QuarterSummary(quarter=quarter_name)
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
                # Find matching file case-insensitively
                file_path = self._find_data_file(ascii_dir, patterns)
                
                if not file_path:
                    self.logger.warning(f"({quarter_name}) No {data_type} files found in {ascii_dir}")
                    continue
                    
                # Process the file
                self.logger.info(f"({quarter_name}) Processing {data_type} file: {file_path}")
                
                table_summary = getattr(quarter_summary, f"{data_type}_summary")
                df = self.process_file(file_path, data_type, quarter_name)
                
                if not df.empty:
                    results[data_type] = df
                    self.logger.info(f"({quarter_name}) Successfully processed {data_type} file. Shape: {df.shape}")
                else:
                    self.logger.warning(f"({quarter_name}) Empty DataFrame returned for {data_type}")
                    
            except Exception as e:
                error_msg = f"({quarter_name}) Error processing {data_type}: {str(e)}"
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

    def _process_common(self, df: pd.DataFrame, data_type: str, quarter_name: str, file_name: str) -> pd.DataFrame:
        """Common processing steps for all data types."""
        try:
            # Add missing columns with appropriate defaults
            if data_type == 'indi':
                if 'indi_pt' not in df.columns:
                    self.logger.warning(f"({quarter_name}) Required column 'indi_pt' not found in {file_name}, adding with default value: <NA>")
                    df['indi_pt'] = pd.NA
            elif data_type == 'demo':
                if 'i_f_code' not in df.columns:
                    self.logger.warning(f"({quarter_name}) Required column 'i_f_code' not found in {file_name}, adding with default value: I")
                    df['i_f_code'] = 'I'
                if 'sex' not in df.columns:
                    self.logger.warning(f"({quarter_name}) Required column 'sex' not found in {file_name}, adding with default value: <NA>")
                    df['sex'] = pd.NA
                if 'country' not in df.columns:
                    self.logger.warning(f"({quarter_name}) Country column not found in {file_name}, skipping country standardization")
            elif data_type == 'drug':
                if 'drugname' not in df.columns:
                    self.logger.warning(f"({quarter_name}) Required column 'drugname' not found in {file_name}, adding with default value: <NA>")
                    df['drugname'] = pd.NA
                if 'prod_ai' not in df.columns:
                    self.logger.warning(f"({quarter_name}) Required column 'prod_ai' not found in {file_name}, adding with default value: <NA>")
                    df['prod_ai'] = pd.NA
                if 'route' not in df.columns:
                    self.logger.warning(f"({quarter_name}) Required column 'route' not found in {file_name}, adding with default value: <NA>")
                    df['route'] = pd.NA
            
            return df
            
        except Exception as e:
            self.logger.error(f"({quarter_name}) Error processing {file_name}: {str(e)}")
            return df

    def process_file(self, file_path: Path, data_type: str, quarter_name: str) -> Optional[pd.DataFrame]:
        """Process a single FAERS data file."""
        try:
            # Read and clean the file
            df = pd.read_csv(file_path, delimiter='$', dtype=str)
            file_name = file_path.name
            
            if df is None:
                self.logger.error(f"({quarter_name}) Failed to read file: {file_name}")
                return None
                
            if df.empty:
                self.logger.error(f"({quarter_name}) File is empty: {file_name}")
                return None
            
            # Common processing steps
            df = self._process_common(df, data_type, quarter_name, file_name)
            
            # Standardize based on data type
            try:
                if data_type == 'demo':
                    df = self.standardizer.standardize_demographics(df)
                elif data_type == 'drug':
                    df = self.standardizer.standardize_drug_info(df)
                elif data_type == 'reac':
                    df = self.standardizer.standardize_reactions(df)
                elif data_type == 'indi':
                    df = self.standardizer.standardize_indications(df)
                else:
                    self.logger.warning(f"({quarter_name}) Unknown data type: {data_type} for file {file_name}")
            except Exception as e:
                self.logger.error(f"({quarter_name}) Error standardizing {data_type} data in {file_name}: {str(e)}")
                return None
            
            if df is not None and not df.empty:
                self.logger.info(f"({quarter_name}) Successfully processed {len(df):,} rows from {file_name}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"({quarter_name}) Error processing {file_path.name}: {str(e)}")
            return None

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
            # Match R's gsub(".TXT",".rds",f, ignore.case = TRUE)
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
            
        try:
            # Process DRUG dataset (general information)
            self.logger.info("Processing DRUG dataset")
            drug_df = self.unify_data(
                files_list=drug_files,
                namekey={
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
                },
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
                namekey={
                    "ISR": "primaryid",
                    "DRUG_SEQ": "drug_seq",
                    "VAL_VBM": "val_vbm",
                    "NDA_NUM": "nda_num",
                    "LOT_NUM": "lot_num",
                    "ROUTE": "route",
                    "DOSE_FORM": "dose_form",
                    "DOSE_FREQ": "dose_freq",
                    "EXP_DT": "exp_dt",
                    "DOSE_VBM": "dose_vbm",
                    "CUM_DOSE_UNIT": "cum_dose_unit",
                    "CUM_DOSE_CHR": "cum_dose_chr",
                    "DOSE_AMT": "dose_amt",
                    "DOSE_UNIT": "dose_unit",
                    "DECHAL": "dechal",
                    "RECHAL": "rechal"
                },
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
        """Process DEMO dataset exactly as in the R implementation."""
        try:
            # Load necessary datasets
            drug_df = pd.read_pickle(output_dir / 'DRUG.rds')
            reac_df = pd.read_pickle(output_dir / 'REAC.rds')
            outc_df = pd.read_pickle(output_dir / 'OUTC.rds')
            indi_df = pd.read_pickle(output_dir / 'INDI.rds')
            ther_df = pd.read_pickle(output_dir / 'THER.rds')
            drug_info_df = pd.read_pickle(output_dir / 'Drug_info.rds')
            rpsr_df = pd.read_pickle(output_dir / 'RPSR.rds')
            
            demo_files = list(input_dir.glob("*DEMO*.txt"))
            if not demo_files:
                raise FileNotFoundError("No DEMO files found in input directory")
                
            # Read and combine all demo files
            dfs = []
            for file in demo_files:
                df = pd.read_csv(file, delimiter="$", dtype=str)
                quarter = re.search(r'DEMO(\d{2}Q\d)', file.name)
                if quarter:
                    df['quarter'] = quarter.group(1)
                dfs.append(df)
                
            demo_df = pd.concat(dfs, ignore_index=True)
            
            # Process DEMO dataset
            demo_df = self.standardizer.standardize_demo(demo_df)
            demo_df = self.deduplicator.deduplicate_primaryids(demo_df)
            demo_df = self.deduplicator.deduplicate_by_caseid(demo_df)
            demo_df = self.deduplicator.deduplicate_by_manufacturer(demo_df)
            demo_df = self.remove_incomplete_reports(demo_df, drug_df, reac_df)
            demo_df = self.identify_premarketing_cases(demo_df, drug_df)
            demo_df = self.deduplicator.identify_rule_based_duplicates(demo_df, reac_df, drug_df)
            demo_df = self.finalize_demo_dataset(demo_df)
            
            # Split and save all datasets
            final_output_dir = output_dir / self._get_quarter_dir()
            self.split_and_save_datasets(
                demo_df, drug_df, reac_df, outc_df, indi_df, 
                ther_df, drug_info_df, rpsr_df, final_output_dir
            )
            
            self.logger.info("Successfully processed and saved all datasets")
            
        except Exception as e:
            self.logger.error(f"Error processing DEMO dataset: {str(e)}")
            raise
            
    def _get_quarter_dir(self) -> str:
        """Get the current quarter directory name (e.g., '23Q1')."""
        current_date = datetime.now()
        year = str(current_date.year)[-2:]  # Get last 2 digits
        quarter = (current_date.month - 1) // 3 + 1
        return f"{year}Q{quarter}"
            
    def split_and_save_datasets(self, demo_df: pd.DataFrame, drug_df: pd.DataFrame, 
                               reac_df: pd.DataFrame, outc_df: pd.DataFrame,
                               indi_df: pd.DataFrame, ther_df: pd.DataFrame,
                               drug_info_df: pd.DataFrame, rpsr_df: pd.DataFrame,
                               output_dir: Path) -> None:
        """Split and save datasets according to the R implementation.
        
        Matches R implementation of dataset splitting and saving, including:
        - Filtering out excluded primaryids
        - Splitting datasets into more manageable pieces
        - Converting categorical columns to ordered factors
        - Saving each dataset in the specified directory
        
        Args:
            demo_df: Demographics DataFrame
            drug_df: Drug DataFrame
            reac_df: Reaction DataFrame
            outc_df: Outcome DataFrame
            indi_df: Indication DataFrame
            ther_df: Therapy DataFrame
            drug_info_df: Drug Info DataFrame
            rpsr_df: RPSR DataFrame
            output_dir: Output directory path
        """
        try:
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Split DEMO into DEMO and DEMO_SUPP
            demo_supp = demo_df[['primaryid', 'caseid', 'caseversion', 'i_f_cod', 
                                'auth_num', 'e_sub', 'lit_ref', 'rept_dt', 'to_mfr',
                                'mfr_sndr', 'mfr_num', 'mfr_dt', 'quarter']].copy()
            
            demo_df = demo_df[['primaryid', 'sex', 'age_in_days', 'wt_in_kgs',
                              'occr_country', 'event_dt', 'occp_cod', 'reporter_country',
                              'rept_cod', 'init_fda_dt', 'fda_dt', 'premarketing',
                              'literature']].copy()
            
            # 2. Filter and split DRUG data
            valid_primaryids = set(demo_df['primaryid'])
            drug_df = drug_df[drug_df['primaryid'].isin(valid_primaryids)]
            
            # Create DRUG_NAME
            drug_name = drug_df[['primaryid', 'drug_seq', 'drugname', 'prod_ai']].drop_duplicates()
            
            # Update DRUG
            drug_df = drug_df[['primaryid', 'drug_seq', 'substance', 'role_cod']].drop_duplicates()
            drug_df['role_cod'] = pd.Categorical(drug_df['role_cod'], 
                                               categories=['C', 'I', 'SS', 'PS'],
                                               ordered=True)
            
            # 3. Filter and process REAC data
            reac_df = reac_df[reac_df['primaryid'].isin(valid_primaryids)]
            
            # Load MedDRA dictionary
            meddra_primary = pd.read_csv(self.config.external_data_dir / 'Dictionaries/MedDRA/meddra_primary.csv', 
                                       sep=';')
            meddra_primary = meddra_primary.sort_values(['soc', 'hlgt', 'hlt', 'pt'])
            
            # Convert pt and drug_rec_act to ordered factors
            reac_df['pt'] = pd.Categorical(reac_df['pt'], 
                                         categories=meddra_primary['pt'].tolist(),
                                         ordered=True)
            reac_df['drug_rec_act'] = pd.Categorical(reac_df['drug_rec_act'],
                                                    categories=meddra_primary['pt'].tolist(),
                                                    ordered=True)
            reac_df = reac_df[['primaryid', 'pt', 'drug_rec_act']]
            
            # 4. Filter and process OUTC data
            outc_df = outc_df[outc_df['primaryid'].isin(valid_primaryids)]
            outc_df['outc_cod'] = pd.Categorical(outc_df['outc_cod'],
                                               categories=['OT', 'CA', 'HO', 'RI', 
                                                         'DS', 'LT', 'DE'],
                                               ordered=True)
            outc_df = outc_df[['primaryid', 'outc_cod']].drop_duplicates()
            
            # 5. Filter and process INDI data
            indi_df = indi_df[indi_df['primaryid'].isin(valid_primaryids)]
            indi_df['indi_pt'] = pd.Categorical(indi_df['indi_pt'],
                                              categories=meddra_primary['pt'].tolist(),
                                              ordered=True)
            indi_df = indi_df[['primaryid', 'drug_seq', 'indi_pt']].drop_duplicates()
            
            # 6. Filter and process THER data
            ther_df = ther_df[ther_df['primaryid'].isin(valid_primaryids)]
            ther_df = ther_df[['primaryid', 'drug_seq', 'start_dt', 'dur_in_days',
                              'end_dt', 'time_to_onset', 'event_dt']].drop_duplicates()
            
            # 7. Filter and split DRUG_INFO data
            drug_info_df = drug_info_df[drug_info_df['primaryid'].isin(valid_primaryids)]
            
            # Create DOSES
            doses = drug_info_df[['primaryid', 'drug_seq', 'dose_vbm', 'cum_dose_unit',
                                'cum_dose_chr', 'dose_amt', 'dose_unit', 
                                'dose_freq']].drop_duplicates()
            
            # Create DRUG_SUPP
            drug_supp = drug_info_df[['primaryid', 'drug_seq', 'route', 'dose_form',
                                    'dechal', 'rechal', 'lot_num', 'exp_dt']].drop_duplicates()
            drug_supp['dose_form'] = pd.Categorical(drug_supp['dose_form'])
            
            # Update DRUG_NAME with additional info
            drug_name = drug_info_df[['primaryid', 'drug_seq', 'val_vbm', 
                                    'nda_num']].merge(drug_name, 
                                                    on=['primaryid', 'drug_seq'])
            
            # 8. Add RPSR info to DEMO_SUPP
            rpsr_df['rpsr_cod'] = pd.Categorical(rpsr_df['rpsr_cod'])
            demo_supp = rpsr_df[['primaryid', 'rpsr_cod']].merge(demo_supp, 
                                                                on='primaryid')
            
            # Save all datasets
            datasets = {
                'DEMO.rds': demo_df,
                'DEMO_SUPP.rds': demo_supp,
                'DRUG.rds': drug_df,
                'DRUG_NAME.rds': drug_name,
                'REAC.rds': reac_df,
                'OUTC.rds': outc_df,
                'INDI.rds': indi_df,
                'THER.rds': ther_df,
                'DOSES.rds': doses,
                'DRUG_SUPP.rds': drug_supp
            }
            
            for filename, df in datasets.items():
                output_path = output_dir / filename
                df.to_pickle(output_path)
                self.logger.info(f"Saved {filename} with shape {df.shape}")
            
        except Exception as e:
            self.logger.error(f"Error splitting and saving datasets: {str(e)}")
            raise
            
    def finalize_demo_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Finalize demo dataset processing.
        
        Matches R implementation:
        cols <- c("caseversion","sex","quarter","i_f_cod","rept_cod",
                "occp_cod","e_sub","age_grp","occr_country",
                "reporter_country")
        Demo[,(cols):=lapply(.SD, as.factor),.SDcols=cols]
        
        Args:
            df: DataFrame to finalize
            
        Returns:
            Finalized DataFrame with categorical columns
        """
        try:
            # Define columns to convert to categorical
            categorical_cols = [
                "caseversion", "sex", "quarter", "i_f_cod", "rept_cod",
                "occp_cod", "e_sub", "age_grp", "occr_country",
                "reporter_country"
            ]
            
            # Convert columns to categorical if they exist
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype('category')
                else:
                    self.logger.warning(f"Column {col} not found in dataset")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error finalizing demo dataset: {str(e)}")
            return df

    def process_indi_dataset(self) -> None:
        """Process INDI dataset exactly as in the R implementation."""
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
        """Process OUTC dataset exactly as in the R implementation."""
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
        """Process REAC dataset exactly as in the R implementation."""
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
        """Process RPSR dataset exactly as in the R implementation."""
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
        """Process THER dataset exactly as in the R implementation."""
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
        """Standardize MedDRA terms exactly as in R implementation."""
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

    def standardize_drugs(self) -> None:
        """Standardize drug names using DIANA dictionary exactly as in R."""
        paths = self.get_project_paths()
        self.logger.info("Starting drug standardization")
        
        try:
            standardizer = DrugStandardizer(paths["external"])
            drug_file = paths["clean"] / "DRUG.rds"
            standardizer.standardize_drugs(drug_file)
            self.logger.info("Completed drug standardization")
            
        except Exception as e:
            self.logger.error(f"Error in drug standardization: {str(e)}")
            raise

    def correct_problematic_file(self, file_path: Path, old_line: str) -> None:
        """Exact match to R's correct_problematic_file function."""
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
        """Exact match to R's store_to_rds function with enhanced error handling."""
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
        """Preprocess FAERS files exactly like the R implementation."""
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

    def process_files(self, input_dir: Path, output_dir: Path) -> None:
        """Process all FAERS data files in the input directory."""
        try:
            # Process demo files first to get event dates
            demo_files = list(input_dir.glob("*DEMO*.txt"))
            if not demo_files:
                raise FileNotFoundError("No demographics files found")
                
            demo_df = pd.concat([
                self.standardizer.standardize_demographics(
                    pd.read_csv(f, sep='$', dtype=str),
                    self.current_date
                )
                for f in demo_files
            ])
            
            # Process therapy files and calculate time to onset
            ther_files = list(input_dir.glob('*THER*.txt'))
            if ther_files:
                ther_df = pd.concat([
                    self.standardizer.standardize_therapies(
                        pd.read_csv(f, sep='$', dtype=str),
                        self.current_date
                    )
                    for f in ther_files
                ])
                
                # Calculate time to onset
                ther_df = self.standardizer.calculate_time_to_onset(demo_df, ther_df)
                
                # Save processed therapy data
                ther_df.to_csv(output_dir / 'THER.csv', index=False)
                
            # Save processed demographics data
            demo_df.to_csv(output_dir / 'DEMO.csv', index=False)
            
            # Process other file types...
            
        except Exception as e:
            logging.error(f"Error processing files: {str(e)}")
            raise

    def process_drug_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process drug information data.
        
        Args:
            df: Raw drug information DataFrame
        
        Returns:
            Processed DataFrame with standardized therapy regimen
        """
        df = self.standardizer.standardize_drug_info(df, self.current_date)
        return df

    def remove_incomplete_reports(self, demo_df: pd.DataFrame, drug_df: pd.DataFrame, 
                                reac_df: pd.DataFrame) -> pd.DataFrame:
        """Remove DEMO cases missing either drug or reaction information.
        
        Matches R implementation:
        Drug <- Drug[!Substance%in%c("no medication","unspecified")]
        Reac <- Reac[!pt%in%c("no adverse event")]
        no_drugs <- setdiff(unique(Demo$primaryid),unique(Drug$primaryid))
        no_event <- setdiff(unique(Demo$primaryid),unique(Reac$primaryid))
        not_complete <- union(no_drugs,no_event)
        Demo <- Demo[!primaryid %in% not_complete]
        
        Args:
            demo_df: Demographics DataFrame
            drug_df: Drug DataFrame
            reac_df: Reaction DataFrame
            
        Returns:
            Demographics DataFrame with incomplete reports removed
        """
        try:
            # 1. Remove invalid drug entries
            invalid_substances = ['no medication', 'unspecified']
            valid_drug_df = drug_df[~drug_df['substance'].isin(invalid_substances)]
            
            # 2. Remove invalid reaction entries
            invalid_reactions = ['no adverse event']
            valid_reac_df = reac_df[~reac_df['pt'].isin(invalid_reactions)]
            
            # 3. Find primaryids missing drug or reaction info
            demo_ids = set(demo_df['primaryid'])
            drug_ids = set(valid_drug_df['primaryid'].unique())
            reac_ids = set(valid_reac_df['primaryid'].unique())
            
            # Find cases missing drug or reaction
            no_drugs = demo_ids - drug_ids
            no_event = demo_ids - reac_ids
            not_complete = no_drugs.union(no_event)
            
            # Keep only complete cases
            complete_demo_df = demo_df[~demo_df['primaryid'].isin(not_complete)]
            
            # Log removal statistics
            total_cases = len(demo_ids)
            complete_cases = len(demo_ids - not_complete)
            removed_cases = len(not_complete)
            
            self.logger.info(f"Report completion statistics:")
            self.logger.info(f"  Total cases: {total_cases}")
            self.logger.info(f"  Cases missing drugs: {len(no_drugs)}")
            self.logger.info(f"  Cases missing events: {len(no_event)}")
            self.logger.info(f"  Total incomplete cases: {removed_cases}")
            self.logger.info(f"  Complete cases kept: {complete_cases}")
            
            if removed_cases > 0:
                removal_pct = (removed_cases / total_cases) * 100
                self.logger.info(f"Removed {removed_cases} incomplete cases ({removal_pct:.1f}%)")
                
                # Log some examples of removed cases
                sample_removed = list(not_complete)[:3]
                if sample_removed:
                    self.logger.debug("Sample of removed cases:")
                    for case_id in sample_removed:
                        case_info = demo_df[demo_df['primaryid'] == case_id].iloc[0]
                        missing_type = []
                        if case_id in no_drugs:
                            missing_type.append("drug")
                        if case_id in no_event:
                            missing_type.append("event")
                        self.logger.debug(
                            f"Primaryid: {case_id}, "
                            f"Missing: {', '.join(missing_type)}, "
                            f"Quarter: {case_info.get('quarter', 'N/A')}"
                        )
            
            return complete_demo_df
            
        except Exception as e:
            self.logger.error(f"Error removing incomplete reports: {str(e)}")
            return demo_df

    def identify_premarketing_cases(self, demo_df: pd.DataFrame, drug_df: pd.DataFrame) -> pd.DataFrame:
        """Identify pre-marketing and literature cases.
        
        Matches R implementation:
        Demo[,premarketing:=primaryid%in%Drug[trial==TRUE]$primaryid]
        Demo[,literature:=!is.na(lit_ref)]
        
        Args:
            demo_df: Demographics DataFrame
            drug_df: Drug DataFrame with trial information
            
        Returns:
            Demographics DataFrame with premarketing and literature flags
        """
        try:
            # 1. Identify pre-marketing cases from trial drugs
            trial_primaryids = set(drug_df[drug_df['trial'] == True]['primaryid'])
            demo_df['premarketing'] = demo_df['primaryid'].isin(trial_primaryids)
            
            # 2. Identify literature cases
            demo_df['literature'] = demo_df['lit_ref'].notna()
            
            # Log statistics
            total_cases = len(demo_df)
            premarketing_cases = demo_df['premarketing'].sum()
            literature_cases = demo_df['literature'].sum()
            
            self.logger.info(f"Case identification statistics:")
            self.logger.info(f"  Total cases: {total_cases}")
            self.logger.info(f"  Pre-marketing cases: {premarketing_cases} ({100*premarketing_cases/total_cases:.1f}%)")
            self.logger.info(f"  Literature cases: {literature_cases} ({100*literature_cases/total_cases:.1f}%)")
            
            # Log some examples of pre-marketing cases
            if premarketing_cases > 0:
                sample_premarketing = demo_df[demo_df['premarketing']].head(3)
                self.logger.debug("Sample of pre-marketing cases:")
                for _, case in sample_premarketing.iterrows():
                    self.logger.debug(
                        f"Primaryid: {case['primaryid']}, "
                        f"Quarter: {case.get('quarter', 'N/A')}"
                    )
            
            # Log some examples of literature cases
            if literature_cases > 0:
                sample_literature = demo_df[demo_df['literature']].head(3)
                self.logger.debug("Sample of literature cases:")
                for _, case in sample_literature.iterrows():
                    self.logger.debug(
                        f"Primaryid: {case['primaryid']}, "
                        f"Quarter: {case.get('quarter', 'N/A')}, "
                        f"Lit_ref: {case['lit_ref']}"
                    )
            
            return demo_df
            
        except Exception as e:
            self.logger.error(f"Error identifying pre-marketing cases: {str(e)}")
            return demo_df

    def standardize_data(self, df: pd.DataFrame, data_type: str, file_path: Optional[str] = None) -> pd.DataFrame:
        """Standardize data based on its type."""
        try:
            file_name = os.path.basename(file_path) if file_path else "unknown_file"
            
            if data_type == 'demo':
                return self.standardizer.standardize_demographics(df)
            elif data_type == 'drug':
                if 'drugname' not in df.columns:
                    self.logger.warning(f"({file_name}) Required column 'drugname' not found, adding with default value: <NA>")
                    df['drugname'] = pd.NA
                return self.standardizer.standardize_drug_info(df)
            elif data_type == 'reac':
                return self.standardizer.standardize_reactions(df)
            else:
                self.logger.warning(f"({file_name}) Unknown data type: {data_type}")
                return df
        except Exception as e:
            self.logger.error(f"({file_name}) Error standardizing {data_type} data: {str(e)}")
            return df

    def process_data_file(self, file_path: Path, data_type: str) -> Optional[pd.DataFrame]:
        """Process a FAERS data file."""
        file_name = file_path.name
        try:
            # Read the file
            df = pd.read_csv(file_path, delimiter='$', dtype=str)
            
            if df is None:
                self.logger.error(f"({file_name}) Failed to read file: {file_path}")
                return None
                
            if df.empty:
                self.logger.error(f"({file_name}) File is empty: {file_path}")
                return None
            
            # Add missing columns with appropriate defaults
            if data_type == 'drug' and 'drugname' not in df.columns:
                self.logger.warning(f"({file_name}) Required column 'drugname' not found, adding with default value: <NA>")
                df['drugname'] = pd.NA
            
            # Standardize the data
            df = self.standardizer.standardize_data(df, data_type, str(file_path))
            
            if df is not None:
                self.logger.info(f"({file_name}) Successfully processed {len(df):,} rows")
            
            return df
            
        except Exception as e:
            self.logger.error(f"({file_name}) Error processing file: {str(e)}")
            return None
