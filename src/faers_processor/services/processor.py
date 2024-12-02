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


@dataclass
class TableSummary:
    """Summary statistics for a single FAERS table."""
    total_rows: int = 0
    processed_rows: int = 0
    invalid_dates: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    missing_columns: List[str] = field(default_factory=list)
    parsing_errors: List[str] = field(default_factory=list)
    processing_time: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate percentage of successfully processed rows."""
        return (self.processed_rows / self.total_rows * 100) if self.total_rows > 0 else 0.0


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

        # Individual quarter summaries
        report.append("## Quarter-by-Quarter Summary\n")
        for quarter, summary in sorted(self.quarter_summaries.items()):
            report.append(f"### Quarter {quarter}\n")

            # Create summary table for all file types
            file_types = {
                'Demographics (DEMO)': summary.demo_summary,
                'Drug Data (DRUG)': summary.drug_summary,
                'Reactions (REAC)': summary.reac_summary,
                'Outcomes (OUTC)': summary.outc_summary,
                'Report Sources (RPSR)': summary.rpsr_summary,
                'Therapies (THER)': summary.ther_summary,
                'Indications (INDI)': summary.indi_summary
            }

            table_data = []
            for file_type, stats in file_types.items():
                row = [
                    file_type,
                    f"{stats.total_rows:,}",
                    f"{stats.processed_rows:,}",
                    f"{stats.success_rate:.1f}%",
                    len(stats.parsing_errors)
                ]
                table_data.append(row)

            # Add summary table
            report.append("| File Type | Total Rows | Processed Rows | Success Rate | Errors |\n")
            report.append("|-----------|------------|----------------|--------------|--------|\n")
            for row in table_data:
                report.append(f"| {' | '.join(str(x) for x in row)} |\n")
            report.append("\n")

            # Add error details if any exist
            has_errors = False
            error_report = ["#### Processing Errors\n"]
            for file_type, stats in file_types.items():
                if stats.parsing_errors:
                    has_errors = True
                    error_report.append(f"\n**{file_type}**\n")
                    for error in stats.parsing_errors:
                        error_report.append(f"- {error}\n")

            if has_errors:
                report.extend(error_report)
                report.append("\n")

            # Add processing time
            report.append(f"Processing time: {summary.processing_time:.2f} seconds\n\n")

        # Grand summary
        report.append("## Grand Summary\n")
        grand_totals = defaultdict(lambda: {'total': 0, 'processed': 0, 'errors': 0})

        for summary in self.quarter_summaries.values():
            for file_type, stats in file_types.items():
                grand_totals[file_type]['total'] += stats.total_rows
                grand_totals[file_type]['processed'] += stats.processed_rows
                grand_totals[file_type]['errors'] += len(stats.parsing_errors)

        # Add grand totals table
        report.append("| File Type | Total Rows | Processed Rows | Success Rate | Total Errors |\n")
        report.append("|-----------|------------|----------------|--------------|--------------|\n")

        for file_type, totals in grand_totals.items():
            success_rate = (totals['processed'] / totals['total'] * 100) if totals['total'] > 0 else 0
            row = [
                file_type,
                f"{totals['total']:,}",
                f"{totals['processed']:,}",
                f"{success_rate:.1f}%",
                totals['errors']
            ]
            report.append(f"| {' | '.join(str(x) for x in row)} |\n")

        return ''.join(report)


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

    def process_all(self, input_dir: Path, output_dir: Path, max_workers: int = None) -> None:
        """Process all quarters in the input directory and save merged results in output_dir."""
        import time
        from datetime import datetime

        # Convert to absolute paths and resolve any symlinks
        input_dir = input_dir.resolve()
        output_dir = output_dir.resolve()

        logging.info(f"Using absolute paths:")
        logging.info(f"Input directory: {input_dir}")
        logging.info(f"Output directory: {output_dir}")

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
            logging.info(f"Cleaning output directory: {output_dir}")
            for file in output_dir.glob('*.txt'):
                file.unlink()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all quarter directories
        quarter_dirs = [d for d in input_dir.iterdir() if d.is_dir() and re.match(r'\d{4}Q[1-4]', d.name, re.IGNORECASE)]
        total_quarters = len(quarter_dirs)
        results['total_quarters'] = total_quarters

        if not quarter_dirs:
            logging.error(f"No quarter directories found in {input_dir}")
            return

        logging.info(f"Found {total_quarters} quarters to process")

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
                        logging.error(f"Error processing {quarter}: {str(e)}")
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
                    logging.error(f"Error processing {quarter}: {str(e)}")
                    pbar.update(1)
                    pbar.set_postfix({"quarter": quarter, "status": "failed"}, refresh=True)

        pbar.close()

        # Merge and save all data types
        for data_type, dfs in all_data.items():
            if dfs:
                try:
                    logging.info(f"Merging {len(dfs)} quarters for {data_type}")
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
                    logging.info(f"Saving {data_type} to: {output_file}")
                    merged_df.to_csv(output_file, sep='$', index=False, encoding='utf-8')
                    logging.info(f"Successfully saved {data_type} to {output_file}")
                    logging.info(f"Shape: {merged_df.shape}")
                except Exception as e:
                    logging.error(f"Error merging {data_type}: {str(e)}")

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

        with open(report_path, "w") as f:
            f.write(summary.generate_markdown_report())

        logging.info(f"Generated processing report: {report_path}")

    def _find_ascii_directory(self, quarter_dir: Path) -> Optional[Path]:
        """Find the ASCII directory - it's always ASCII or ascii."""
        # Just check for ASCII or ascii - that's all we need
        ascii_dir = quarter_dir / 'ASCII'
        if ascii_dir.is_dir():
            return ascii_dir

        ascii_dir = quarter_dir / 'ascii'
        if ascii_dir.is_dir():
            return ascii_dir

        # Log what we found to help debug
        logging.info(f"Contents of {quarter_dir}:")
        for item in quarter_dir.iterdir():
            logging.info(f"  - {item.name}")

        return None

    def process_quarter(self, quarter_dir: Path) -> Dict[str, pd.DataFrame]:
        """Process a single quarter directory."""
        quarter_dir = self._normalize_quarter_path(quarter_dir)
        ascii_dir = self._find_ascii_directory(quarter_dir)

        if not ascii_dir:
            raise ValueError(f"No ASCII directory found in {quarter_dir}")

        logging.info(f"Processing quarter directory: {quarter_dir}")
        logging.info(f"Using ASCII directory: {ascii_dir}")

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
                # Find matching files
                matched_files = []
                for pattern in patterns:
                    matched_files.extend(ascii_dir.glob(f'*{pattern}*.txt'))

                if not matched_files:
                    logging.warning(f"No {data_type} files found in {ascii_dir}")
                    continue

                # Use the first matching file
                file_path = matched_files[0]
                logging.info(f"Processing {data_type} file: {file_path}")

                # Process the file
                df = self.process_file(file_path, data_type)
                if not df.empty:
                    results[data_type] = df
                    logging.info(f"Successfully processed {data_type} file. Shape: {df.shape}")
                else:
                    logging.warning(f"Empty DataFrame returned for {data_type}")

            except Exception as e:
                logging.error(f"Error processing {data_type} in {quarter_dir}: {str(e)}")
                raise

        return results

    def _fix_known_data_issues(self, file_path: Path, content: str) -> str:
        """Fix known data formatting issues in specific FAERS files.

        Args:
            file_path: Path to the file being processed
            content: Current content of the file

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
            lines = content.splitlines()

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

    def process_file(self, file_path: Path, data_type: str) -> pd.DataFrame:
        """Process a single FAERS file with enhanced encoding detection and error handling."""
        logging.info(f"Processing {data_type} file: {file_path}")

        def detect_encoding(file_path: Path) -> str:
            """Detect file encoding using chardet."""
            try:
                # Read a sample (first 1MB) for encoding detection
                with open(file_path, 'rb') as f:
                    raw = f.read(1024 * 1024)
                result = chardet.detect(raw)
                confidence = result.get('confidence', 0)
                encoding = result.get('encoding', 'utf-8')

                logging.info(f"Detected encoding for {file_path.name}: {encoding} (confidence: {confidence:.2%})")
                return encoding
            except Exception as e:
                logging.warning(f"Error detecting encoding for {file_path.name}: {str(e)}")
                return None

        def read_problematic_line(line_num: int, encoding: str) -> str:
            """Read a specific line from the file using given encoding."""
            try:
                with open(file_path, 'rb') as f:  # Open in binary mode
                    for i, line in enumerate(f, 1):
                        if i == line_num:
                            try:
                                return line.decode(encoding).strip()
                            except UnicodeDecodeError:
                                # If specific line fails to decode, try to show hex
                                return f"[Binary data: {line.hex()}]"
            except Exception as e:
                logging.warning(f"Could not read line {line_num} with {encoding} encoding: {str(e)}")
                return None

        def log_bad_row(line_num: int, expected: int, actual: int, encoding: str):
            """Log detailed information about problematic rows."""
            row_data = read_problematic_line(line_num, encoding)
            if row_data:
                logging.warning(
                    f"\nProblematic row in {file_path.name}:\n"
                    f"Line {line_num}: Expected {expected} fields but got {actual}\n"
                    f"Encoding: {encoding}\n"
                    f"Data: {row_data}"
                )

        # Set up warning handler for pandas
        def warning_handler(message, category, filename, lineno, file=None, line=None):
            if category == pd.errors.ParserWarning:
                match = re.search(r'Skipping line (\d+): expected (\d+) fields, saw (\d+)', str(message))
                if match:
                    line_num, expected, actual = map(int, match.groups())
                    log_bad_row(line_num, expected, actual, current_encoding)

        import warnings
        original_handler = warnings.showwarning
        warnings.showwarning = warning_handler

        try:
            # Special handling for known problematic files
            if "DRUG19Q3" in str(file_path):
                logging.info(f"Using special handling for known problematic file: {file_path.name}")
                try:
                    # Try reading with error_bad_lines=False first
                    df = pd.read_csv(
                        file_path,
                        delimiter='$',
                        dtype=str,
                        na_values=['', 'NULL', 'null'],
                        keep_default_na=False,
                        encoding='latin1',  # Try latin1 first for DRUG files
                        on_bad_lines='skip',  # Skip bad lines
                        engine='python',
                        quoting=3,  # QUOTE_NONE
                        escapechar=None
                    )
                    logging.info(f"Successfully read {file_path.name} with special handling")
                    logging.info(f"Read {len(df)} rows from {file_path.name}")
                    return df
                except Exception as e:
                    logging.error(f"Special handling failed for {file_path.name}: {str(e)}")
                    # Fall through to normal processing

            # Normal processing with encoding detection
            detected = detect_encoding(file_path)
            encodings = []
            if detected and detected.lower() not in ['utf-8', 'ascii', 'latin1', 'cp1252', 'iso-8859-1']:
                encodings.append(detected)

            # For DRUG files, try latin1 first as it's often correct
            if "DRUG" in str(file_path):
                encodings = ['latin1', 'utf-8', 'cp1252', 'iso-8859-1']
            else:
                encodings.extend(['utf-8', 'latin1', 'cp1252', 'iso-8859-1'])

            # Try each encoding
            last_error = None
            for encoding in encodings:
                try:
                    current_encoding = encoding
                    logging.info(f"Attempting to read {file_path.name} with {encoding} encoding")

                    df = pd.read_csv(
                        file_path,
                        delimiter='$',
                        dtype=str,
                        na_values=['', 'NULL', 'null'],
                        keep_default_na=False,
                        encoding=encoding,
                        on_bad_lines='warn',
                        engine='python',
                        quoting=3,  # QUOTE_NONE
                        escapechar=None
                    )

                    logging.info(f"Successfully read {file_path.name} with {encoding} encoding")
                    logging.info(f"Read {len(df)} rows from {file_path.name}")

                    return df

                except UnicodeDecodeError as e:
                    last_error = e
                    logging.warning(f"Failed to read {file_path.name} with {encoding} encoding: {str(e)}")
                    continue
                except Exception as e:
                    last_error = e
                    logging.error(f"Error processing {file_path.name} with {encoding} encoding: {str(e)}")
                    if encoding == encodings[-1]:
                        raise
                    continue

        finally:
            # Restore original warning handler
            warnings.showwarning = original_handler

        # If we get here, all encodings failed
        raise ValueError(f"Failed to read {file_path.name} with any encoding. Last error: {str(last_error)}")

    def _process_dataframe(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Process a DataFrame based on its type."""
        if df.empty:
            return df

        try:
            # Ensure we have a DataFrame
            if isinstance(df, pd.Series):
                df = df.to_frame().T

            # Convert dtypes to string to avoid issues
            df = df.astype(str)

            # Standardize based on type
            if data_type == 'demo':
                return self.standardizer.standardize_demographics(df)
            elif data_type == 'drug':
                return self.standardizer.standardize_drugs(df)
            elif data_type == 'reac':
                return self.standardizer.standardize_reactions(df)
            else:
                logging.warning(f"Unknown data type: {data_type}")
                return df

        except Exception as e:
            logging.error(f"Error processing DataFrame: {str(e)}")
            return pd.DataFrame()

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
