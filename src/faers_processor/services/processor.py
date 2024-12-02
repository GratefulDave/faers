"""Service for processing FAERS data files."""
import logging
import re
from pathlib import Path
from typing import Dict, List

import dask.dataframe as dd
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import io
import os

from .standardizer import DataStandardizer


class FAERSProcessor:
    """Processes FAERS data files."""

    def __init__(
            self,
            standardizer: DataStandardizer,
            use_dask: bool = False
    ):
        """Initialize processor with standardizer.
        
        Args:
            standardizer: DataStandardizer instance initialized with external data
            use_dask: Whether to use Dask for parallel processing
        """
        self.standardizer = standardizer
        self.use_dask = use_dask

    def process_all(self, input_dir: Path, output_dir: Path) -> None:
        """Process all quarters in the input directory.
        
        Args:
            input_dir: Directory containing quarter subdirectories
            output_dir: Directory to save processed files
        """
        # Find all quarter directories
        quarter_dirs = [d for d in input_dir.iterdir() if d.is_dir() and re.match(r'\d{4}Q[1-4]', d.name)]
        
        if not quarter_dirs:
            logging.error(f"No quarter directories found in {input_dir}")
            return
            
        logging.info(f"Found {len(quarter_dirs)} quarters to process")
        
        if self.use_dask:
            import dask.bag as db
            from distributed import get_client
            
            # Get the dask client
            client = get_client()
            
            # Create a dask bag from quarter directories
            quarters_bag = db.from_sequence(quarter_dirs)
            
            # Process quarters in parallel
            results = quarters_bag.map(lambda d: self.process_quarter(d)).compute()
            
            # Combine results
            all_results = {}
            for quarter_result in results:
                for data_type, df in quarter_result.items():
                    if data_type not in all_results:
                        all_results[data_type] = []
                    all_results[data_type].append(df)
            
            # Concatenate and save results
            for data_type, dfs in all_results.items():
                if dfs:
                    combined_df = pd.concat(dfs, ignore_index=True)
                    output_file = output_dir / f"combined_{data_type}.txt"
                    combined_df.to_csv(output_file, sep='$', index=False)
                    logging.info(f"Saved combined {data_type} data to {output_file}")
        else:
            # Sequential processing
            for quarter_dir in tqdm(quarter_dirs, desc="Processing quarters"):
                try:
                    results = self.process_quarter(quarter_dir)
                    
                    # Save results for this quarter
                    quarter = quarter_dir.name
                    for data_type, df in results.items():
                        if not df.empty:
                            output_file = output_dir / f"{quarter}_{data_type}.txt"
                            df.to_csv(output_file, sep='$', index=False)
                            logging.info(f"Saved {data_type} data for {quarter} to {output_file}")
                            
                except Exception as e:
                    logging.error(f"Error processing quarter {quarter_dir}: {str(e)}")
                    continue

    def process_quarter(self, quarter_dir: Path) -> Dict[str, pd.DataFrame]:
        """Process a single quarter's worth of FAERS data.
        
        Args:
            quarter_dir: Path to quarter directory containing ASCII files
            
        Returns:
            Dictionary of processed DataFrames for demographics, drugs, and reactions
        """
        results = {}
        ascii_dir = quarter_dir / 'ascii'
        
        if not ascii_dir.exists():
            logging.error(f"ASCII directory not found in {quarter_dir}")
            return results
        
        # Find relevant files (case-insensitive)
        demo_file = next((f for f in ascii_dir.glob('[Dd][Ee][Mm][Oo]*.[Tt][Xx][Tt]')), None)
        drug_file = next((f for f in ascii_dir.glob('[Dd][Rr][Uu][Gg]*.[Tt][Xx][Tt]')), None)
        reac_file = next((f for f in ascii_dir.glob('[Rr][Ee][Aa][Cc]*.[Tt][Xx][Tt]')), None)
        
        if not all([demo_file, drug_file, reac_file]):
            logging.error(f"Missing required files in {quarter_dir}")
            return results
            
        try:
            if self.use_dask:
                # Process files in parallel using dask
                import dask.delayed as delayed
                
                # Create delayed objects for each file processing task
                demo_task = delayed(self.process_file)(demo_file, 'demographics')
                drug_task = delayed(self.process_file)(drug_file, 'drugs')
                reac_task = delayed(self.process_file)(reac_file, 'reactions')
                
                # Compute all tasks in parallel
                demo_df, drug_df, reac_df = delayed(lambda x, y, z: (x, y, z))(
                    demo_task, drug_task, reac_task
                ).compute()
                
                # Standardize results
                if not demo_df.empty:
                    demo_df = self.standardizer.standardize_demographics(demo_df)
                    results['demographics'] = demo_df
                
                if not drug_df.empty:
                    drug_df = self.standardizer.standardize_drugs(drug_df)
                    results['drugs'] = drug_df
                
                if not reac_df.empty:
                    reac_df = self.standardizer.standardize_reactions(reac_df)
                    results['reactions'] = reac_df
            else:
                # Sequential processing
                # Process demographics
                logging.info(f"Processing demographics from {demo_file}")
                demo_df = self.process_file(demo_file, 'demographics')
                if not demo_df.empty:
                    demo_df = self.standardizer.standardize_demographics(demo_df)
                    results['demographics'] = demo_df
                
                # Process drugs
                logging.info(f"Processing drugs from {drug_file}")
                drug_df = self.process_file(drug_file, 'drugs')
                if not drug_df.empty:
                    drug_df = self.standardizer.standardize_drugs(drug_df)
                    results['drugs'] = drug_df
                
                # Process reactions
                logging.info(f"Processing reactions from {reac_file}")
                reac_df = self.process_file(reac_file, 'reactions')
                if not reac_df.empty:
                    reac_df = self.standardizer.standardize_reactions(reac_df)
                    results['reactions'] = reac_df
                
        except Exception as e:
            logging.error(f"Error processing quarter {quarter_dir}: {str(e)}")
            
        return results

    def merge_quarters(self, output_dir: Path) -> None:
        """Merge all processed quarterly files into final datasets.
        
        Args:
            output_dir: Directory containing processed quarterly files
        """
        for data_type in ['demographics', 'drugs', 'reactions']:
            try:
                # Find all quarterly files for this type
                pattern = f'*_{data_type}.txt'
                quarter_files = list(output_dir.glob(pattern))
                
                if not quarter_files:
                    logging.warning(f"No quarterly files found for {data_type}")
                    continue
                
                logging.info(f"Merging {len(quarter_files)} files for {data_type}")
                
                # Read and concatenate all quarters
                dfs = []
                for file in quarter_files:
                    try:
                        quarter = file.name.split('_')[0]  # Extract quarter from filename
                        df = pd.read_csv(
                            file,
                            sep='$',
                            dtype=str,
                            na_values=['', 'NA', 'NULL'],
                            keep_default_na=True,
                            low_memory=False,
                            encoding='utf-8'
                        )
                        df['quarter'] = quarter
                        dfs.append(df)
                    except Exception as e:
                        logging.error(f"Error reading {file}: {str(e)}")
                
                if not dfs:
                    logging.error(f"No valid data found for {data_type}")
                    continue
                
                # Merge all quarters
                merged_df = pd.concat(dfs, ignore_index=True)
                
                # Convert numeric columns
                if 'primaryid' in merged_df.columns:
                    merged_df['primaryid'] = pd.to_numeric(merged_df['primaryid'], errors='coerce')
                if data_type == 'demographics':
                    if 'caseid' in merged_df.columns:
                        merged_df['caseid'] = pd.to_numeric(merged_df['caseid'], errors='coerce')
                    if 'age' in merged_df.columns:
                        merged_df['age'] = pd.to_numeric(merged_df['age'], errors='coerce')
                elif data_type == 'drugs' and 'drug_seq' in merged_df.columns:
                    merged_df['drug_seq'] = pd.to_numeric(merged_df['drug_seq'], errors='coerce')
                
                # Sort by primaryid and quarter
                merged_df = merged_df.sort_values(['primaryid', 'quarter'])
                
                # Save merged file
                output_file = output_dir / f'merged_{data_type}.txt'
                merged_df.to_csv(output_file, sep='$', index=False, encoding='utf-8')
                
                logging.info(f"Saved merged {data_type} to {output_file}")
                logging.info(f"Shape: {merged_df.shape}")
                
            except Exception as e:
                logging.error(f"Error merging {data_type}: {str(e)}")

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
        """Process a single FAERS file.
        
        Args:
            file_path: Path to the file to process
            data_type: Type of data being processed (demographics, drugs, etc.)
            
        Returns:
            Processed DataFrame
        """
        try:
            if not file_path.exists():
                logging.error(f"File does not exist: {file_path}")
                return pd.DataFrame()
            
            # Use dask if enabled
            if self.use_dask:
                try:
                    df = dd.read_csv(
                        file_path,
                        sep='$',
                        dtype=str,
                        na_values=['', 'NA', 'NULL'],
                        keep_default_na=True,
                        header=0,
                        assume_missing=True,
                        encoding='utf-8'
                    )
                    # Convert dask DataFrame to pandas DataFrame after processing
                    return self._process_dataframe(df.compute(), data_type)
                except Exception as e:
                    logging.warning(f"Dask parsing failed for {file_path}, falling back to pandas: {str(e)}")
            
            # Standard pandas processing
            try:
                df = pd.read_csv(
                    file_path,
                    sep='$',
                    dtype=str,
                    na_values=['', 'NA', 'NULL'],
                    keep_default_na=True,
                    header=0,
                    low_memory=False,
                    encoding='utf-8'
                )
                return self._process_dataframe(df, data_type)
            except Exception as e:
                logging.warning(f"Standard parsing failed for {file_path}, attempting manual fix: {str(e)}")
                
                # If standard parsing fails, try with our manual fix
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Fix known data issues
                fixed_content = self._fix_known_data_issues(file_path, content)
                
                # Try reading the fixed content
                try:
                    df = pd.read_csv(
                        io.StringIO(fixed_content),
                        sep='$',
                        dtype=str,
                        na_values=['', 'NA', 'NULL'],
                        keep_default_na=True,
                        header=0,
                        low_memory=False
                    )
                    return self._process_dataframe(df, data_type)
                except Exception as e2:
                    logging.error(f"Failed to process {file_path} even after fixes: {str(e2)}")
                    return pd.DataFrame()
                    
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            return pd.DataFrame()

    def _process_dataframe(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Process a DataFrame after it has been loaded.
        
        Args:
            df: DataFrame to process
            data_type: Type of data being processed
            
        Returns:
            Processed DataFrame
        """
        try:
            # First process the data through the standardizer
            if data_type == 'demographics':
                df = self.standardizer.process_demographics(df)
            elif data_type == 'drugs':
                df = self.standardizer.process_drugs(df)
            else:  # reactions
                df = self.standardizer.process_reactions(df)
                
            # Then apply any additional standardization
            if data_type == 'demographics':
                df = self.standardizer.standardize_demographics(df)
            elif data_type == 'drugs':
                df = self.standardizer.standardize_drugs(df)
            else:  # reactions
                df = self.standardizer.standardize_reactions(df)
                
            return df
            
        except Exception as e:
            logging.error(f"Error processing DataFrame: {str(e)}")
            return pd.DataFrame()
