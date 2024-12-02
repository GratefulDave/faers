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
        quarter_dirs = [d for d in input_dir.iterdir() if d.is_dir() and re.match(r'\d{4}Q[1-4]', d.name, re.IGNORECASE)]
        
        if not quarter_dirs:
            logging.error(f"No quarter directories found in {input_dir}")
            return
            
        logging.info(f"Found {len(quarter_dirs)} quarters to process")
        
        if self.use_dask:
            import dask.dataframe as dd
            from distributed import get_client
            
            # Get the dask client
            client = get_client()
            
            # Process quarters in smaller batches to avoid memory issues
            batch_size = 4  # Process 4 quarters at a time
            for i in range(0, len(quarter_dirs), batch_size):
                batch = quarter_dirs[i:i + batch_size]
                logging.info(f"Processing batch of {len(batch)} quarters")
                
                # Process batch in parallel
                futures = []
                for quarter_dir in batch:
                    future = client.submit(self.process_quarter, quarter_dir)
                    futures.append(future)
                
                # Gather results as they complete
                for future, quarter_dir in zip(futures, batch):
                    try:
                        results = future.result()
                        quarter = quarter_dir.name
                        
                        # Save results for this quarter
                        for data_type, df in results.items():
                            if not df.empty:
                                output_file = output_dir / f"{quarter}_{data_type}.txt"
                                df.to_csv(output_file, sep='$', index=False)
                                logging.info(f"Saved {data_type} data for {quarter} to {output_file}")
                    
                    except Exception as e:
                        logging.error(f"Error processing quarter {quarter_dir}: {str(e)}")
                        continue
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
        # Normalize quarter path first
        quarter_dir = self._normalize_quarter_path(quarter_dir)
        results = {}
        
        # Look for ASCII files in both quarter_dir and ascii subdirectory
        ascii_paths = [
            quarter_dir,  # Try root directory first
            quarter_dir / 'ascii',  # Then try ascii subdirectory
            quarter_dir / 'ASCII',  # Then try ASCII subdirectory (case sensitive)
            quarter_dir.parent / quarter_dir.name.lower() / 'ascii',  # Try lowercase path
            quarter_dir.parent / quarter_dir.name.upper() / 'ASCII'   # Try uppercase path
        ]
        
        # Find the first valid path that contains our files
        ascii_dir = None
        for path in ascii_paths:
            if path.exists() and any(path.glob('*.[Tt][Xx][Tt]')):
                ascii_dir = path
                break
        
        if not ascii_dir:
            logging.error(f"No valid ASCII directory found in {quarter_dir}")
            return results
            
        logging.info(f"Using ASCII directory: {ascii_dir}")
        
        # Find relevant files (case-insensitive)
        demo_file = next((f for f in ascii_dir.glob('[Dd][Ee][Mm][Oo]*.[Tt][Xx][Tt]')), None)
        drug_file = next((f for f in ascii_dir.glob('[Dd][Rr][Uu][Gg]*.[Tt][Xx][Tt]')), None)
        reac_file = next((f for f in ascii_dir.glob('[Rr][Ee][Aa][Cc]*.[Tt][Xx][Tt]')), None)
        
        if not all([demo_file, drug_file, reac_file]):
            missing = []
            if not demo_file: missing.append("demographics")
            if not drug_file: missing.append("drugs")
            if not reac_file: missing.append("reactions")
            logging.error(f"Missing files in {quarter_dir}: {', '.join(missing)}")
            return results
            
        try:
            if self.use_dask:
                # Process files sequentially but use dask for the file reading
                # This avoids memory issues while still maintaining parallelism
                logging.info(f"Processing demographics from {demo_file}")
                demo_df = self.process_file(demo_file, 'demographics')
                if not demo_df.empty:
                    demo_df = self.standardizer.standardize_demographics(demo_df)
                    results['demographics'] = demo_df
                
                logging.info(f"Processing drugs from {drug_file}")
                drug_df = self.process_file(drug_file, 'drugs')
                if not drug_df.empty:
                    drug_df = self.standardizer.standardize_drugs(drug_df)
                    results['drugs'] = drug_df
                
                logging.info(f"Processing reactions from {reac_file}")
                reac_df = self.process_file(reac_file, 'reactions')
                if not reac_df.empty:
                    reac_df = self.standardizer.standardize_reactions(reac_df)
                    results['reactions'] = reac_df
            else:
                # Sequential processing (same as above)
                logging.info(f"Processing demographics from {demo_file}")
                demo_df = self.process_file(demo_file, 'demographics')
                if not demo_df.empty:
                    demo_df = self.standardizer.standardize_demographics(demo_df)
                    results['demographics'] = demo_df
                
                logging.info(f"Processing drugs from {drug_file}")
                drug_df = self.process_file(drug_file, 'drugs')
                if not drug_df.empty:
                    drug_df = self.standardizer.standardize_drugs(drug_df)
                    results['drugs'] = drug_df
                
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
                            header=0,
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
            
            # Use dask if enabled - read in chunks to avoid memory issues
            if self.use_dask:
                try:
                    # Calculate optimal chunk size based on file size
                    file_size = file_path.stat().st_size
                    chunk_size = min(100_000, max(10_000, file_size // (50 * 1024 * 1024)))  # Aim for ~50MB chunks
                    
                    # Read file in chunks using dask
                    df = dd.read_csv(
                        file_path,
                        sep='$',
                        dtype=str,
                        na_values=['', 'NA', 'NULL'],
                        keep_default_na=True,
                        header=0,
                        assume_missing=True,
                        blocksize=chunk_size * 1024,  # Convert to bytes
                        encoding='utf-8',
                        sample=10000  # Sample fewer rows for metadata
                    )
                    # Convert to pandas immediately to avoid large graph
                    return self._process_dataframe(df.compute(), data_type)
                except Exception as e:
                    logging.warning(f"Dask parsing failed for {file_path}, falling back to pandas: {str(e)}")
            
            # Standard pandas processing
            try:
                # Read in chunks with pandas
                chunk_size = 100_000
                chunks = []
                for chunk in pd.read_csv(
                    file_path,
                    sep='$',
                    dtype=str,
                    na_values=['', 'NA', 'NULL'],
                    keep_default_na=True,
                    header=0,
                    chunksize=chunk_size,
                    low_memory=False,
                    encoding='utf-8'
                ):
                    chunks.append(chunk)
                
                if chunks:
                    df = pd.concat(chunks, ignore_index=True)
                    return self._process_dataframe(df, data_type)
                return pd.DataFrame()
                
            except Exception as e:
                logging.warning(f"Standard parsing failed for {file_path}, attempting manual fix: {str(e)}")
                
                # If standard parsing fails, try with our manual fix
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Fix known data issues
                fixed_content = self._fix_known_data_issues(file_path, content)
                
                # Try reading the fixed content in chunks
                try:
                    chunks = []
                    for chunk in pd.read_csv(
                        io.StringIO(fixed_content),
                        sep='$',
                        dtype=str,
                        na_values=['', 'NA', 'NULL'],
                        keep_default_na=True,
                        header=0,
                        chunksize=chunk_size,
                        low_memory=False
                    ):
                        chunks.append(chunk)
                    
                    if chunks:
                        df = pd.concat(chunks, ignore_index=True)
                        return self._process_dataframe(df, data_type)
                    return pd.DataFrame()
                    
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
