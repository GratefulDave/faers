"""Service for processing FAERS data files."""
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import io
import os
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from .standardizer import DataStandardizer


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
        """Process all quarters in the input directory.
        
        Args:
            input_dir: Directory containing quarter subdirectories
            output_dir: Directory to save processed files
            max_workers: Maximum number of worker processes to use
        """
        # Find all quarter directories
        quarter_dirs = [d for d in input_dir.iterdir() if d.is_dir() and re.match(r'\d{4}Q[1-4]', d.name, re.IGNORECASE)]
        
        if not quarter_dirs:
            logging.error(f"No quarter directories found in {input_dir}")
            return
            
        logging.info(f"Found {len(quarter_dirs)} quarters to process")
        
        if self.use_parallel:
            # Use ProcessPoolExecutor for parallel processing
            if max_workers is None:
                max_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Process quarters in batches to control memory
                batch_size = 4
                for i in range(0, len(quarter_dirs), batch_size):
                    batch = quarter_dirs[i:i + batch_size]
                    logging.info(f"Processing batch of {len(batch)} quarters")
                    
                    # Submit batch for processing
                    futures = {
                        executor.submit(self.process_quarter, quarter_dir): quarter_dir 
                        for quarter_dir in batch
                    }
                    
                    # Process results as they complete
                    for future in concurrent.futures.as_completed(futures):
                        quarter_dir = futures[future]
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

    def _find_ascii_directory(self, quarter_dir: Path) -> Optional[Path]:
        """Find the ASCII directory using multiple search strategies."""
        # Common ASCII directory names
        ascii_names = ['ascii', 'ASCII', 'Ascii']
        
        # Strategy 1: Direct subdirectory
        for name in ascii_names:
            ascii_dir = quarter_dir / name
            if ascii_dir.is_dir():
                return ascii_dir
        
        # Strategy 2: Case-insensitive search in immediate subdirectories
        for subdir in quarter_dir.iterdir():
            if subdir.is_dir() and 'ascii' in subdir.name.lower():
                return subdir
        
        # Strategy 3: Look for .txt files directly in quarter directory
        txt_files = list(quarter_dir.glob('*.txt'))
        if txt_files:
            return quarter_dir
            
        # Strategy 4: Search one level deeper
        for subdir in quarter_dir.iterdir():
            if not subdir.is_dir():
                continue
            # Check for ASCII subdirectory
            for name in ascii_names:
                ascii_dir = subdir / name
                if ascii_dir.is_dir():
                    return ascii_dir
            # Check for .txt files
            txt_files = list(subdir.glob('*.txt'))
            if txt_files:
                return subdir
        
        return None

    def process_quarter(self, quarter_dir: Path) -> Dict[str, pd.DataFrame]:
        """Process a single quarter directory."""
        results = {}
        
        try:
            # Find ASCII directory using multiple strategies
            ascii_dir = self._find_ascii_directory(quarter_dir)
            
            if not ascii_dir:
                logging.error(f"No valid ASCII directory found in {quarter_dir}")
                logging.info(f"Searched in: {[d.name for d in quarter_dir.iterdir() if d.is_dir()]}")
                return results
                
            logging.info(f"Found ASCII directory: {ascii_dir}")
            
            # Find required files (case-insensitive)
            demo_file = None
            drug_file = None
            reac_file = None
            
            # Look for files in ASCII directory and its subdirectories
            for file in ascii_dir.rglob('*.txt'):
                if not file.is_file():
                    continue
                    
                name_lower = file.name.lower()
                if 'demo' in name_lower:
                    demo_file = file
                    logging.info(f"Found demographics file: {file}")
                elif 'drug' in name_lower:
                    drug_file = file
                    logging.info(f"Found drug file: {file}")
                elif 'reac' in name_lower:
                    reac_file = file
                    logging.info(f"Found reactions file: {file}")
            
            # Check if all required files are found
            missing_files = []
            if not demo_file:
                missing_files.append('demographics')
            if not drug_file:
                missing_files.append('drugs')
            if not reac_file:
                missing_files.append('reactions')
                
            if missing_files:
                logging.error(f"Missing files in {ascii_dir}: {', '.join(missing_files)}")
                logging.info(f"Found files: {[f.name for f in ascii_dir.rglob('*.txt')]}")
                return results
            
            # Process each file type
            logging.info(f"Processing demographics from {demo_file}")
            demo_df = self.process_file(demo_file, 'demographics')
            if not demo_df.empty:
                results['demographics'] = demo_df
            
            logging.info(f"Processing drugs from {drug_file}")
            drug_df = self.process_file(drug_file, 'drugs')
            if not drug_df.empty:
                results['drugs'] = drug_df
            
            logging.info(f"Processing reactions from {reac_file}")
            reac_df = self.process_file(reac_file, 'reactions')
            if not reac_df.empty:
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
        """Process a single FAERS file."""
        try:
            if not file_path.exists():
                logging.error(f"File does not exist: {file_path}")
                return pd.DataFrame()
            
            # Calculate chunk size based on file size
            file_size = file_path.stat().st_size
            chunk_size = min(50_000, max(10_000, file_size // (100 * 1024 * 1024)))
            
            chunks = []
            total_chunks = 0
            
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            last_error = None
            
            for encoding in encodings:
                try:
                    # Read and process in chunks
                    for chunk in pd.read_csv(
                        file_path,
                        sep='$',
                        dtype=str,  # Read all columns as strings
                        na_values=['', 'NA', 'NULL'],
                        keep_default_na=True,
                        header=0,
                        chunksize=chunk_size,
                        low_memory=False,
                        encoding=encoding,
                        on_bad_lines='warn'  # More permissive line parsing
                    ):
                        # Process each chunk immediately
                        processed_chunk = self._process_dataframe(chunk, data_type)
                        if not processed_chunk.empty:
                            chunks.append(processed_chunk)
                            total_chunks += 1
                            
                        # Log progress
                        if total_chunks % 10 == 0:
                            logging.info(f"Processed {total_chunks} chunks from {file_path}")
                    
                    # If we get here, reading succeeded
                    break
                    
                except Exception as e:
                    last_error = e
                    logging.warning(f"Failed to read with {encoding} encoding: {str(e)}")
                    continue
            
            if chunks:
                # Combine processed chunks
                return pd.concat(chunks, ignore_index=True)
            
            if last_error:
                logging.error(f"Failed to process {file_path} with any encoding: {str(last_error)}")
            return pd.DataFrame()
                
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            return pd.DataFrame()

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
            if data_type == 'demographics':
                return self.standardizer.standardize_demographics(df)
            elif data_type == 'drugs':
                return self.standardizer.standardize_drugs(df)
            elif data_type == 'reactions':
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
