"""Service for processing FAERS data files."""
import logging
from pathlib import Path
from typing import Dict, List

import dask.dataframe as dd
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from .standardizer import DataStandardizer


class FAERSProcessor:
    """Processes FAERS data files."""

    def __init__(
            self,
            standardizer: DataStandardizer
    ):
        """Initialize processor with standardizer.
        
        Args:
            standardizer: DataStandardizer instance initialized with external data
        """
        self.standardizer = standardizer

    def process_quarter(self, quarter_dir: Path) -> Dict[str, pd.DataFrame]:
        """Process a single quarter's worth of FAERS data.
        
        Args:
            quarter_dir: Path to quarter directory containing ASCII files
            
        Returns:
            Dictionary of processed DataFrames for demographics, drugs, and reactions
        """
        results = {}
        ascii_dir = quarter_dir / 'ascii'
        
        # Find relevant files
        demo_file = next(ascii_dir.glob('DEMO*.txt'), None)
        drug_file = next(ascii_dir.glob('DRUG*.txt'), None)
        reac_file = next(ascii_dir.glob('REAC*.txt'), None)
        
        if not all([demo_file, drug_file, reac_file]):
            logging.error(f"Missing required files in {quarter_dir}")
            return results
            
        try:
            # Process demographics
            logging.info(f"Processing demographics from {demo_file}")
            demo_df = self.standardizer.process_file(demo_file, 'demographics')
            if not demo_df.empty:
                demo_df = self.standardizer.standardize_demographics(demo_df)
                results['demographics'] = demo_df
            
            # Process drugs
            logging.info(f"Processing drugs from {drug_file}")
            drug_df = self.standardizer.process_file(drug_file, 'drugs')
            if not drug_df.empty:
                drug_df = self.standardizer.standardize_drugs(drug_df)
                results['drugs'] = drug_df
            
            # Process reactions
            logging.info(f"Processing reactions from {reac_file}")
            reac_df = self.standardizer.process_file(reac_file, 'reactions')
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

    def process_all(self, input_dir: Path, output_dir: Path) -> None:
        """Process all FAERS quarters and merge results.
        
        Args:
            input_dir: Directory containing raw FAERS data
            output_dir: Directory to save processed files
        """
        # Create output directory if needed
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all quarter directories
        quarter_dirs = [d for d in input_dir.iterdir() if d.is_dir() and re.match(r'\d{4}q[1-4]', d.name)]
        quarter_dirs.sort()  # Process in chronological order
        
        logging.info(f"Found {len(quarter_dirs)} quarters to process")
        
        # Process each quarter
        for quarter_dir in tqdm(quarter_dirs, desc="Processing quarters"):
            try:
                quarter_results = self.process_quarter(quarter_dir)
                
                if not quarter_results:
                    logging.warning(f"No results for quarter {quarter_dir}")
                    continue
                
                # Save quarterly results
                quarter_prefix = quarter_dir.name
                for data_type, df in quarter_results.items():
                    output_file = output_dir / f"{quarter_prefix}_{data_type}.txt"
                    df.to_csv(output_file, sep='$', index=False, encoding='utf-8')
                    logging.info(f"Saved {data_type} to {output_file}")
                
            except Exception as e:
                logging.error(f"Error processing quarter {quarter_dir}: {str(e)}")
        
        # Merge all quarters
        self.merge_quarters(output_dir)

    def process_file(self, file_path: Path, data_type: str) -> pd.DataFrame:
        """Process a single FAERS file.
        
        Args:
            file_path: Path to the file
            data_type: Type of data ('demographics', 'drugs', 'reactions')
        
        Returns:
            Processed DataFrame
        """
        try:
            logging.info(f"Reading {data_type} file: {file_path}")
            
            if not file_path.exists():
                logging.error(f"File does not exist: {file_path}")
                return pd.DataFrame()
            
            # Define expected columns based on data type (matching R script column names)
            expected_columns = {
                'demographics': ['primaryid', 'caseid', 'caseversion', 'i_f_code', 'event_dt', 'mfr_dt', 'init_fda_dt',
                               'fda_dt', 'rept_cod', 'mfr_num', 'mfr_sndr', 'age', 'age_cod', 'age_grp', 'sex',
                               'e_sub', 'wt', 'wt_cod', 'rept_dt', 'to_mfr', 'occp_cod', 'reporter_country', 'occr_country'],
                'drugs': ['primaryid', 'caseid', 'drug_seq', 'role_cod', 'drugname', 'prod_ai', 'val_vbm', 'route',
                         'dose_vbm', 'cum_dose_chr', 'cum_dose_unit', 'dechal', 'rechal', 'lot_num', 'exp_dt',
                         'nda_num', 'dose_amt', 'dose_unit', 'dose_form', 'dose_freq'],
                'reactions': ['primaryid', 'caseid', 'pt', 'drug_rec_act']
            }
            
            # Map old column names to new ones (matching R script)
            column_mapping = {
                'demographics': {
                    'isr': 'primaryid',
                    'case': 'caseid',
                    'i_f_cod': 'i_f_code',
                    'gndr_cod': 'sex'
                },
                'drugs': {
                    'isr': 'primaryid',
                    'drug_seq': 'drug_seq',
                    'DRUGNAME': 'drugname',  # Map uppercase to lowercase
                    'drugname': 'drugname',  # Keep lowercase mapping for newer files
                    'prod_ai': 'prod_ai'
                },
                'reactions': {
                    'isr': 'primaryid',
                    'pt': 'pt'
                }
            }
            
            # Read file with pandas
            try:
                # Read file with $ delimiter and first row as header
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
                    logging.debug(f"File {file_path.name}")
                    logging.debug(f"Original columns: {df.columns.tolist()}")
                except UnicodeDecodeError:
                    # If UTF-8 fails, try with latin1 encoding
                    df = pd.read_csv(
                        file_path,
                        sep='$',
                        dtype=str,
                        na_values=['', 'NA', 'NULL'],
                        keep_default_na=True,
                        header=0,
                        low_memory=False,
                        encoding='latin1'
                    )
                    logging.debug(f"File {file_path.name} (latin1)")
                    logging.debug(f"Original columns: {df.columns.tolist()}")
                
                # Rename columns according to mapping
                if data_type in column_mapping:
                    logging.debug(f"Before mapping: {df.columns.tolist()}")
                    df = df.rename(columns=column_mapping[data_type])
                    logging.debug(f"After mapping: {df.columns.tolist()}")
                    logging.debug(f"Has 'drugname': {'drugname' in df.columns}")
                
                # Convert numeric columns
                if 'primaryid' in df.columns:
                    df['primaryid'] = pd.to_numeric(df['primaryid'], errors='coerce').fillna(-1).astype('int64')
                if data_type == 'demographics':
                    if 'caseid' in df.columns:
                        df['caseid'] = pd.to_numeric(df['caseid'], errors='coerce')
                    if 'age' in df.columns:
                        df['age'] = pd.to_numeric(df['age'], errors='coerce')
                elif data_type == 'drugs' and 'drug_seq' in df.columns:
                    df['drug_seq'] = pd.to_numeric(df['drug_seq'], errors='coerce').fillna(-1).astype('int64')
                
                # Process based on data type
                if data_type == 'demographics':
                    result = self.standardizer.process_demographics(df)
                elif data_type == 'drugs':
                    # First clean and standardize drug names (like R script)
                    drugname_col = 'DRUGNAME' if 'DRUGNAME' in df.columns else 'drugname'
                    if drugname_col in df.columns:
                        df[drugname_col] = df[drugname_col].str.lower().str.strip()
                        df[drugname_col] = df[drugname_col].str.replace(r'\s+', ' ', regex=True)
                        df[drugname_col] = df[drugname_col].str.replace(r'\.$', '', regex=True)
                        df[drugname_col] = df[drugname_col].str.replace(r'\( ', '(', regex=True)
                        df[drugname_col] = df[drugname_col].str.replace(r' \)', ')', regex=True)
                        # Rename to lowercase after cleaning if needed
                        if drugname_col == 'DRUGNAME':
                            df = df.rename(columns={'DRUGNAME': 'drugname'})
                    else:
                        logging.error(f"Required column '{drugname_col}' not found in columns: {df.columns.tolist()}")
                        return pd.DataFrame()
                    
                    # Then process through standardizer
                    result = self.standardizer.process_drugs(df)
                else:  # reactions
                    result = self.standardizer.process_reactions(df)

                return result
                
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {str(e)}")
                return pd.DataFrame()

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            return pd.DataFrame()
