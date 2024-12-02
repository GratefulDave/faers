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
            data_dir: Path,
            external_dir: Path,
            chunk_size: int = 100000,
            use_dask: bool = False
    ):
        """Initialize processor with configuration.
        
        Args:
            data_dir: Base directory containing raw data
            external_dir: Directory containing external reference data
            chunk_size: Size of data chunks for processing
            use_dask: Whether to use Dask for out-of-core processing
        """
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / 'clean'
        self.external_dir = Path(external_dir)
        self.chunk_size = chunk_size
        self.use_dask = use_dask

        # Initialize standardizer with external data
        self.standardizer = DataStandardizer(external_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Data directory: {self.data_dir}")
        logging.info(f"Output directory: {self.output_dir}")

    def process_all(self) -> None:
        """Process all FAERS data files."""
        try:
            # Get list of all quarters from data directory (excluding clean dir)
            quarters = []
            logging.info("Discovering quarters...")
            for d in self.data_dir.iterdir():
                if d.is_dir() and d.name.lower() != 'clean':
                    # Check if it matches quarter pattern (e.g., 2004q1)
                    if len(d.name) == 6 and d.name[:4].isdigit() and d.name[4] == 'q' and d.name[5].isdigit():
                        quarters.append(d.name)
                        logging.info(f"Found quarter: {d.name}")
            
            quarters.sort()  # Sort quarters chronologically
            
            if not quarters:
                logging.warning(f"No quarters found in {self.data_dir}")
                logging.info("Directory contents:")
                for item in self.data_dir.iterdir():
                    logging.info(f"  {item.name} ({'dir' if item.is_dir() else 'file'})")
                return
                
            logging.info(f"Found {len(quarters)} quarters to process")
            
            # Process each quarter
            with tqdm(total=len(quarters), desc="Processing quarters") as pbar:
                for quarter in quarters:
                    try:
                        # Check for ascii directory case-insensitively
                        quarter_path = self.data_dir / quarter
                        if not quarter_path.exists():
                            logging.warning(f"Quarter directory {quarter} not found")
                            continue

                        # Find ascii directory case-insensitively
                        logging.info(f"Looking for ascii directory in {quarter_path}")
                        ascii_dir = None
                        for d in quarter_path.iterdir():
                            logging.info(f"  Checking directory: {d.name}")
                            if d.is_dir() and d.name.lower() == 'ascii':
                                ascii_dir = d
                                break
                        
                        if not ascii_dir:
                            logging.warning(f"No ascii directory found for quarter {quarter}")
                            continue
                            
                        logging.info(f"Processing quarter {quarter} from {ascii_dir}")
                        
                        # List all files in ascii directory
                        logging.info(f"All files in {ascii_dir}:")
                        for f in ascii_dir.iterdir():
                            logging.info(f"  Found file: {f.name}")
                        
                        # Look for files case-insensitively
                        demo_files = []
                        drug_files = []
                        reac_files = []
                        
                        for f in ascii_dir.iterdir():
                            if f.is_file() and f.suffix.lower() == '.txt':
                                fname = f.name.lower()
                                logging.info(f"  Checking file: {f.name} (lowercase: {fname})")
                                if 'demo' in fname:
                                    demo_files.append(f)
                                    logging.info(f"    Added as demo file")
                                elif 'drug' in fname:
                                    drug_files.append(f)
                                    logging.info(f"    Added as drug file")
                                elif 'reac' in fname:
                                    reac_files.append(f)
                                    logging.info(f"    Added as reac file")
                        
                        demo_file = demo_files[0] if demo_files else None
                        drug_file = drug_files[0] if drug_files else None
                        reac_file = reac_files[0] if reac_files else None
                        
                        if not all([demo_file, drug_file, reac_file]):
                            logging.warning(f"Missing files for quarter {quarter}:")
                            logging.warning(f"  DEMO: {demo_file}")
                            logging.warning(f"  DRUG: {drug_file}")
                            logging.warning(f"  REAC: {reac_file}")
                            continue
                            
                        # Process demographics
                        demo_df = self.process_file(demo_file, 'demographics')
                        if not demo_df.empty:
                            save_path = self.output_dir / f"{quarter}_demographics.txt"
                            demo_df.to_csv(save_path, sep='$', index=False, encoding='utf-8')
                        
                        # Process drugs
                        drug_df = self.process_file(drug_file, 'drugs')
                        if not drug_df.empty:
                            save_path = self.output_dir / f"{quarter}_drugs.txt"
                            drug_df.to_csv(save_path, sep='$', index=False, encoding='utf-8')
                        
                        # Process reactions
                        reac_df = self.process_file(reac_file, 'reactions')
                        if not reac_df.empty:
                            save_path = self.output_dir / f"{quarter}_reactions.txt"
                            reac_df.to_csv(save_path, sep='$', index=False, encoding='utf-8')
                        
                        logging.info(f"Successfully processed quarter {quarter}")
                        
                    except Exception as e:
                        logging.error(f"Error processing quarter {quarter}: {str(e)}")
                    finally:
                        pbar.update(1)
                        
        except Exception as e:
            logging.error(f"Error in process_all: {str(e)}")
            raise

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
                    'drugname': 'drugname',
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
                    df = pd.read_csv(file_path, sep='$', dtype=str, na_values=['', 'NA', 'NULL'], 
                                   keep_default_na=True, header=0, encoding='utf-8')
                except UnicodeDecodeError:
                    # If UTF-8 fails, try with latin1 encoding
                    df = pd.read_csv(file_path, sep='$', dtype=str, na_values=['', 'NA', 'NULL'],
                                   keep_default_na=True, header=0, encoding='latin1')
                
                # Rename columns according to mapping
                if data_type in column_mapping:
                    df = df.rename(columns=column_mapping[data_type])
                
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
                    df['drugname'] = df['drugname'].str.lower().str.strip()
                    df['drugname'] = df['drugname'].str.replace(r'\s+', ' ', regex=True)
                    df['drugname'] = df['drugname'].str.replace(r'\.$', '', regex=True)
                    df['drugname'] = df['drugname'].str.replace(r'\( ', '(', regex=True)
                    df['drugname'] = df['drugname'].str.replace(r' \)', ')', regex=True)
                    
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
