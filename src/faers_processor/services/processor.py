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
            quarters = [d.name for d in self.data_dir.iterdir() 
                       if d.is_dir() and d.name.lower() != 'clean' 
                       and any(c.isdigit() for c in d.name)]  # Only include quarter directories
            
            if not quarters:
                logging.warning(f"No quarters found to process in {self.data_dir}")
                return
                
            logging.info(f"Found {len(quarters)} quarters to process: {quarters}")
            
            # Process each quarter
            with tqdm(total=len(quarters), desc="Processing quarters") as pbar:
                for quarter in quarters:
                    try:
                        quarter_dir = self.data_dir / quarter
                        logging.info(f"Processing quarter {quarter} from {quarter_dir}")
                        
                        # Look for files case-insensitively
                        demo_file = next((f for f in quarter_dir.glob("*.txt") 
                                        if "DEMO" in f.name.upper()), None)
                        drug_file = next((f for f in quarter_dir.glob("*.txt") 
                                        if "DRUG" in f.name.upper()), None)
                        reac_file = next((f for f in quarter_dir.glob("*.txt") 
                                        if "REAC" in f.name.upper()), None)
                        
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
            # Common read options
            read_opts = {
                'filepath_or_buffer': str(file_path),  # Convert Path to string
                'sep': '$',
                'dtype': str,
                'na_values': ['', 'NA', 'NULL'],
                'keep_default_na': True
            }

            # Read data with optimized settings
            if self.use_dask:
                df = dd.read_csv(
                    **read_opts,
                    blocksize=self.chunk_size * 1024
                )
            else:
                chunks = pd.read_csv(
                    **read_opts,
                    chunksize=self.chunk_size
                )
                # Concatenate chunks into single DataFrame
                df = pd.concat(chunks, ignore_index=True)

            # Process based on data type
            if data_type == 'demographics':
                result = self.standardizer.process_demographics(df)
            elif data_type == 'drugs':
                result = self.standardizer.process_drugs(df)
            else:  # reactions
                result = self.standardizer.process_reactions(df)

            # Compute if using Dask
            if self.use_dask:
                result = result.compute()

            return result

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            return pd.DataFrame()
