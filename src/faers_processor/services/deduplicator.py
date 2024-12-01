"""Deduplication of FAERS data."""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


class Deduplicator:
    """Handles deduplication of FAERS data files."""

    def __init__(self, data_dir: Path):
        """Initialize deduplicator.
        
        Args:
            data_dir: Directory containing processed FAERS data
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

    def deduplicate_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate demographics data.
        
        Args:
            df: Demographics DataFrame
            
        Returns:
            Deduplicated DataFrame
        """
        # Sort by date and keep most recent
        if not df.empty and 'date' in df.columns:
            df = df.sort_values('date', ascending=False)
            
        # Drop duplicates based on key fields
        key_fields = ['primaryid', 'caseid', 'age', 'sex', 'reporter_country']
        key_fields = [f for f in key_fields if f in df.columns]
        
        return df.drop_duplicates(subset=key_fields, keep='first')

    def deduplicate_drugs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate drugs data.
        
        Args:
            df: Drugs DataFrame
            
        Returns:
            Deduplicated DataFrame
        """
        # Sort by drug sequence to keep primary suspect drugs
        if not df.empty and 'drug_seq' in df.columns:
            df = df.sort_values('drug_seq')
            
        # Drop duplicates based on key fields
        key_fields = ['primaryid', 'caseid', 'drug_name', 'route', 'dose_amt', 'dose_unit']
        key_fields = [f for f in key_fields if f in df.columns]
        
        return df.drop_duplicates(subset=key_fields, keep='first')

    def deduplicate_reactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate reactions data.
        
        Args:
            df: Reactions DataFrame
            
        Returns:
            Deduplicated DataFrame
        """
        # Drop duplicates based on key fields
        key_fields = ['primaryid', 'caseid', 'pt', 'outcome']
        key_fields = [f for f in key_fields if f in df.columns]
        
        return df.drop_duplicates(subset=key_fields, keep='first')

    def deduplicate_file(self, file_path: Path) -> None:
        """Deduplicate a single parquet file.
        
        Args:
            file_path: Path to parquet file
        """
        try:
            # Read parquet file
            df = pd.read_parquet(file_path)
            
            # Determine file type and apply appropriate deduplication
            if 'demographics' in file_path.stem.lower():
                df_dedup = self.deduplicate_demographics(df)
            elif 'drugs' in file_path.stem.lower():
                df_dedup = self.deduplicate_drugs(df)
            elif 'reactions' in file_path.stem.lower():
                df_dedup = self.deduplicate_reactions(df)
            else:
                logging.warning(f"Unknown file type: {file_path}")
                return
                
            # Save deduplicated file
            output_path = file_path.parent / f"{file_path.stem}_dedup.parquet"
            df_dedup.to_parquet(output_path, engine='pyarrow', index=False)
            
            # Log results
            reduction = ((len(df) - len(df_dedup)) / len(df)) * 100 if len(df) > 0 else 0
            logging.info(f"Deduplicated {file_path.name}: {reduction:.1f}% reduction")
            
        except Exception as e:
            logging.error(f"Error deduplicating {file_path}: {str(e)}")
            raise

    def deduplicate_all(self) -> None:
        """Deduplicate all processed FAERS data files."""
        try:
            # Find all parquet files
            parquet_files = list(self.data_dir.glob('*.parquet'))
            if not parquet_files:
                logging.warning("No parquet files found to deduplicate")
                return
                
            # Process each file
            with tqdm(total=len(parquet_files), desc="Deduplicating files") as pbar:
                for file_path in parquet_files:
                    try:
                        if not file_path.stem.endswith('_dedup'):  # Skip already deduplicated files
                            self.deduplicate_file(file_path)
                    except Exception as e:
                        logging.error(f"Error processing {file_path}: {str(e)}")
                    finally:
                        pbar.update(1)
        except Exception as e:
            logging.error(f"Error in deduplicate_all: {str(e)}")
            raise
