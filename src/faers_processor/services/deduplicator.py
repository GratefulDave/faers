"""Service for handling deduplication of FAERS data."""

import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple

class FAERSDeduplicator:
    """Service for deduplicating FAERS data following R implementation."""
    
    def __init__(self):
        """Initialize the deduplicator service."""
        self.logger = logging.getLogger(__name__)
    
    def deduplicate_primaryids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicated primaryids keeping only the most recent quarter entry.
        
        Matches R implementation:
        Demo <- Demo[Demo[,.I[quarter==last(quarter)],by=primaryid]$V1]
        
        Args:
            df: DataFrame with potential duplicate primaryids
            
        Returns:
            DataFrame with duplicates removed, keeping most recent quarter
        """
        try:
            if 'primaryid' not in df.columns or 'quarter' not in df.columns:
                self.logger.warning("Cannot remove duplicates: missing required columns 'primaryid' or 'quarter'")
                return df
                
            # Get indices of rows to keep (most recent quarter for each primaryid)
            indices = df.groupby('primaryid')['quarter'].idxmax()
            
            # Keep only the selected rows
            df_deduped = df.loc[indices]
            
            # Log duplicate removal statistics
            total_rows = len(df)
            kept_rows = len(df_deduped)
            removed_rows = total_rows - kept_rows
            
            if removed_rows > 0:
                self.logger.info(f"Removed {removed_rows} duplicate primaryid entries, keeping {kept_rows} unique entries")
                
                # Log some examples of removed duplicates for verification
                dupes = df[df.duplicated(subset=['primaryid'], keep=False)].sort_values(['primaryid', 'quarter'])
                if not dupes.empty:
                    sample_dupes = dupes.groupby('primaryid').head(2).head(6)  # Show up to 3 pairs of duplicates
                    self.logger.debug("Sample of removed duplicates (showing primaryid, quarter, caseid):")
                    for _, group in sample_dupes.groupby('primaryid'):
                        self.logger.debug(f"\nPrimaryid: {group['primaryid'].iloc[0]}")
                        for _, row in group.iterrows():
                            self.logger.debug(f"Quarter: {row['quarter']}, Caseid: {row.get('caseid', 'N/A')}")
        
            return df_deduped
            
        except Exception as e:
            self.logger.error(f"Error removing duplicate primaryids: {str(e)}")
            return df
    
    def deduplicate_dataset(self, input_path: Path, output_path: Path) -> None:
        """Deduplicate an entire dataset, preserving most recent entries.
        
        Args:
            input_path: Path to input dataset
            output_path: Path to save deduplicated dataset
        """
        try:
            # Read the dataset
            df = pd.read_pickle(input_path)
            
            # Perform deduplication
            df_deduped = self.deduplicate_primaryids(df)
            
            # Save deduplicated dataset
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_deduped.to_pickle(output_path)
            
            self.logger.info(f"Successfully deduplicated dataset: {input_path.name}")
            
        except Exception as e:
            self.logger.error(f"Error deduplicating dataset {input_path}: {str(e)}")
            raise
