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
    
    def deduplicate_by_caseid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only the last record for each caseid.
        
        Matches R implementation:
        emo <- Demo[Demo[,.I%in%c(Demo[,.I[.N],by="caseid"]$V1)]]
        
        Args:
            df: DataFrame with potential duplicate caseids
            
        Returns:
            DataFrame with only last record per caseid
        """
        try:
            if 'caseid' not in df.columns:
                self.logger.warning("Cannot deduplicate by caseid: missing caseid column")
                return df
                
            # Get indices of last record for each caseid
            indices = df.groupby('caseid').tail(1).index
            
            # Keep only the selected rows
            df_deduped = df.loc[indices]
            
            # Log deduplication statistics
            total_rows = len(df)
            kept_rows = len(df_deduped)
            removed_rows = total_rows - kept_rows
            
            if removed_rows > 0:
                self.logger.info(f"Removed {removed_rows} duplicate caseid entries, keeping {kept_rows} unique entries")
                
                # Log some examples of removed duplicates for verification
                dupes = df[df.duplicated(subset=['caseid'], keep='last')].sort_values('caseid')
                if not dupes.empty:
                    sample_dupes = dupes.head(3)  # Show up to 3 examples
                    self.logger.debug("Sample of removed caseid duplicates:")
                    for _, row in sample_dupes.iterrows():
                        self.logger.debug(f"Caseid: {row['caseid']}")
            
            return df_deduped
            
        except Exception as e:
            self.logger.error(f"Error deduplicating by caseid: {str(e)}")
            return df

    def deduplicate_by_manufacturer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicated manufacturer IDs keeping most recent by FDA date.
        
        Matches R implementation:
        Demo <- Demo[order(fda_dt)]
        Demo <- Demo[Demo[,.I%in%c(Demo[,.I[.N],by=c("mfr_num","mfr_sndr")]$V1,
                                Demo[,which(is.na(mfr_num))],
                                Demo[,which(is.na(mfr_sndr))])]]
        
        Args:
            df: DataFrame with potential duplicate manufacturer IDs
            
        Returns:
            DataFrame with duplicates removed, keeping most recent by FDA date
        """
        try:
            if not all(col in df.columns for col in ['mfr_num', 'mfr_sndr', 'fda_dt']):
                self.logger.warning("Cannot deduplicate by manufacturer: missing required columns")
                return df
            
            # Convert fda_dt to numeric for sorting
            df = df.copy()
            df['fda_dt'] = pd.to_numeric(df['fda_dt'], errors='coerce')
            
            # Sort by FDA date
            df = df.sort_values('fda_dt')
            
            # Get indices to keep:
            # 1. Last record for each mfr_num, mfr_sndr combination
            valid_mfr = df.dropna(subset=['mfr_num', 'mfr_sndr'])
            keep_indices = valid_mfr.groupby(['mfr_num', 'mfr_sndr']).tail(1).index
            
            # 2. Records with NA mfr_num
            na_mfr_num = df[df['mfr_num'].isna()].index
            
            # 3. Records with NA mfr_sndr
            na_mfr_sndr = df[df['mfr_sndr'].isna()].index
            
            # Combine all indices to keep
            all_indices = pd.Index(keep_indices).union(na_mfr_num).union(na_mfr_sndr)
            
            # Keep only the selected rows
            df_deduped = df.loc[all_indices]
            
            # Log deduplication statistics
            total_rows = len(df)
            kept_rows = len(df_deduped)
            removed_rows = total_rows - kept_rows
            
            if removed_rows > 0:
                self.logger.info(f"Removed {removed_rows} duplicate manufacturer entries, keeping {kept_rows} entries")
                self.logger.info(f"Kept {len(keep_indices)} manufacturer combinations")
                self.logger.info(f"Kept {len(na_mfr_num)} records with NA mfr_num")
                self.logger.info(f"Kept {len(na_mfr_sndr)} records with NA mfr_sndr")
                
                # Log some examples of removed duplicates for verification
                dupes = df[df.duplicated(subset=['mfr_num', 'mfr_sndr'], keep='last')].sort_values(['mfr_num', 'mfr_sndr'])
                if not dupes.empty:
                    sample_dupes = dupes.head(3)  # Show up to 3 examples
                    self.logger.debug("Sample of removed manufacturer duplicates:")
                    for _, row in sample_dupes.iterrows():
                        self.logger.debug(
                            f"mfr_num: {row['mfr_num']}, "
                            f"mfr_sndr: {row['mfr_sndr']}, "
                            f"fda_dt: {row['fda_dt']}"
                        )
            
            return df_deduped
            
        except Exception as e:
            self.logger.error(f"Error deduplicating by manufacturer: {str(e)}")
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

    def identify_rule_based_duplicates(self, demo_df: pd.DataFrame, reac_df: pd.DataFrame, 
                                    drug_df: pd.DataFrame) -> pd.DataFrame:
        """Identify rule-based duplicates using two algorithms.
        
        Matches R implementation:
        1. Conservative algorithm (RB_duplicates) - checks exact matches in:
           - event_dt, sex, reporter_country, age_in_days, wt_in_kgs
           - list of PTs
           - list of drugs (PS, SS, IC)
           
        2. Speculative algorithm (RB_duplicates_only_susp) - checks exact matches in:
           - event_dt, sex, reporter_country, age_in_days, wt_in_kgs
           - list of PTs
           - list of suspected drugs only
        
        Args:
            demo_df: Demographics DataFrame
            reac_df: Reactions DataFrame
            drug_df: Drug DataFrame
            
        Returns:
            Demographics DataFrame with duplicate flags
        """
        try:
            # 1. Prepare reaction data - concatenate PTs
            temp_reac = (reac_df.sort_values('pt')
                        .groupby('primaryid')['pt']
                        .agg(lambda x: '; '.join(x))
                        .reset_index()
                        .drop_duplicates())
            
            # 2. Prepare drug data by role code
            # Primary Suspect (PS)
            temp_drug_ps = (drug_df[drug_df['role_cod'] == 'PS']
                          .sort_values('substance')
                          .groupby('primaryid')['substance']
                          .agg(lambda x: '; '.join(x))
                          .reset_index()
                          .rename(columns={'substance': 'PS'})
                          .drop_duplicates())
            
            # Secondary Suspect (SS)
            temp_drug_ss = (drug_df[drug_df['role_cod'] == 'SS']
                          .sort_values('substance')
                          .groupby('primaryid')['substance']
                          .agg(lambda x: '; '.join(x))
                          .reset_index()
                          .rename(columns={'substance': 'SS'})
                          .drop_duplicates())
            
            # Interacting/Concomitant (I/C)
            temp_drug_ic = (drug_df[drug_df['role_cod'].isin(['I', 'C'])]
                         .sort_values('substance')
                         .groupby('primaryid')['substance']
                         .agg(lambda x: '; '.join(x))
                         .reset_index()
                         .rename(columns={'substance': 'IC'})
                         .drop_duplicates())
            
            # All Suspected (PS + SS)
            temp_drug_suspected = (drug_df[drug_df['role_cod'].isin(['PS', 'SS'])]
                                .sort_values('substance')
                                .groupby('primaryid')['substance']
                                .agg(lambda x: '; '.join(x))
                                .reset_index()
                                .rename(columns={'substance': 'suspected'})
                                .drop_duplicates())
            
            # 3. Merge all temporary dataframes
            merge_cols = ['event_dt', 'sex', 'reporter_country', 'age_in_days', 
                         'wt_in_kgs', 'primaryid', 'fda_dt']
            temp = demo_df[merge_cols].merge(temp_reac, on='primaryid')
            temp = temp.merge(temp_drug_ps, on='primaryid', how='left')
            temp = temp.merge(temp_drug_ss, on='primaryid', how='left')
            temp = temp.merge(temp_drug_ic, on='primaryid', how='left')
            temp = temp.merge(temp_drug_suspected, on='primaryid', how='left')
            
            # Sort by FDA date
            temp = temp.sort_values('fda_dt')
            
            # 4. Conservative algorithm (all drugs)
            complete_duplicates = ['event_dt', 'sex', 'reporter_country', 'age_in_days',
                                 'wt_in_kgs', 'pt', 'PS', 'SS', 'IC']
            
            # Create duplicate groups
            temp['DUP_ID'] = temp.groupby(complete_duplicates).ngroup()
            
            # Find singlets and duplicates
            dup_counts = temp.groupby('DUP_ID').size().reset_index(name='N')
            singlets_pids = temp[temp['DUP_ID'].isin(dup_counts[dup_counts['N'] == 1]['DUP_ID'])]['primaryid']
            duplicates = temp[~temp['primaryid'].isin(singlets_pids)]
            
            # Keep last record of each duplicate group
            duplicates_pids = (duplicates.groupby('DUP_ID')
                             .apply(lambda x: x.iloc[-1])['primaryid'])
            
            # Add RB_duplicates flag
            demo_df['RB_duplicates'] = ~demo_df['primaryid'].isin(pd.concat([singlets_pids, duplicates_pids]))
            
            # 5. Speculative algorithm (only suspected drugs)
            complete_duplicates = ['event_dt', 'sex', 'reporter_country', 'age_in_days',
                                 'wt_in_kgs', 'pt', 'suspected']
            
            # Create duplicate groups
            temp['DUP_ID'] = temp.groupby(complete_duplicates).ngroup()
            
            # Find singlets and duplicates
            dup_counts = temp.groupby('DUP_ID').size().reset_index(name='N')
            singlets_pids = temp[temp['DUP_ID'].isin(dup_counts[dup_counts['N'] == 1]['DUP_ID'])]['primaryid']
            duplicates = temp[~temp['primaryid'].isin(singlets_pids)]
            
            # Keep last record of each duplicate group
            duplicates_pids = (duplicates.groupby('DUP_ID')
                             .apply(lambda x: x.iloc[-1])['primaryid'])
            
            # Add RB_duplicates_only_susp flag
            demo_df['RB_duplicates_only_susp'] = ~demo_df['primaryid'].isin(pd.concat([singlets_pids, duplicates_pids]))
            
            # Log statistics
            total_cases = len(demo_df)
            conservative_dups = demo_df['RB_duplicates'].sum()
            speculative_dups = demo_df['RB_duplicates_only_susp'].sum()
            
            self.logger.info(f"Rule-based deduplication statistics:")
            self.logger.info(f"  Total cases: {total_cases}")
            self.logger.info(f"  Conservative duplicates: {conservative_dups} ({100*conservative_dups/total_cases:.1f}%)")
            self.logger.info(f"  Speculative duplicates: {speculative_dups} ({100*speculative_dups/total_cases:.1f}%)")
            
            return demo_df
            
        except Exception as e:
            self.logger.error(f"Error in rule-based deduplication: {str(e)}")
            return demo_df
