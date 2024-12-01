"""
FAERS Data Standardization Module

This module provides functionality for standardizing and processing FDA Adverse Event 
Reporting System (FAERS) data. It implements comprehensive data cleaning, standardization,
and organization methods following the original R code's logic.

Key Features:
- Data standardization (age, weight, countries, etc.)
- Duplicate detection and handling
- Data organization and saving
- Comprehensive logging

Author: DiAna Team
License: MIT
"""

import logging
import os
import pandas as pd
from typing import List, Set, Dict, Tuple, Optional

class DataStandardizer:
    """
    A class for standardizing and processing FAERS data.
    
    This class provides methods for cleaning, standardizing, and organizing FAERS data,
    including handling of demographics, drugs, reactions, and other related information.
    It follows the logic of the original R implementation while providing Pythonic interfaces.
    
    Attributes:
        None
        
    Methods:
        handle_manufacturer_records: Process manufacturer records with proper sorting
        identify_special_cases: Flag pre-marketing and literature cases
        identify_duplicates: Detect and mark duplicate cases
        save_processed_data: Save processed data in quarterly directories
    """
    
    def handle_manufacturer_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process manufacturer records by sorting by FDA date and handling manufacturer numbers.
        
        This method implements the logic from the original R code:
        Demo <- Demo[order(fda_dt)]
        Demo <- Demo[Demo[,.I%in%c(Demo[,.I[.N],by=c("mfr_num","mfr_sndr")]$V1,
                               Demo[,which(is.na(mfr_num))],
                               Demo[,which(is.na(mfr_sndr))])]]
        
        Args:
            df: DataFrame with manufacturer information including fda_dt, mfr_num, mfr_sndr
        
        Returns:
            DataFrame with processed manufacturer records
            
        Example:
            >>> standardizer = DataStandardizer()
            >>> processed_df = standardizer.handle_manufacturer_records(demo_df)
        """
        df = df.copy()
        
        # Sort by FDA date
        df = df.sort_values('fda_dt')
        
        # Get indices for records to keep:
        # 1. Last record for each manufacturer number/sender combination
        # 2. Records with missing manufacturer number
        # 3. Records with missing manufacturer sender
        
        # Get last records for each mfr_num/mfr_sndr combination
        last_mfr_records = df.groupby(['mfr_num', 'mfr_sndr']).tail(1).index
        
        # Get records with missing manufacturer info
        missing_mfr_num = df[df['mfr_num'].isna()].index
        missing_mfr_sndr = df[df['mfr_sndr'].isna()].index
        
        # Combine all indices to keep
        indices_to_keep = last_mfr_records.union(missing_mfr_num).union(missing_mfr_sndr)
        
        # Keep only the selected records
        df = df.loc[indices_to_keep]
        
        # Log processing results
        logging.info(f"Records after manufacturer processing: {len(df)}")
        logging.info(f"Records with missing mfr_num: {len(missing_mfr_num)}")
        logging.info(f"Records with missing mfr_sndr: {len(missing_mfr_sndr)}")
        logging.info(f"Unique manufacturer combinations: {len(last_mfr_records)}")
        
        return df

    def identify_special_cases(self, demo_df: pd.DataFrame, drug_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify pre-marketing cases and literature references.
        
        This method implements the logic from the original R code:
        Demo[,premarketing:=primaryid%in%Drug[trial==TRUE]$primaryid]
        Demo[,literature:=!is.na(lit_ref)]
        
        Args:
            demo_df: Demographics DataFrame with lit_ref column
            drug_df: Drug DataFrame with trial information
        
        Returns:
            DataFrame with pre-marketing and literature flags added
            
        Example:
            >>> standardizer = DataStandardizer()
            >>> demo_df = standardizer.identify_special_cases(demo_df, drug_df)
        """
        demo_df = demo_df.copy()
        
        # Identify pre-marketing cases (those with trial drugs)
        trial_cases = set(drug_df[drug_df['trial'] == True]['primaryid'].unique())
        demo_df['premarketing'] = demo_df['primaryid'].isin(trial_cases)
        
        # Identify literature cases (those with literature references)
        demo_df['literature'] = ~demo_df['lit_ref'].isna()
        
        # Log results
        premarketing_count = demo_df['premarketing'].sum()
        literature_count = demo_df['literature'].sum()
        total_cases = len(demo_df)
        
        logging.info(f"Total cases: {total_cases}")
        logging.info(f"Pre-marketing cases: {premarketing_count} ({premarketing_count/total_cases*100:.2f}%)")
        logging.info(f"Literature cases: {literature_count} ({literature_count/total_cases*100:.2f}%)")
        
        return demo_df

    def identify_duplicates(self, demo_df: pd.DataFrame, reac_df: pd.DataFrame, drug_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify duplicates based on complete information and suspected drugs only.
        
        This method implements duplicate detection following the original R code's logic,
        considering both complete case information and suspected drugs only. It adds two
        flags to the demographics DataFrame:
        - RB_duplicates: Duplicates based on complete information
        - RB_duplicates_only_susp: Duplicates based only on suspected drugs
        
        The duplicate detection considers the following fields:
        Complete duplicates: event_dt, sex, reporter_country, age_in_days, wt_in_kgs,
                           pt, PS, SS, IC
        Suspected duplicates: event_dt, sex, reporter_country, age_in_days, wt_in_kgs,
                            pt, suspected
        
        Args:
            demo_df: Demographics DataFrame
            reac_df: Reactions DataFrame
            drug_df: Drug DataFrame
        
        Returns:
            DataFrame with duplicate flags added
            
        Example:
            >>> standardizer = DataStandardizer()
            >>> demo_df = standardizer.identify_duplicates(demo_df, reac_df, drug_df)
        """
        def _get_drug_groups(df, role_codes):
            return (df[df['role_cod'].isin(role_codes)]
                    .sort_values('substance')
                    .groupby('primaryid')['substance']
                    .agg('; '.join)
                    .reset_index())
        
        def _find_duplicates(temp_df, group_cols):
            temp_df = temp_df.copy()
            temp_df['DUP_ID'] = temp_df.groupby(group_cols).ngroup()
            dup_counts = temp_df.groupby('DUP_ID').size()
            singlets = temp_df[temp_df['DUP_ID'].isin(dup_counts[dup_counts == 1].index)]['primaryid']
            duplicates = temp_df[~temp_df['primaryid'].isin(singlets)]
            dup_pids = duplicates.groupby('DUP_ID').last()['primaryid']
            return pd.concat([singlets, dup_pids])
        
        # Prepare base dataframe
        temp = demo_df.copy()
        
        # Add reactions
        temp = temp.merge(
            reac_df.sort_values('pt')
              .groupby('primaryid')['pt']
              .agg('; '.join)
              .reset_index(),
            on='primaryid'
        )
        
        # Add drug groups
        for role, codes in [('PS', ['PS']), ('SS', ['SS']), 
                           ('IC', ['I', 'C']), ('suspected', ['PS', 'SS'])]:
            temp = temp.merge(
                _get_drug_groups(drug_df, codes).rename(columns={'substance': role}),
                on='primaryid',
                how='left'
            )
        
        # Sort by FDA date
        temp = temp.sort_values('fda_dt')
        
        # Find duplicates with complete information
        complete_cols = ['event_dt', 'sex', 'reporter_country', 'age_in_days', 
                        'wt_in_kgs', 'pt', 'PS', 'SS', 'IC']
        keep_ids = _find_duplicates(temp, complete_cols)
        demo_df['RB_duplicates'] = ~demo_df['primaryid'].isin(keep_ids)
        
        # Find duplicates with suspected drugs only
        suspect_cols = ['event_dt', 'sex', 'reporter_country', 'age_in_days', 
                       'wt_in_kgs', 'pt', 'suspected']
        keep_ids = _find_duplicates(temp, suspect_cols)
        demo_df['RB_duplicates_only_susp'] = ~demo_df['primaryid'].isin(keep_ids)
        
        return demo_df

    def save_processed_data(self, data_directory: str, demo_df: pd.DataFrame, drug_df: pd.DataFrame, 
                          reac_df: pd.DataFrame, outc_df: pd.DataFrame, indi_df: pd.DataFrame, 
                          ther_df: pd.DataFrame, drug_info_df: pd.DataFrame, rpsr_df: pd.DataFrame,
                          meddra_primary_path: str) -> None:
        """
        Save processed data into quarterly directories with proper formatting.
        
        This method implements the data saving logic from the original R code,
        organizing processed data into quarterly directories (e.g., "Data/23Q1").
        It handles proper formatting of categorical variables and maintains data
        relationships across files.
        
        Files saved:
        - DEMO_SUPP.pkl: Supplementary demographic information
        - DEMO.pkl: Core demographic information
        - DRUG_NAME.pkl: Drug names and product information
        - DRUG.pkl: Drug substances and roles
        - REAC.pkl: Adverse reactions
        - OUTC.pkl: Outcomes
        - INDI.pkl: Indications
        - THER.pkl: Therapy information
        - DOSES.pkl: Dosage information
        - DRUG_SUPP.pkl: Supplementary drug information
        
        Args:
            data_directory: Target directory for saving data (e.g., "Data/23Q1")
            demo_df: Demographics DataFrame
            drug_df: Drug DataFrame
            reac_df: Reactions DataFrame
            outc_df: Outcomes DataFrame
            indi_df: Indications DataFrame
            ther_df: Therapy DataFrame
            drug_info_df: Drug information DataFrame
            rpsr_df: Report source DataFrame
            meddra_primary_path: Path to MedDRA primary dictionary
            
        Example:
            >>> standardizer = DataStandardizer()
            >>> standardizer.save_processed_data(
            ...     data_directory="Data/23Q1",
            ...     demo_df=demo_df,
            ...     drug_df=drug_df,
            ...     reac_df=reac_df,
            ...     outc_df=outc_df,
            ...     indi_df=indi_df,
            ...     ther_df=ther_df,
            ...     drug_info_df=drug_info_df,
            ...     rpsr_df=rpsr_df,
            ...     meddra_primary_path="External Sources/Dictionaries/MedDRA/meddra_primary.csv"
            ... )
        """
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(data_directory, exist_ok=True)
        
        # Read MedDRA primary dictionary
        meddra_primary = pd.read_csv(meddra_primary_path, sep=';')
        meddra_primary = meddra_primary.sort_values(['soc', 'hlgt', 'hlt', 'pt'])
        
        # Save DEMO_SUPP
        demo_supp = demo_df[['primaryid', 'caseid', 'caseversion', 'i_f_cod', 'auth_num', 
                             'e_sub', 'lit_ref', 'rept_dt', 'to_mfr', 'mfr_sndr', 
                             'mfr_num', 'mfr_dt', 'quarter']].copy()
        demo_supp = pd.merge(rpsr_df[['primaryid', 'rpsr_cod']], demo_supp, on='primaryid', how='right')
        demo_supp['rpsr_cod'] = pd.Categorical(demo_supp['rpsr_cod'])
        demo_supp.to_pickle(os.path.join(data_directory, 'DEMO_SUPP.pkl'))
        
        # Save DEMO
        demo = demo_df[['primaryid', 'sex', 'age_in_days', 'wt_in_kgs', 'occr_country',
                        'event_dt', 'occp_cod', 'reporter_country', 'rept_cod', 
                        'init_fda_dt', 'fda_dt', 'premarketing', 'literature']].copy()
        demo.to_pickle(os.path.join(data_directory, 'DEMO.pkl'))
        
        # Save DRUG_NAME
        drug_name = drug_df[['primaryid', 'drug_seq', 'drugname', 'prod_ai']].drop_duplicates()
        drug_name = pd.merge(
            drug_info_df[['primaryid', 'drug_seq', 'val_vbm', 'nda_num']],
            drug_name,
            on=['primaryid', 'drug_seq']
        )
        drug_name.to_pickle(os.path.join(data_directory, 'DRUG_NAME.pkl'))
        
        # Save DRUG
        drug = drug_df[['primaryid', 'drug_seq', 'Substance', 'role_cod']].drop_duplicates()
        drug = drug.rename(columns={'Substance': 'substance'})
        drug['role_cod'] = pd.Categorical(drug['role_cod'], 
                                        categories=['C', 'I', 'SS', 'PS'],
                                        ordered=True)
        drug.to_pickle(os.path.join(data_directory, 'DRUG.pkl'))
        
        # Save REAC
        reac = reac_df[['primaryid', 'pt', 'drug_rec_act']].copy()
        reac['pt'] = pd.Categorical(reac['pt'], 
                                   categories=meddra_primary['pt'].tolist(),
                                   ordered=True)
        reac['drug_rec_act'] = pd.Categorical(reac['drug_rec_act'],
                                             categories=meddra_primary['pt'].tolist(),
                                             ordered=True)
        reac.to_pickle(os.path.join(data_directory, 'REAC.pkl'))
        
        # Save OUTC
        outc = outc_df[['primaryid', 'outc_cod']].drop_duplicates()
        outc['outc_cod'] = pd.Categorical(outc['outc_cod'],
                                         categories=['OT', 'CA', 'HO', 'RI', 'DS', 'LT', 'DE'],
                                         ordered=True)
        outc.to_pickle(os.path.join(data_directory, 'OUTC.pkl'))
        
        # Save INDI
        indi = indi_df[['primaryid', 'drug_seq', 'indi_pt']].drop_duplicates()
        indi['indi_pt'] = pd.Categorical(indi['indi_pt'],
                                        categories=meddra_primary['pt'].tolist(),
                                        ordered=True)
        indi.to_pickle(os.path.join(data_directory, 'INDI.pkl'))
        
        # Save THER
        ther = ther_df[['primaryid', 'drug_seq', 'start_dt', 'dur_in_days',
                        'end_dt', 'time_to_onset', 'event_dt']].drop_duplicates()
        ther.to_pickle(os.path.join(data_directory, 'THER.pkl'))
        
        # Save DOSES
        doses = drug_info_df[['primaryid', 'drug_seq', 'dose_vbm', 'cum_dose_unit',
                             'cum_dose_chr', 'dose_amt', 'dose_unit', 
                             'dose_freq']].drop_duplicates()
        doses.to_pickle(os.path.join(data_directory, 'DOSES.pkl'))
        
        # Save DRUG_SUPP
        drug_supp = drug_info_df[['primaryid', 'drug_seq', 'route', 'dose_form',
                                 'dechal', 'rechal', 'lot_num', 'exp_dt']].drop_duplicates()
        drug_supp['dose_form'] = pd.Categorical(drug_supp['dose_form'])
        drug_supp.to_pickle(os.path.join(data_directory, 'DRUG_SUPP.pkl'))
        
        logging.info(f"All processed data saved to {data_directory}")
        logging.info(f"Number of cases in final dataset: {len(demo)}")

    def remove_deleted_cases(self, df: pd.DataFrame, faers_list_path: str = "Clean Data/faers_list.csv") -> pd.DataFrame:
        """
        Remove deleted cases from the dataset based on deleted files.
        
        Args:
            df: DataFrame with case data
            faers_list_path: Path to CSV containing list of FAERS files
        
        Returns:
            DataFrame with deleted cases removed
        """
        df = df.copy()
        
        # Read FAERS file list
        faers_list = pd.read_csv(faers_list_path, sep=';')['x'].tolist()
        
        # Filter for deleted files
        deleted_files = [f for f in faers_list if 'deleted' in f.lower()]
        
        if not deleted_files:
            logging.info("No deleted files found")
            return df
        
        # Read and combine all deleted case IDs
        all_deleted_cases = []
        for file in deleted_files:
            try:
                # Read deleted cases file, skipping first line, using $ as separator
                deleted_cases = pd.read_csv(
                    file,
                    sep='$',
                    skiprows=1,
                    names=['caseid'],
                    comment=None,
                    quoting=3  # QUOTE_NONE
                )
                all_deleted_cases.append(deleted_cases)
            except Exception as e:
                logging.error(f"Error reading deleted cases file {file}: {str(e)}")
        
        if not all_deleted_cases:
            logging.warning("No deleted cases found in files")
            return df
        
        # Combine all deleted cases and remove duplicates
        deleted_cases = pd.concat(all_deleted_cases, ignore_index=True).drop_duplicates()
        
        # Log number of cases to be removed
        initial_count = len(df)
        df = df[~df['caseid'].isin(deleted_cases['caseid'])]
        removed_count = initial_count - len(df)
        
        logging.info(f"Removed {removed_count} deleted cases")
        logging.info(f"Remaining cases: {len(df)}")
        
        return df

    def remove_duplicate_primaryids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate primary IDs, keeping only the most recent quarter for each ID.
        
        Args:
            df: DataFrame with primaryid and quarter columns
        
        Returns:
            DataFrame with duplicates removed, keeping most recent quarter
        """
        df = df.copy()
        
        # Log initial state
        initial_count = len(df)
        initial_unique_ids = df['primaryid'].nunique()
        logging.info(f"Initial records: {initial_count}")
        logging.info(f"Initial unique primary IDs: {initial_unique_ids}")
        
        # Group by primaryid and keep only the rows from the last quarter
        df = df.loc[df.groupby('primaryid')['quarter'].idxmax()]
        
        # Log results
        final_count = len(df)
        removed_count = initial_count - final_count
        logging.info(f"Removed {removed_count} duplicate records")
        logging.info(f"Remaining records: {final_count}")
        
        return df

    def finalize_demo_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Finalize demographic data processing by handling case versions and converting column types.
        
        Args:
            df: DataFrame with demographic data
        
        Returns:
            DataFrame with latest case versions and categorical columns
        """
        df = df.copy()
        
        # Keep only the last record for each caseid (matches R's Demo[,.I%in%c(Demo[,.I[.N],by="caseid"]$V1)])
        df = df.loc[df.groupby('caseid').tail(1).index]
        
        # Columns to convert to categorical
        categorical_cols = [
            'caseversion', 'sex', 'quarter', 'i_f_cod', 'rept_cod',
            'occp_cod', 'e_sub', 'age_grp', 'occr_country',
            'reporter_country'
        ]
        
        # Convert specified columns to categorical
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
                
                # Log category information
                categories = df[col].value_counts()
                logging.info(f"\nCategories for {col}:")
                for cat, count in categories.items():
                    logging.info(f"{cat}: {count}")
        
        # Log final record count
        logging.info(f"\nFinal record count: {len(df)}")
        logging.info(f"Unique cases: {df['caseid'].nunique()}")
        
        return df

    def standardize_pt(self, term: str) -> str:
        """
        Standardize a term to its Preferred Term (PT) using multiple approaches.
        
        Args:
            term: The term to standardize
        
        Returns:
            Standardized PT or original term if standardization fails
        """
        if pd.isna(term) or not term:
            return term
        
        self.standardization_stats['total_terms'] += 1
        term_lower = term.lower().strip()
        
        # Direct PT match
        if hasattr(self, 'pt_data'):
            pt_match = self.pt_data[self.pt_data['pt_name'].str.lower() == term_lower]
            if not pt_match.empty:
                self.standardization_stats['direct_pt_matches'] += 1
                return pt_match.iloc[0]['pt_name']
        
        # LLT to PT translation
        if term_lower in self.pt_to_llt_map:
            self.standardization_stats['llt_translations'] += 1
            return self.pt_to_llt_map[term_lower]
        
        # Manual fix
        if term_lower in self.manual_pt_fixes:
            self.standardization_stats['manual_fixes'] += 1
            return self.manual_pt_fixes[term_lower]
        
        # Log unstandardized term
        self.standardization_stats['unstandardized'] += 1
        self._log_unstandardized_term(term)
        return term

    def _log_unstandardized_term(self, term: str):
        """Log unstandardized terms for future manual review."""
        unstandardized_file = self.external_dir / 'manual_fix' / 'unstandardized_pts.csv'
        
        # Create or append to unstandardized terms file
        mode = 'a' if unstandardized_file.exists() else 'w'
        header = not unstandardized_file.exists()
        
        with open(unstandardized_file, mode, encoding='utf-8') as f:
            if header:
                f.write('term,frequency\n')
            f.write(f'{term},1\n')

    def get_standardization_stats(self) -> Dict[str, int]:
        """Get statistics about term standardization."""
        return self.standardization_stats

    def standardize_sex(self, df: pd.DataFrame, col: str = 'sex') -> pd.DataFrame:
        """Standardize sex values to F/M."""
        df = df.copy()
        df.loc[~df[col].isin(['F', 'M']), col] = np.nan
        return df
    
    def standardize_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize age values to days and years, handling different time units.
        
        Args:
            df: DataFrame containing age data with 'age' and 'age_cod' columns
        
        Returns:
            DataFrame with standardized age columns
        """
        df = df.copy()
        
        # Define age unit conversion factors
        age_conversion = {
            'DEC': 3650,          # Decade to days
            'YR': 365,            # Year to days
            'MON': 30.41667,      # Month to days
            'WK': 7,              # Week to days
            'DY': 1,              # Days
            'HR': 0.00011415525114155251,    # Hour to days
            'MIN': 1.9025875190259e-06,      # Minute to days
            'SEC': 3.1709791983764586e-08    # Second to days
        }
        
        # Create age_corrector column based on age_cod
        df['age_corrector'] = df['age_cod'].map(age_conversion)
        # Default to years (365) when age_cod is NA
        df.loc[df['age_cod'].isna(), 'age_corrector'] = 365
        
        # Convert age to numeric, taking absolute value, and multiply by correction factor
        df['age_in_days'] = np.round(
            np.abs(pd.to_numeric(df['age'], errors='coerce')) * 
            df['age_corrector']
        )
        
        # Handle plausibility correction
        # If age > 122 years (maximum recorded human age) and not in decades, set to NA
        max_age_days = 122 * 365
        df.loc[
            (df['age_in_days'] > max_age_days) & 
            (df['age_cod'] != 'DEC'),
            'age_in_days'
        ] = np.nan
        
        # If age > 122 years and in decades, convert back from decades
        decade_mask = (
            (df['age_in_days'] > max_age_days) & 
            (df['age_cod'] == 'DEC')
        )
        df.loc[decade_mask, 'age_in_days'] = (
            df.loc[decade_mask, 'age_in_days'] / 
            df.loc[decade_mask, 'age_corrector']
        )
        
        # Convert to years
        df['age_in_years'] = np.round(df['age_in_days'] / 365)
        
        # Drop intermediate columns
        df = df.drop(columns=['age_corrector', 'age', 'age_cod'])
        
        # Log summary statistics
        logging.info("Age in years summary statistics:")
        logging.info(f"Mean: {df['age_in_years'].mean():.2f}")
        logging.info(f"Median: {df['age_in_years'].median():.2f}")
        logging.info(f"Min: {df['age_in_years'].min():.2f}")
        logging.info(f"Max: {df['age_in_years'].max():.2f}")
        
        return df
    
    def standardize_weight(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]:
        """
        Standardize weight values to kilograms, handling different units.
        
        Args:
            df: DataFrame with weight data (wt and wt_cod columns)
        
        Returns:
            DataFrame with standardized weight in kilograms and a matplotlib figure
        """
        df = df.copy()
        
        # Initialize weight corrector with NA
        df['wt_corrector'] = np.nan
        
        # Define weight unit conversion factors
        weight_conversions = {
            'LBS': 0.453592,  # Pounds to kg
            'IB': 0.453592,   # Pounds to kg
            'KG': 1,          # Already in kg
            'KGS': 1,         # Already in kg
            'GMS': 0.001,     # Grams to kg
            'MG': 1e-06       # Milligrams to kg
        }
        
        # Apply conversion factors based on weight code
        for code, factor in weight_conversions.items():
            df.loc[df['wt_cod'].isin([code]), 'wt_corrector'] = factor
        
        # Default to kg (1) when wt_cod is NA
        df.loc[df['wt_cod'].isna(), 'wt_corrector'] = 1
        
        # Convert weight to numeric, take absolute value, and apply correction factor
        df['wt_in_kgs'] = np.round(
            np.abs(pd.to_numeric(df['wt'], errors='coerce')) * 
            df['wt_corrector']
        )
        
        # Handle implausible weights (> 635 kg, world record is ~635 kg)
        df.loc[df['wt_in_kgs'] > 635, 'wt_in_kgs'] = np.nan
        
        # Create weight distribution visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        weight_dist = df.groupby('wt_in_kgs').size().reset_index(name='N')
        weight_dist = weight_dist.sort_values('N', ascending=False)
        
        ax.bar(weight_dist['wt_in_kgs'], weight_dist['N'])
        ax.set_xlabel('Weight (kg)')
        ax.set_ylabel('Frequency')
        ax.set_title('Weight Distribution')
        
        # Log weight distribution statistics
        logging.info("Weight distribution summary:")
        logging.info(f"Mean weight: {df['wt_in_kgs'].mean():.2f} kg")
        logging.info(f"Median weight: {df['wt_in_kgs'].median():.2f} kg")
        logging.info(f"Min weight: {df['wt_in_kgs'].min():.2f} kg")
        logging.info(f"Max weight: {df['wt_in_kgs'].max():.2f} kg")
        
        # Drop intermediate columns
        df = df.drop(columns=['wt_corrector', 'wt', 'wt_cod'])
        
        return df, fig

    def standardize_country(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize country names using reference data.
        
        Args:
            df: DataFrame with occr_country and reporter_country columns
        
        Returns:
            DataFrame with standardized country names
        """
        df = df.copy()
        
        # Read country mapping file
        countries = pd.read_csv(
            "external_data/manual_fix/countries.csv",
            sep=";",
            skipinitialspace=True
        )
        
        # Handle Namibia edge case (avoid losing it due to NA)
        countries.loc[countries['country'].isna(), 'country'] = "NA"
        
        # Check for untranslated countries
        all_countries = set(df['occr_country'].unique()) | set(df['reporter_country'].unique())
        untranslated = all_countries - set(countries['country'].unique())
        if untranslated:
            logging.warning(f"Found untranslated countries: {untranslated}")
        
        # Standardize occurrence country
        country_map = dict(zip(countries['country'], countries['Country_Name']))
        df['occr_country'] = df['occr_country'].map(country_map)
        
        # Standardize reporter country
        df['reporter_country'] = df['reporter_country'].map(country_map)
        
        # Drop any temporary columns and remove unused categories
        if 'country' in df.columns:
            df = df.drop(columns=['country'])
        
        # Convert country columns to category type to save memory
        df['occr_country'] = df['occr_country'].astype('category')
        df['reporter_country'] = df['reporter_country'].astype('category')
        
        # Log standardization results
        logging.info("Country standardization summary:")
        logging.info(f"Unique occurrence countries: {len(df['occr_country'].unique())}")
        logging.info(f"Unique reporter countries: {len(df['reporter_country'].unique())}")
        
        return df
    
    def standardize_occupation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize occupation codes, keeping only valid codes and setting others to NA.
        
        Valid codes:
        - MD: Medical Doctor
        - CN: Consumer
        - OT: Other health professional
        - PH: Pharmacist
        - HP: Health Professional
        - LW: Lawyer
        - RN: Registered Nurse
        
        Args:
            df: DataFrame with occp_cod column
        
        Returns:
            DataFrame with standardized occupation codes
        """
        df = df.copy()
        
        # List of valid occupation codes
        valid_codes = ['MD', 'CN', 'OT', 'PH', 'HP', 'LW', 'RN']
        
        # Log initial distribution
        initial_dist = df['occp_cod'].value_counts().sort_values(ascending=False)
        logging.info("Initial occupation code distribution:")
        for code, count in initial_dist.items():
            logging.info(f"{code}: {count}")
        
        # Set invalid codes to NA
        df.loc[~df['occp_cod'].isin(valid_codes), 'occp_cod'] = pd.NA
        
        # Convert to category and remove unused categories
        df['occp_cod'] = df['occp_cod'].astype('category')
        
        # Log final distribution
        final_dist = df['occp_cod'].value_counts().sort_values(ascending=False)
        logging.info("\nFinal occupation code distribution:")
        for code, count in final_dist.items():
            logging.info(f"{code}: {count}")
        
        return df
    
    def standardize_route(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize administration routes."""
        df = df.copy()
        df['route'] = df['route'].str.lower().str.strip()
        df['route'] = df['route'].map(self.route_map)
        return df
    
    def standardize_dose_form(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize dose forms."""
        df = df.copy()
        df['dose_form'] = df['dose_form'].str.lower().str.strip()
        df['dose_form'] = df['dose_form'].map(self.dose_form_map)
        return df
    
    def standardize_dose_freq(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize dose frequencies."""
        df = df.copy()
        df['dose_freq'] = df['dose_freq'].map(self.dose_freq_map)
        return df
    
    def standardize_dates(self, df: pd.DataFrame, date_cols: List[str], min_year: int = 1985) -> pd.DataFrame:
        """Standardize date fields."""
        df = df.copy()
        max_date = datetime.now().strftime("%Y%m%d")
        
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].astype(str)
                
                # Validate dates based on length and range
                mask = df[col].str.len().isin([4, 6, 8])
                df.loc[~mask, col] = np.nan
                
                # Validate year range
                year_mask = (df[col].str[:4].astype(float) >= min_year) & \
                           (df[col].str[:4].astype(float) <= int(max_date[:4]))
                df.loc[~year_mask, col] = np.nan
                
                # Validate month/day if present
                month_mask = df[col].str.len() >= 6
                df.loc[month_mask & (df[col].str[4:6].astype(float) > 12), col] = np.nan
                
                day_mask = df[col].str.len() == 8
                df.loc[day_mask & (df[col].str[6:8].astype(float) > 31), col] = np.nan
        
        return df
    
    def calculate_time_to_onset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time to onset from start date to event date."""
        df = df.copy()
        
        def to_date(dt):
            if pd.isna(dt) or len(str(dt)) != 8:
                return None
            return datetime.strptime(str(dt), '%Y%m%d')
        
        df['start_date'] = df['start_dt'].apply(to_date)
        df['event_date'] = df['event_dt'].apply(to_date)
        
        df['time_to_onset'] = (df['event_date'] - df['start_date']).dt.days + 1
        
        # Clean up invalid values
        df.loc[(df['time_to_onset'] <= 0) & (df['event_dt'] <= 20121231), 'time_to_onset'] = None
        
        df = df.drop(['start_date', 'event_date'], axis=1)
        return df

    def standardize_drugs(self, df: pd.DataFrame, drugname_col: str = 'drugname', 
                         update_dictionary: bool = False) -> pd.DataFrame:
        """
        Standardize drug names using DiAna dictionary and rules.
        
        Args:
            df: DataFrame containing drug data
            drugname_col: Name of the column containing drug names
            update_dictionary: Whether to update the DiAna dictionary with new frequencies
        
        Returns:
            DataFrame with standardized drug names and substances
        """
        # Make a copy to avoid modifying the input
        df = df.copy()
        
        # Clean drug names
        df[drugname_col] = df[drugname_col].apply(self._clean_drugname)
        
        if update_dictionary:
            # Calculate frequencies
            drug_freq = df[drugname_col].value_counts().reset_index()
            drug_freq.columns = ['drugname', 'N']
            drug_freq['freq'] = 100 * drug_freq['N'] / drug_freq['N'].sum()
            
            # Merge with existing dictionary
            merged_dict = pd.merge(
                drug_freq, 
                self.diana_dict[['drugname', 'Substance']], 
                on='drugname', 
                how='left'
            )
            
            # Save updated dictionary
            dict_path = self.external_dir / 'DiAna_dictionary' / 'drugnames_standardized.csv'
            merged_dict.to_csv(dict_path, sep=';', index=False)
            
            # Reload dictionary
            self._load_diana_dictionary()
        
        # Merge with dictionary to get standardized substances
        df = pd.merge(df, self.diana_dict, on=drugname_col, how='left')
        
        # Handle multi-substance entries
        multi_mask = df['Substance'].str.contains(';', na=False)
        
        # Split multi-substance entries
        if multi_mask.any():
            # Process multi-substance entries
            multi_drugs = df[multi_mask].copy()
            single_drugs = df[~multi_mask].copy()
            
            # Split substances and create new rows
            split_drugs = []
            for _, row in multi_drugs.iterrows():
                substances = row['Substance'].split(';')
                for substance in substances:
                    new_row = row.copy()
                    new_row['Substance'] = substance
                    split_drugs.append(new_row)
            
            multi_drugs = pd.DataFrame(split_drugs)
            df = pd.concat([single_drugs, multi_drugs], ignore_index=True)
        
        # Handle trial markings
        df['trial'] = df['Substance'].str.contains(', trial', na=False)
        df['Substance'] = df['Substance'].str.replace(', trial', '')
        
        # Convert columns to categorical
        for col in [drugname_col, 'prod_ai', 'Substance']:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        return df

    def analyze_age_groups(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]:
        """
        Analyze and visualize age groups distribution.
        
        Args:
            df: DataFrame with age_in_years and age_grp columns
        
        Returns:
            Tuple of (processed DataFrame, matplotlib figure with visualization)
        """
        # Create temp DataFrame for analysis
        temp = df[df['age_grp'].notna() & df['age_in_years'].notna()].groupby(
            ['age_in_years', 'age_grp']
        ).size().reset_index(name='N')
        
        # Combine Neonate and Infant groups
        temp.loc[temp['age_grp'].isin(['N', 'I']), 'age_grp'] = 'N&I'
        
        # Set age group order
        temp['age_grp'] = pd.Categorical(
            temp['age_grp'],
            categories=['N&I', 'C', 'T', 'A', 'E'],
            ordered=True
        )
        
        # Define age thresholds
        age_thresholds = pd.DataFrame({
            'age_group': ['C', 'T', 'A', 'E'],
            'age_threshold': [2, 12, 18, 65]
        })
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot points and density
        for group in temp['age_grp'].unique():
            group_data = temp[temp['age_grp'] == group]
            ax.scatter(group_data['age_in_years'], group_data['N'], 
                      label=group, alpha=0.6)
            
            # Add density curve
            sns.kdeplot(data=group_data, x='age_in_years', weights='N',
                       fill=True, alpha=0.3, ax=ax)
        
        # Add threshold markers
        ax.scatter(age_thresholds['age_threshold'], 
                  np.zeros_like(age_thresholds['age_threshold']),
                  color='black', zorder=5)
        
        # Add threshold labels
        for _, row in age_thresholds.iterrows():
            ax.text(row['age_threshold'], -500, str(row['age_threshold']),
                    rotation=45, ha='right', va='top')
        
        # Customize plot
        ax.set_xlabel('Age (yr)')
        ax.set_ylabel('Freq')
        ax.legend(title='Age Group',
                 labels=['Neonate&Infant', 'Child', 'Teenager', 'Adult', 'Elderly'])
        
        return temp, fig

    def standardize_age_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize age groups based on age thresholds.
        
        Args:
            df: DataFrame with age_in_years and age_in_days columns
        
        Returns:
            DataFrame with standardized age groups
        """
        df = df.copy()
        
        # Initialize age_grp_st column with NA
        df['age_grp_st'] = pd.NA
        
        # Apply age group rules
        mask = df['age_in_years'].notna()
        df.loc[mask, 'age_grp_st'] = 'E'  # Default to Elderly
        df.loc[mask & (df['age_in_years'] < 65), 'age_grp_st'] = 'A'
        df.loc[mask & (df['age_in_years'] < 18), 'age_grp_st'] = 'T'
        df.loc[mask & (df['age_in_years'] < 12), 'age_grp_st'] = 'C'
        df.loc[mask & (df['age_in_years'] < 2), 'age_grp_st'] = 'I'
        df.loc[df['age_in_days'] < 28, 'age_grp_st'] = 'N'
        
        # Calculate and log age group distribution
        age_dist = df['age_grp_st'].value_counts().reset_index()
        age_dist.columns = ['age_grp_st', 'N']
        age_dist['perc'] = np.round(100 * age_dist['N'] / age_dist['N'].sum(), 2)
        
        logging.info("Age group distribution:")
        for _, row in age_dist.iterrows():
            logging.info(f"{row['age_grp_st']}: {row['N']} ({row['perc']}%)")
        
        # Drop original age_grp and rename age_grp_st
        if 'age_grp' in df.columns:
            df = df.drop(columns=['age_grp'])
        df = df.rename(columns={'age_grp_st': 'age_grp'})
        
        return df

    def check_date(self, date_series: pd.Series, max_date: int = 20230331) -> pd.Series:
        """
        Validate dates based on length and range criteria.
        
        Args:
            date_series: Series of date values
            max_date: Maximum allowed date (YYYYMMDD format)
        
        Returns:
            Series with invalid dates set to NA
        """
        result = date_series.copy()
        
        # Convert to string and get lengths
        date_lengths = result.astype(str).str.len()
        
        # Get year components for comparison
        max_year = int(str(max_date)[:4])
        
        # Create masks for different date formats
        invalid_4digit = (date_lengths == 4) & ((result < 1985) | (result > max_year))
        invalid_6digit = (date_lengths == 6) & ((result < 198500) | (result > int(str(max_date)[:6])))
        invalid_8digit = (date_lengths == 8) & ((result < 19850000) | (result > max_date))
        invalid_length = ~date_lengths.isin([4, 6, 8])
        
        # Combine all invalid conditions
        invalid_dates = invalid_4digit | invalid_6digit | invalid_8digit | invalid_length
        
        # Set invalid dates to NA
        result[invalid_dates] = pd.NA
        
        return result

    def standardize_dates(self, df: pd.DataFrame, max_date: int = 20230331) -> pd.DataFrame:
        """
        Standardize date columns in the dataframe.
        
        Args:
            df: DataFrame with date columns
            max_date: Maximum allowed date (YYYYMMDD format)
        
        Returns:
            DataFrame with standardized dates
        """
        df = df.copy()
        
        # Date columns to process
        date_columns = ['fda_dt', 'rept_dt', 'mfr_dt', 'init_fda_dt', 'event_dt']
        
        # Process each date column
        for col in date_columns:
            if col in df.columns:
                df[col] = self.check_date(df[col], max_date)
        
        return df

    def standardize_therapy_dates(self, df: pd.DataFrame, max_date: int = 20230331) -> pd.DataFrame:
        """
        Standardize therapy dates and durations.
        
        Args:
            df: DataFrame with therapy dates and duration
            max_date: Maximum allowed date
        
        Returns:
            DataFrame with standardized dates and durations
        """
        df = df.copy()
        
        # Standardize start and end dates
        for col in ['start_dt', 'end_dt']:
            if col in df.columns:
                df[col] = self.check_date(df[col], max_date)
        
        # Duration conversion factors (to days)
        dur_factors = {
            'YR': 365,
            'MON': 30.41667,
            'WK': 7,
            'DAY': 1,
            'HR': 0.04166667,
            'MIN': 0.0006944444,
            'SEC': 1.157407e-05
        }
        
        # Convert duration to numeric
        df['dur'] = pd.to_numeric(df['dur'], errors='coerce')
        
        # Create duration corrector
        df['dur_corrector'] = df['dur_cod'].map(dur_factors)
        
        # Calculate duration in days
        df['dur_in_days'] = abs(df['dur']) * df['dur_corrector']
        
        # Handle implausible durations (> 50 years)
        df.loc[df['dur_in_days'] > 50 * 365, 'dur_in_days'] = pd.NA
        
        # Calculate standardized duration from dates
        df['dur_std'] = pd.NA
        mask_8digit = (df['start_dt'].astype(str).str.len() == 8) & (df['end_dt'].astype(str).str.len() == 8)
        
        if mask_8digit.any():
            start_dates = pd.to_datetime(df.loc[mask_8digit, 'start_dt'].astype(str), format='%Y%m%d')
            end_dates = pd.to_datetime(df.loc[mask_8digit, 'end_dt'].astype(str), format='%Y%m%d')
            df.loc[mask_8digit, 'dur_std'] = (end_dates - start_dates).dt.days + 1
        
        # Handle negative durations
        df.loc[df['dur_std'] < 0, 'dur_std'] = pd.NA
        
        # Use calculated duration if date-based duration is NA
        df.loc[df['dur_std'].isna(), 'dur_std'] = df.loc[df['dur_std'].isna(), 'dur_in_days']
        
        # Backfill missing dates using duration
        def fill_dates(row):
            if pd.isna(row['start_dt']) and not pd.isna(row['end_dt']) and not pd.isna(row['dur_std']):
                end_date = pd.to_datetime(str(row['end_dt']), format='%Y%m%d')
                start_date = end_date - pd.Timedelta(days=row['dur_std'] - 1)
                return int(start_date.strftime('%Y%m%d'))
            return row['start_dt']
        
        def fill_end_dates(row):
            if pd.isna(row['end_dt']) and not pd.isna(row['start_dt']) and not pd.isna(row['dur_std']):
                start_date = pd.to_datetime(str(row['start_dt']), format='%Y%m%d')
                end_date = start_date + pd.Timedelta(days=row['dur_std'] - 1)
                return int(end_date.strftime('%Y%m%d'))
            return row['end_dt']
        
        df['start_dt'] = df.apply(fill_dates, axis=1)
        df['end_dt'] = df.apply(fill_end_dates, axis=1)
        
        # Final duration assignment and cleanup
        df['dur_in_days'] = df['dur_std']
        df = df.drop(columns=['dur_std', 'dur_corrector', 'dur', 'dur_cod'])
        
        return df

    def standardize_drug_info(self, df: pd.DataFrame, max_date: int = 20500101) -> pd.DataFrame:
        """
        Standardize drug information including routes, dose forms, frequencies, and dates.
        
        Args:
            df: DataFrame with drug information
            max_date: Maximum allowed date for expiration dates
        
        Returns:
            DataFrame with standardized drug information
        """
        df = df.copy()
        
        # 1. Route standardization
        if 'route' in df.columns:
            # Clean route strings
            df['route'] = df['route'].str.lower().str.strip()
            
            # Load route standardization mapping
            route_st = pd.read_csv(
                "external_data/manual_fix/route_st.csv",
                sep=";",
                usecols=['route', 'route_st']
            ).drop_duplicates()
            
            # Merge standardized routes
            df = pd.merge(df, route_st, on='route', how='left')
            
            # Log untranslated routes
            untranslated_routes = df[df['route_st'].isna()]['route'].unique()
            if len(untranslated_routes) > 0:
                logging.warning(f"Untranslated routes: {'; '.join(untranslated_routes)}")
        
        # 2. Challenge/Rechallenge standardization
        valid_responses = ['Y', 'N', 'D']
        for col in ['dechal', 'rechal']:
            if col in df.columns:
                df[col] = df[col].where(df[col].isin(valid_responses), pd.NA)
                df[col] = df[col].astype('category')
        
        # 3. Dose form standardization
        if 'dose_form' in df.columns:
            # Clean dose form strings
            df['dose_form'] = df['dose_form'].str.lower().str.strip()
            
            # Load dose form standardization mapping
            dose_form_st = pd.read_csv(
                "external_data/manual_fix/dose_form_st.csv",
                sep=";",
                usecols=['dose_form', 'dose_form_st']
            )
            
            # Merge standardized dose forms
            df = pd.merge(df, dose_form_st, on='dose_form', how='left')
            df['dose_form_st'] = df['dose_form_st'].astype('category')
            
            # Log untranslated dose forms
            untranslated_forms = df[df['dose_form_st'].isna()]['dose_form'].unique()
            if len(untranslated_forms) > 0:
                logging.warning(f"Untranslated dose forms: {'; '.join(untranslated_forms)}")
        
        # 4. Dose frequency standardization
        if 'dose_freq' in df.columns:
            # Load dose frequency standardization mapping
            dose_freq_st = pd.read_csv(
                "external_data/manual_fix/dose_freq_st.csv",
                sep=";",
                usecols=['dose_freq', 'dose_freq_st']
            ).dropna(subset=['dose_freq_st']).drop_duplicates()
            
            # Merge standardized frequencies
            df = pd.merge(df, dose_freq_st, on='dose_freq', how='left')
            df['dose_freq_st'] = df['dose_freq_st'].astype('category')
            
            # Log untranslated frequencies
            untranslated_freq = df[df['dose_freq_st'].isna()]['dose_freq'].unique()
            if len(untranslated_freq) > 0:
                logging.warning(f"Untranslated frequencies: {'; '.join(untranslated_freq)}")
        
        # 5. Route-form standardization
        if 'dose_form_st' in df.columns:
            # Load route-form mapping
            route_form_st = pd.read_csv(
                "external_data/manual_fix/route_form_st.csv",
                sep=";",
                usecols=['dose_form_st', 'route_plus']
            ).drop_duplicates()
            
            # Merge route suggestions
            df = pd.merge(df, route_form_st, on='dose_form_st', how='left')
            
            # Update routes where missing or unknown
            mask = (df['route_st'].isna()) | (df['route_st'] == 'unknown')
            df.loc[mask, 'route_st'] = df.loc[mask, 'route_plus']
            df['route_st'] = df['route_st'].astype('category')
        
        # 6. Select and reorder final columns
        final_columns = [
            'primaryid', 'drug_seq', 'val_vbm', 'route_st', 'dose_vbm',
            'cum_dose_unit', 'cum_dose_chr', 'dose_amt', 'dose_unit',
            'dose_form_st', 'dose_freq_st', 'dechal', 'rechal',
            'lot_num', 'nda_num', 'exp_dt'
        ]
        df = df[final_columns].rename(columns={
            'route_st': 'route',
            'dose_form_st': 'dose_form',
            'dose_freq_st': 'dose_freq'
        })
        
        # 7. Standardize expiration date
        if 'exp_dt' in df.columns:
            df['exp_dt'] = self.check_date(df['exp_dt'], max_date)
        
        return df

    def remove_incomplete_cases(self, demo_df: pd.DataFrame, drug_df: pd.DataFrame, reac_df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove cases that don't have valid drugs or reactions.
        
        Args:
            demo_df: Demographics DataFrame
            drug_df: Drug DataFrame
            reac_df: Reactions DataFrame
        
        Returns:
            Demographics DataFrame with incomplete cases removed
        """
        demo_df = demo_df.copy()
        drug_df = drug_df.copy()
        reac_df = reac_df.copy()
        
        initial_cases = len(demo_df)
        
        # Remove invalid drugs and reactions
        drug_df = drug_df[~drug_df['Substance'].isin(['no medication', 'unspecified'])]
        reac_df = reac_df[~reac_df['pt'].isin(['no adverse event'])]
        
        # Find cases without valid drugs
        valid_drug_cases = set(drug_df['primaryid'].unique())
        all_cases = set(demo_df['primaryid'].unique())
        cases_without_drugs = all_cases - valid_drug_cases
        
        # Find cases without valid reactions
        valid_reaction_cases = set(reac_df['primaryid'].unique())
        cases_without_reactions = all_cases - valid_reaction_cases
        
        # Combine all incomplete cases
        incomplete_cases = cases_without_drugs.union(cases_without_reactions)
        
        # Remove incomplete cases from demographics
        demo_df = demo_df[~demo_df['primaryid'].isin(incomplete_cases)]
        
        # Log results
        removed_cases = initial_cases - len(demo_df)
        logging.info(f"Initial cases: {initial_cases}")
        logging.info(f"Cases without valid drugs: {len(cases_without_drugs)}")
        logging.info(f"Cases without valid reactions: {len(cases_without_reactions)}")
        logging.info(f"Total incomplete cases removed: {len(incomplete_cases)}")
        logging.info(f"Remaining cases: {len(demo_df)}")
        
        return demo_df

    def __init__(self, external_dir: Path):
        """Initialize standardizer with external data directory."""
        self.external_dir = external_dir
        self._load_reference_data()
        self._load_meddra_data()
        self._load_diana_dictionary()
    
    def _load_reference_data(self):
        """Load reference data for standardization."""
        # Load country mappings
        country_file = self.external_dir / 'manual_fix' / 'countries.csv'
        if country_file.exists():
            self.country_map = pd.read_csv(country_file, sep=';', dtype=str).set_index('country')['Country_Name'].to_dict()
        
        # Load occupation codes
        self.valid_occupations = {'MD', 'CN', 'OT', 'PH', 'HP', 'LW', 'RN'}
        
        # Load route standardization
        route_file = self.external_dir / 'manual_fix' / 'route_st.csv'
        if route_file.exists():
            self.route_map = pd.read_csv(route_file, sep=';', dtype=str).set_index('route')['route_st'].to_dict()
        
        # Load dose form standardization
        dose_form_file = self.external_dir / 'manual_fix' / 'dose_form_st.csv'
        if dose_form_file.exists():
            self.dose_form_map = pd.read_csv(dose_form_file, sep=';', dtype=str).set_index('dose_form')['dose_form_st'].to_dict()
        
        # Load dose frequency standardization
        dose_freq_file = self.external_dir / 'manual_fix' / 'dose_freq_st.csv'
        if dose_freq_file.exists():
            self.dose_freq_map = pd.read_csv(dose_freq_file, sep=';', dtype=str).set_index('dose_freq')['dose_freq_st'].to_dict()

    def _load_meddra_data(self):
        """Load MedDRA terminology data from external files."""
        meddra_dir = self.external_dir / 'meddra' / 'MedAscii'
        
        # Load SOC (System Organ Class) data
        soc_file = meddra_dir / 'soc.asc'
        if soc_file.exists():
            self.soc_data = pd.read_csv(soc_file, sep='$', dtype=str)
            self.soc_data.columns = ['soc_code', 'soc_name', 'soc_abbrev', 'soc_whoart_code', 'soc_costart_sym', 'soc_harts_code', 'soc_costart_code', 'soc_icd9_code', 'soc_icd9cm_code', 'soc_icd10_code', 'soc_jart_code']
        
        # Load PT (Preferred Term) data
        pt_file = meddra_dir / 'pt.asc'
        if pt_file.exists():
            self.pt_data = pd.read_csv(pt_file, sep='$', dtype=str)
            self.pt_data.columns = ['pt_code', 'pt_name', 'null_field', 'pt_soc_code', 'pt_whoart_code', 'pt_harts_code', 'pt_costart_sym', 'pt_icd9_code', 'pt_icd9cm_code', 'pt_icd10_code', 'pt_jart_code']
        
        # Load LLT (Lowest Level Term) data
        llt_file = meddra_dir / 'llt.asc'
        if llt_file.exists():
            self.llt_data = pd.read_csv(llt_file, sep='$', dtype=str)
            self.llt_data.columns = ['llt_code', 'llt_name', 'pt_code', 'llt_whoart_code', 'llt_harts_code', 'llt_costart_sym', 'llt_icd9_code', 'llt_icd9cm_code', 'llt_icd10_code', 'llt_currency', 'llt_jart_code']
        
        # Create PT to LLT mapping
        self.pt_to_llt_map = {}
        if hasattr(self, 'llt_data') and hasattr(self, 'pt_data'):
            pt_llt_merged = pd.merge(self.llt_data[['llt_name', 'pt_code']], 
                                    self.pt_data[['pt_code', 'pt_name']], 
                                    on='pt_code')
            for _, row in pt_llt_merged.iterrows():
                self.pt_to_llt_map[row['llt_name'].lower()] = row['pt_name']

        # Initialize standardization tracking
        self.standardization_stats = {
            'total_terms': 0,
            'direct_pt_matches': 0,
            'llt_translations': 0,
            'manual_fixes': 0,
            'unstandardized': 0
        }
        
        # Load manual fixes for terms
        self.manual_pt_fixes = {}
        manual_fix_file = self.external_dir / 'manual_fix' / 'pt_manual_fixes.csv'
        if manual_fix_file.exists():
            manual_fixes = pd.read_csv(manual_fix_file)
            self.manual_pt_fixes = dict(zip(manual_fixes['original'].str.lower(), 
                                          manual_fixes['standardized']))

    def _load_diana_dictionary(self):
        """Load DiAna drug name dictionary."""
        diana_dict_file = self.external_dir / 'DiAna_dictionary' / 'drugnames_standardized.csv'
        if diana_dict_file.exists():
            self.diana_dict = pd.read_csv(diana_dict_file, sep=';')
            # Filter out invalid entries
            self.diana_dict = self.diana_dict[
                (self.diana_dict['Substance'] != 'na') & 
                (~self.diana_dict['Substance'].isna())
            ][['drugname', 'Substance']]

    def _clean_drugname(self, drugname: str) -> str:
        """
        Clean and standardize drug name using DiAna rules.
        
        Args:
            drugname: Raw drug name string
        
        Returns:
            Cleaned drug name string
        """
        if pd.isna(drugname):
            return drugname
            
        # Convert to lowercase and trim whitespace
        name = drugname.lower().strip()
        
        # Remove trailing periods
        name = re.sub(r'\.$', '', name)
        
        # Normalize whitespace
        name = re.sub(r'\s+', ' ', name)
        
        # Remove text after last closing parenthesis
        name = re.sub(r'[^)[:^punct:]]+$', '', name, flags=re.PERL).strip()
        
        # Remove text before first opening parenthesis
        name = re.sub(r'^[^([:^punct:]]+', '', name, flags=re.PERL).strip()
        
        # Repeat the above two steps to handle nested cases
        name = re.sub(r'[^)[:^punct:]]+$', '', name, flags=re.PERL).strip()
        name = re.sub(r'^[^([:^punct:]]+', '', name, flags=re.PERL).strip()
        
        # Fix spacing around parentheses
        name = name.replace('( ', '(').replace(' )', ')')
        
        return name.strip()
