"""
FAERS data standardization service.

Key features:
- Standardizes demographic data
- Standardizes drug names and dosages
- Standardizes reaction terms
- Handles missing and inconsistent data
- NumPy operations optimized for Apple Silicon
"""

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class DataStandardizer:
    """Standardizes FAERS data fields."""
    
    def __init__(self, external_dir: Path):
        """Initialize standardizer with reference data directory.
        
        Args:
            external_dir: Path to external reference data
        """
        self.external_dir = Path(external_dir)
        self._init_logging()
        self._load_reference_data()
        self._load_meddra_data()
        self._load_diana_dictionary()
        
        # Configure NumPy optimizations for Apple Silicon
        try:
            import numpy.distutils.system_info as sysinfo
            blas_info = sysinfo.get_info('blas_opt')
            if blas_info:
                logging.info("Using optimized BLAS for Apple Silicon")
                # Set NumPy threading to match physical cores for M1/M2
                import multiprocessing as mp
                physical_cores = mp.cpu_count() // 2  # Account for efficiency cores
                os.environ["OMP_NUM_THREADS"] = str(physical_cores)
                os.environ["OPENBLAS_NUM_THREADS"] = str(physical_cores)
                os.environ["MKL_NUM_THREADS"] = str(physical_cores)
                os.environ["VECLIB_MAXIMUM_THREADS"] = str(physical_cores)
                os.environ["NUMEXPR_NUM_THREADS"] = str(physical_cores)
            else:
                logging.info("Standard BLAS configuration in use")
        except Exception as e:
            logging.info(f"Using default NumPy configuration: {str(e)}")

    def _init_logging(self):
        """Initialize logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def get_drug_dictionary(self) -> Dict[str, str]:
        """Get or load drug standardization dictionary."""
        if not hasattr(self, '_drug_dictionary'):
            dict_path = self.external_dir / 'drug_dictionary.csv'
            if dict_path.exists():
                df = pd.read_csv(dict_path, low_memory=False)
                self._drug_dictionary = dict(zip(df['original'], df['standard']))
            else:
                logging.warning(f"Drug dictionary not found at {dict_path}")
                self._drug_dictionary = {}
        return self._drug_dictionary

    def standardize_dates(self, df: pd.DataFrame, max_date: int = 20230331) -> pd.DataFrame:
        """Standardize date columns in the dataframe.
        
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
                df[col] = self._check_date(df[col], max_date)

        return df

    def standardize_therapy_dates(self, df: pd.DataFrame, max_date: int = 20230331) -> pd.DataFrame:
        """Standardize therapy dates and durations.
        
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
                df[col] = self._check_date(df[col], max_date)

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
        """Standardize drug information including routes, dose forms, frequencies, and dates.
        
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
                usecols=['route', 'route_st'],
                low_memory=False
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
                usecols=['dose_form', 'dose_form_st'],
                low_memory=False
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
                usecols=['dose_freq', 'dose_freq_st'],
                low_memory=False
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
                usecols=['dose_form_st', 'route_plus'],
                low_memory=False
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
            df['exp_dt'] = self._check_date(df['exp_dt'], max_date)

        return df

    def remove_incomplete_cases(self, demo_df: pd.DataFrame, drug_df: pd.DataFrame,
                                reac_df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove cases that don't have valid drugs or reactions.
        
        Args:
            demo_df: Demographics DataFrame
            drug_df: Drug DataFrame
            reac_df: Reactions DataFrame
        
        Returns:
            DataFrame with incomplete cases removed
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

        # Calculate and log results
        removed_cases = initial_cases - len(demo_df)
        logging.info(f"Initial cases: {initial_cases}")
        logging.info(f"Cases without valid drugs: {len(cases_without_drugs)}")
        logging.info(f"Cases without valid reactions: {len(cases_without_reactions)}")
        logging.info(f"Total incomplete cases removed: {removed_cases}")
        logging.info(f"Remaining cases: {len(demo_df)}")

        return demo_df

    def _load_reference_data(self):
        """Load reference data for standardization."""
        # Load country mappings
        country_file = self.external_dir / 'manual_fix' / 'countries.csv'
        if country_file.exists():
            self.country_map = pd.read_csv(country_file, sep=';', dtype=str, low_memory=False).set_index('country')['Country_Name'].to_dict()

        # Load occupation codes (matching R script)
        self.valid_occupations = {'MD', 'CN', 'OT', 'PH', 'HP', 'LW', 'RN'}

        # Load route standardization
        route_file = self.external_dir / 'manual_fix' / 'route_st.csv'
        if route_file.exists():
            self.route_map = pd.read_csv(route_file, sep=';', dtype=str, low_memory=False).set_index('route')['route_st'].to_dict()

        # Load dose form standardization
        dose_form_file = self.external_dir / 'manual_fix' / 'dose_form_st.csv'
        if dose_form_file.exists():
            self.dose_form_map = pd.read_csv(dose_form_file, sep=';', dtype=str, low_memory=False).set_index('dose_form')['dose_form_st'].to_dict()

        # Load dose frequency standardization
        dose_freq_file = self.external_dir / 'manual_fix' / 'dose_freq_st.csv'
        if dose_freq_file.exists():
            self.dose_freq_map = pd.read_csv(dose_freq_file, sep=';', dtype=str, low_memory=False).set_index('dose_freq')['dose_freq_st'].to_dict()

    def _load_meddra_data(self):
        """Load MedDRA terminology data from external files."""
        meddra_dir = self.external_dir / 'meddra' / 'MedAscii'

        # Load SOC data
        soc_file = meddra_dir / 'soc.asc'
        if soc_file.exists():
            self.soc_data = pd.read_csv(soc_file, sep='$', dtype=str, low_memory=False, usecols=[0,1,2])
            self.soc_data.columns = ['soc_code', 'soc_name', 'soc_abbrev']

        # Load PT data
        pt_file = meddra_dir / 'pt.asc'
        if pt_file.exists():
            self.pt_data = pd.read_csv(pt_file, sep='$', dtype=str, low_memory=False, usecols=[0,1,3])
            self.pt_data.columns = ['pt_code', 'pt_name', 'pt_soc_code']

        # Load LLT data
        llt_file = meddra_dir / 'llt.asc'
        if llt_file.exists():
            self.llt_data = pd.read_csv(llt_file, sep='$', dtype=str, low_memory=False, usecols=[0,1,2])
            self.llt_data.columns = ['llt_code', 'llt_name', 'pt_code']

        # Create PT to LLT mapping (case-insensitive)
        self.pt_to_llt_map = {}
        if hasattr(self, 'llt_data') and hasattr(self, 'pt_data'):
            pt_llt_merged = pd.merge(
                self.llt_data[['llt_name', 'pt_code']],
                self.pt_data[['pt_code', 'pt_name']],
                on='pt_code'
            )
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
            manual_fixes = pd.read_csv(manual_fix_file, low_memory=False)
            self.manual_pt_fixes = dict(zip(manual_fixes['original'].str.lower(), 
                                          manual_fixes['standardized']))

    def _load_diana_dictionary(self):
        """Load and prepare the DiAna drug dictionary."""
        try:
            dict_path = self.external_dir / 'DiAna_dictionary' / 'drugnames_standardized.csv'
            if not dict_path.exists():
                raise FileNotFoundError(f"DiAna dictionary not found at {dict_path}")
            
            # Read with error_bad_lines=False to skip problematic rows
            self.diana_dict = pd.read_csv(
                dict_path,
                dtype={'drugname': str, 'Substance': str},
                low_memory=False,
                on_bad_lines='skip',  # Skip problematic lines
                delimiter=';'  # Use semicolon delimiter
            )
            
            # Clean dictionary entries (matching R script)
            self.diana_dict['drugname'] = self.diana_dict['drugname'].apply(self._clean_drugname)
            self.diana_dict['Substance'] = self.diana_dict['Substance'].fillna('UNKNOWN')
            
            # Create mapping dictionary
            self.drug_map = dict(zip(
                self.diana_dict['drugname'].str.lower().str.strip(),
                self.diana_dict['Substance']
            ))
            
            logging.info(f"Loaded DiAna dictionary with {len(self.diana_dict)} entries")
            
        except Exception as e:
            logging.error(f"Error loading DiAna dictionary: {str(e)}")
            self.diana_dict = pd.DataFrame(columns=['drugname', 'Substance'])
            self.drug_map = {}

    def standardize_pt(self, df: pd.DataFrame, pt_variable: str) -> pd.DataFrame:
        """Standardize PT terms exactly as in R script.
        
        Args:
            df: DataFrame with PT terms
            pt_variable: Name of the PT column
            
        Returns:
            DataFrame with standardized PT terms
        """
        df = df.copy()
        
        # 1. Load MedDRA PT list
        meddra_df = pd.read_csv(self.external_dir / 'Dictionaries/MedDRA/meddra.csv', sep=';', low_memory=False)
        pt_list = pd.Series(meddra_df['pt'].unique()).str.lower().str.strip().unique()
        
        # 2. Calculate PT frequencies
        df[pt_variable] = df[pt_variable].str.lower().str.strip()
        pt_freq = (df[~df[pt_variable].isna()]
                   .groupby(pt_variable).size()
                   .reset_index(name='N')
                   .sort_values('N', ascending=False))
        
        # 3. Check if PTs are standardized
        pt_freq['standard_pt'] = np.where(pt_freq[pt_variable].isin(pt_list), 
                                         pt_freq[pt_variable], 
                                         np.nan)
        pt_freq['freq'] = np.round(pt_freq['N'] / pt_freq['N'].sum() * 100, 2)
        
        # 4. Get unstandardized PTs
        not_pts = pt_freq[pt_freq['standard_pt'].isna()][[pt_variable, 'N', 'freq']]
        
        # 5. Calculate initial non-standardized percentage
        initial_nonstd_pct = np.round(not_pts['N'].sum() * 100 / len(df[~df[pt_variable].isna()]), 3)
        logging.info(f"Initial non-standardized PT percentage: {initial_nonstd_pct}%")
        
        # 6. Try to translate through LLTs
        llt_mappings = meddra_df[['pt', 'llt']].copy()
        llt_mappings.columns = ['standard_pt', pt_variable]
        not_pts = not_pts.merge(llt_mappings, on=pt_variable, how='left')
        not_llts = not_pts[not_pts['standard_pt'].isna()].drop('standard_pt', axis=1)
        
        # 7. Load and apply manual fixes
        manual_fixes = pd.read_csv(self.external_dir / 'Manual_fix/pt_fixed.csv', sep=';', low_memory=False)
        manual_fixes = manual_fixes[[pt_variable, 'standard_pt']]
        
        not_llts = not_llts.merge(manual_fixes, on=pt_variable, how='left')
        still_unstandardized = not_llts[not_llts['standard_pt'].isna()][[pt_variable, 'standard_pt']]
        
        # 8. Combine all standardization sources
        pt_fixed = pd.concat([
            manual_fixes,
            still_unstandardized,
            not_pts[~not_pts['standard_pt'].isna()][[pt_variable, 'standard_pt']]
        ]).drop_duplicates()
        
        # 9. Check for duplicates
        duplicates = pt_fixed[pt_fixed[pt_variable].duplicated()]
        if not duplicates.empty:
            logging.warning(f"Duplicate PT mappings found: {duplicates[pt_variable].tolist()}")
        
        # 10. Apply standardization to original data
        df['pt_temp'] = df[pt_variable].str.lower().str.strip()
        df = df.merge(pt_fixed, left_on='pt_temp', right_on=pt_variable, how='left')
        df[pt_variable] = df['standard_pt'].fillna(df['pt_temp'])
        df = df.drop(['pt_temp', 'standard_pt'], axis=1)
        
        # 11. Calculate final standardization percentage
        final_std_pct = np.round(len(df[df[pt_variable].isin(pt_list)]) * 100 / len(df[~df[pt_variable].isna()]), 3)
        logging.info(f"Final standardized PT percentage: {final_std_pct}%")
        
        return df

    def standardize_drugs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize drug names using DiAna dictionary.
        
        Args:
            df: DataFrame containing drug data
            
        Returns:
            DataFrame with standardized drug names and substances
        """
        df = df.copy()
        
        # 1. Ensure drugname column exists
        if 'drugname' not in df.columns and 'drug_name' in df.columns:
            df = df.rename(columns={'drug_name': 'drugname'})
        
        if 'drugname' not in df.columns:
            logging.error("No drugname or drug_name column found in DataFrame")
            return df
        
        # 2. Clean drug names
        df['drugname'] = df['drugname'].fillna('').astype(str).apply(self._clean_drugname)
        
        # 3. Map to standardized substances using the precomputed mapping
        df['Substance'] = df['drugname'].str.lower().str.strip().map(self.drug_map).fillna('UNKNOWN')
        
        # 4. Handle multi-substance drugs
        multi_substance = df[df['Substance'].str.contains(';', na=False)]
        single_substance = df[~df['Substance'].str.contains(';', na=False)]
        
        # Split multi-substance drugs
        if len(multi_substance) > 0:
            split_substances = []
            for _, row in multi_substance.iterrows():
                substances = row['Substance'].split(';')
                for substance in substances:
                    new_row = row.copy()
                    new_row['Substance'] = substance.strip()
                    split_substances.append(new_row)
            multi_substance_df = pd.DataFrame(split_substances)
            
            # Combine back with single substance drugs
            df = pd.concat([single_substance, multi_substance_df], ignore_index=True)
        
        # 5. Mark trial drugs
        df['trial'] = df['Substance'].str.contains(', trial', na=False)
        df['Substance'] = df['Substance'].str.replace(', trial', '', regex=False)
        
        # 6. Convert to categorical for memory efficiency
        df['drugname'] = df['drugname'].astype('category')
        df['Substance'] = df['Substance'].astype('category')
        if 'prod_ai' in df.columns:
            df['prod_ai'] = df['prod_ai'].astype('category')
        
        # 7. Log standardization results
        total_drugs = len(df)
        unique_substances = df['Substance'].nunique()
        trial_drugs = df['trial'].sum()
        logging.info(f"Drug standardization results:")
        logging.info(f"  Total drugs: {total_drugs}")
        logging.info(f"  Unique substances: {unique_substances}")
        logging.info(f"  Trial drugs: {trial_drugs}")
        
        return df

    def standardize_sex(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize sex values.
        
        Args:
            df: DataFrame with sex column
            
        Returns:
            DataFrame with standardized sex values
        """
        df = df.copy()
        df.loc[~df['sex'].isin(['F', 'M']), 'sex'] = pd.NA
        return df

    def standardize_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize age values to days and years.
        
        Args:
            df: DataFrame with age columns
            
        Returns:
            DataFrame with standardized age values
        """
        df = df.copy()
        
        # Convert age to numeric, handling various formats
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(
                df['age'].astype(str).str.replace(',', ''),
                errors='coerce'
            )
        
        # Define age unit conversion factors
        age_factors = {
            'DEC': 3650,
            'YR': 365,
            'MON': 30.41667,
            'WK': 7,
            'DY': 1,
            'HR': 0.00011415525114155251,
            'MIN': 1.9025875190259e-06,
            'SEC': 3.1709791983764586e-08
        }
        
        # Convert age to days
        df['age_corrector'] = df['age_cod'].map(age_factors)
        df['age_in_days'] = df['age'].abs() * df['age_corrector']
        
        # Handle plausible age range
        max_age_days = 122 * 365  # Maximum recorded human age
        df.loc[df['age_in_days'] > max_age_days, 'age_in_days'] = pd.NA
        
        # Convert to years
        df['age_in_years'] = (df['age_in_days'] / 365).round()
        
        # Clean up temporary columns
        df = df.drop(columns=['age_corrector', 'age', 'age_cod'])
        
        return df

    def standardize_age_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize age groups based on thresholds.
        
        Args:
            df: DataFrame with age values
            
        Returns:
            DataFrame with standardized age groups
        """
        df = df.copy()
        
        # Initialize age group column
        df['age_grp_st'] = pd.NA
        
        # Apply age group rules
        mask = df['age_in_years'].notna()
        df.loc[mask, 'age_grp_st'] = 'E'  # Default to Elderly
        df.loc[mask & (df['age_in_years'] < 65), 'age_grp_st'] = 'A'
        df.loc[mask & (df['age_in_years'] < 18), 'age_grp_st'] = 'T'
        df.loc[mask & (df['age_in_years'] < 12), 'age_grp_st'] = 'C'
        df.loc[mask & (df['age_in_years'] < 2), 'age_grp_st'] = 'I'
        df.loc[df['age_in_days'] < 28, 'age_grp_st'] = 'N'
        
        # Log distribution
        dist = df['age_grp_st'].value_counts()
        total = len(df)
        for group, count in dist.items():
            percent = round(100 * count / total, 2)
            logging.info(f"{group}: {count} ({percent}%)")
        
        # Update column name
        if 'age_grp' in df.columns:
            df = df.drop(columns=['age_grp'])
        df = df.rename(columns={'age_grp_st': 'age_grp'})
        
        return df

    def standardize_weight(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize weight values to kilograms.
        
        Args:
            df: DataFrame with weight columns
            
        Returns:
            DataFrame with standardized weight values
        """
        df = df.copy()
        
        # Weight unit conversion factors
        wt_factors = {
            'LBS': 0.453592,  # Pounds to kg
            'IB': 0.453592,   # Pounds to kg (alternative code)
            'KG': 1,          # Already in kg
            'KGS': 1,         # Already in kg (alternative code)
            'GMS': 0.001,     # Grams to kg
            'MG': 1e-06       # Milligrams to kg
        }
        
        # Convert weight to kg
        df['wt_corrector'] = df['wt_cod'].map(wt_factors).fillna(1)
        df['wt_in_kgs'] = pd.to_numeric(df['wt'].abs(), errors='coerce') * df['wt_corrector']
        
        # Round to nearest kg
        df['wt_in_kgs'] = df['wt_in_kgs'].round()
        
        # Handle implausible weights (> 635 kg)
        df.loc[df['wt_in_kgs'] > 635, 'wt_in_kgs'] = pd.NA
        
        # Log weight distribution
        weight_stats = df['wt_in_kgs'].describe()
        logging.info("Weight distribution (kg):")
        for stat, value in weight_stats.items():
            logging.info(f"  {stat}: {value:.2f}")
        
        # Clean up temporary columns
        df = df.drop(columns=['wt_corrector', 'wt', 'wt_cod'])
        
        return df

    def standardize_country(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize country codes using ISO standards.
        
        Args:
            df: DataFrame with country columns
            
        Returns:
            DataFrame with standardized country names
        """
        df = df.copy()
        
        # Load country mappings
        countries_df = pd.read_csv(self.external_dir / 'Manual_fix/countries.csv', sep=';', low_memory=False)
        
        # Handle special case for Namibia (NA)
        countries_df.loc[countries_df['country'].isna(), 'country'] = 'NA'
        
        # Create mapping dictionary
        country_map = dict(zip(countries_df['country'], countries_df['Country_Name']))
        
        # Standardize country columns
        country_cols = ['occr_country', 'reporter_country']
        for col in country_cols:
            if col in df.columns:
                # Apply mapping
                df[col] = df[col].map(country_map)
                
                # Log unmapped countries
                unmapped = df[~df[col].isna() & ~df[col].isin(country_map.values())][col].unique()
                if len(unmapped) > 0:
                    logging.warning(f"Unmapped countries in {col}: {unmapped}")
                
                # Convert to categorical
                df[col] = df[col].astype('category')
        
        return df

    def standardize_occupation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize occupation codes.
        
        Args:
            df: DataFrame with occupation codes
            
        Returns:
            DataFrame with standardized occupation codes
        """
        df = df.copy()
        
        # Valid occupation codes
        valid_codes = {
            'MD': 'Medical Doctor',
            'CN': 'Consumer',
            'OT': 'Other Health Professional',
            'PH': 'Pharmacist',
            'HP': 'Health Professional',
            'LW': 'Lawyer',
            'RN': 'Registered Nurse'
        }
        
        # Standardize codes
        df.loc[~df['occp_cod'].isin(valid_codes.keys()), 'occp_cod'] = pd.NA
        
        # Convert to categorical
        df['occp_cod'] = df['occp_cod'].astype('category')
        
        # Log occupation distribution
        occ_dist = df['occp_cod'].value_counts(dropna=False)
        total = len(df)
        logging.info("Occupation distribution:")
        for code, count in occ_dist.items():
            pct = round(100 * count / total, 2)
            name = valid_codes.get(code, 'Unknown/NA')
            logging.info(f"  {code} ({name}): {count} ({pct}%)")
        
        return df

    def standardize_manufacturer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process manufacturer records in demographics data.
        
        Args:
            df: DataFrame with manufacturer information
            
        Returns:
            DataFrame with processed manufacturer records
        """
        df = df.copy()
        
        # Sort by FDA date
        df = df.sort_values('fda_dt')
        
        # Group by case ID and get last record for each manufacturer
        last_mfr_records = df.groupby(['caseid', 'mfr_sndr']).tail(1)
        
        # Analyze manufacturer data
        total_records = len(df)
        unique_cases = df['caseid'].nunique()
        unique_mfrs = df['mfr_sndr'].nunique()
        missing_mfr = df['mfr_sndr'].isna().sum()
        
        # Log manufacturer statistics
        logging.info("Manufacturer record analysis:")
        logging.info(f"  Total records: {total_records}")
        logging.info(f"  Unique cases: {unique_cases}")
        logging.info(f"  Unique manufacturers: {unique_mfrs}")
        logging.info(f"  Records with missing manufacturer: {missing_mfr}")
        logging.info(f"  Records after keeping last per manufacturer: {len(last_mfr_records)}")
        
        # Analyze cases with multiple manufacturers
        cases_multiple_mfrs = df.groupby('caseid')['mfr_sndr'].nunique()
        multi_mfr_cases = cases_multiple_mfrs[cases_multiple_mfrs > 1]
        if len(multi_mfr_cases) > 0:
            logging.info(f"  Cases with multiple manufacturers: {len(multi_mfr_cases)}")
            logging.info("  Distribution of manufacturers per case:")
            for n_mfrs, count in multi_mfr_cases.value_counts().items():
                logging.info(f"    {n_mfrs} manufacturers: {count} cases")
        
        return df

    def standardize_route(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize medication administration routes.
        
        Args:
            df: DataFrame with route information
            
        Returns:
            DataFrame with standardized routes
        """
        df = df.copy()
        
        if 'route' not in df.columns:
            return df
        
        # Clean route strings
        df['route'] = df['route'].str.lower().str.strip()
        
        # Load route standardization mapping
        route_st = pd.read_csv(
            self.external_dir / 'Manual_fix/route_st.csv',
            sep=';',
            usecols=['route', 'route_st'],
            low_memory=False
        ).drop_duplicates()
        
        # Create mapping dictionary
        route_map = dict(zip(route_st['route'], route_st['route_st']))
        
        # Apply standardization
        df['route_st'] = df['route'].map(route_map)
        
        # Log unmapped routes
        unmapped = df[~df['route'].isna() & df['route_st'].isna()]['route'].unique()
        if len(unmapped) > 0:
            logging.warning(f"Unmapped routes: {unmapped}")
        
        # Convert to categorical
        df['route_st'] = df['route_st'].astype('category')
        
        # Update column name
        df = df.drop(columns=['route']).rename(columns={'route_st': 'route'})
        
        return df

    def standardize_dose_form(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize medication dose forms.
        
        Args:
            df: DataFrame with dose form information
            
        Returns:
            DataFrame with standardized dose forms
        """
        df = df.copy()
        
        if 'dose_form' not in df.columns:
            return df
        
        # Clean dose form strings
        df['dose_form'] = df['dose_form'].str.lower().str.strip()
        
        # Load dose form standardization mapping
        dose_form_st = pd.read_csv(
            self.external_dir / 'Manual_fix/dose_form_st.csv',
            sep=';',
            usecols=['dose_form', 'dose_form_st'],
            low_memory=False
        ).drop_duplicates()
        
        # Create mapping dictionary
        form_map = dict(zip(dose_form_st['dose_form'], dose_form_st['dose_form_st']))
        
        # Apply standardization
        df['dose_form_st'] = df['dose_form'].map(form_map)
        
        # Log unmapped forms
        unmapped = df[~df['dose_form'].isna() & df['dose_form_st'].isna()]['dose_form'].unique()
        if len(unmapped) > 0:
            logging.warning(f"Unmapped dose forms: {unmapped}")
        
        # Convert to categorical
        df['dose_form_st'] = df['dose_form_st'].astype('category')
        
        # Update column name
        df = df.drop(columns=['dose_form']).rename(columns={'dose_form_st': 'dose_form'})
        
        return df

    def standardize_dose(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize medication doses and units.
        
        Args:
            df: DataFrame with dose information
            
        Returns:
            DataFrame with standardized doses
        """
        df = df.copy()
        
        # Load unit standardization mapping
        unit_st = pd.read_csv(
            self.external_dir / 'Manual_fix/unit_st.csv',
            sep=';',
            usecols=['unit', 'unit_st', 'conversion_factor'],
            low_memory=False
        ).drop_duplicates()
        
        # Create unit mapping dictionary
        unit_map = dict(zip(unit_st['unit'], unit_st['unit_st']))
        factor_map = dict(zip(unit_st['unit'], unit_st['conversion_factor']))
        
        # Standardize dose units
        if 'dose_unit' in df.columns:
            df['dose_unit'] = df['dose_unit'].str.lower().str.strip()
            df['dose_unit_st'] = df['dose_unit'].map(unit_map)
            df['conversion_factor'] = df['dose_unit'].map(factor_map).fillna(1.0)
            
            # Log unmapped units
            unmapped = df[~df['dose_unit'].isna() & df['dose_unit_st'].isna()]['dose_unit'].unique()
            if len(unmapped) > 0:
                logging.warning(f"Unmapped dose units: {unmapped}")
            
            # Convert dose amounts
            if 'dose_amt' in df.columns:
                df['dose_amt'] = pd.to_numeric(df['dose_amt'], errors='coerce')
                df['dose_amt'] = df['dose_amt'] * df['conversion_factor']
            
            # Clean up columns
            df = df.drop(columns=['dose_unit', 'conversion_factor'])
            df = df.rename(columns={'dose_unit_st': 'dose_unit'})
            df['dose_unit'] = df['dose_unit'].astype('category')
        
        # Standardize cumulative dose units
        if 'cum_dose_unit' in df.columns:
            df['cum_dose_unit'] = df['cum_dose_unit'].str.lower().str.strip()
            df['cum_dose_unit_st'] = df['cum_dose_unit'].map(unit_map)
            df['conversion_factor'] = df['cum_dose_unit'].map(factor_map).fillna(1.0)
            
            # Log unmapped units
            unmapped = df[~df['cum_dose_unit'].isna() & df['cum_dose_unit_st'].isna()]['cum_dose_unit'].unique()
            if len(unmapped) > 0:
                logging.warning(f"Unmapped cumulative dose units: {unmapped}")
            
            # Convert cumulative dose amounts
            if 'cum_dose_chr' in df.columns:
                df['cum_dose_chr'] = pd.to_numeric(df['cum_dose_chr'], errors='coerce')
                df['cum_dose_chr'] = df['cum_dose_chr'] * df['conversion_factor']
            
            # Clean up columns
            df = df.drop(columns=['cum_dose_unit', 'conversion_factor'])
            df = df.rename(columns={'cum_dose_unit_st': 'cum_dose_unit'})
            df['cum_dose_unit'] = df['cum_dose_unit'].astype('category')
        
        return df

    def standardize_outcome(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize adverse event outcomes.
        
        Args:
            df: DataFrame with outcome information
            
        Returns:
            DataFrame with standardized outcomes
        """
        df = df.copy()
        
        if 'outc_cod' not in df.columns:
            return df
        
        # Valid outcome codes and their descriptions
        valid_outcomes = {
            'DE': 'Death',
            'LT': 'Life-Threatening',
            'HO': 'Hospitalization',
            'DS': 'Disability',
            'CA': 'Congenital Anomaly',
            'RI': 'Required Intervention',
            'OT': 'Other'
        }
        
        # Standardize codes
        df.loc[~df['outc_cod'].isin(valid_outcomes.keys()), 'outc_cod'] = pd.NA
        
        # Convert to categorical
        df['outc_cod'] = df['outc_cod'].astype('category')
        
        # Log outcome distribution
        outcome_dist = df['outc_cod'].value_counts(dropna=False)
        total = len(df)
        logging.info("Outcome distribution:")
        for code, count in outcome_dist.items():
            pct = round(100 * count / total, 2)
            desc = valid_outcomes.get(code, 'Unknown/NA')
            logging.info(f"  {code} ({desc}): {count} ({pct}%)")
        
        return df

    def _clean_drugname(self, name: str) -> str:
        """Clean drug name by removing special characters and standardizing format.
        
        Args:
            name: Drug name to clean
            
        Returns:
            Cleaned drug name
        """
        if pd.isna(name):
            return name
        
        # Convert to string and lowercase
        name = str(name).lower()
        
        # Remove special characters but keep hyphens and spaces
        name = re.sub(r'[^\w\s-]', '', name)
        
        # Replace multiple spaces with single space
        name = re.sub(r'\s+', ' ', name)
        
        # Strip leading/trailing spaces
        name = name.strip()
        
        return name

    def _check_date(self, date_series: pd.Series, max_date: int = 20230331) -> pd.Series:
        """Check and standardize date values.
        
        Args:
            date_series: Series of dates to check
            max_date: Maximum allowed date in YYYYMMDD format
            
        Returns:
            Series with standardized dates
        """
        def is_valid_date(date_val) -> bool:
            if pd.isna(date_val):
                return False
                
            try:
                # Convert to string and ensure 8 digits
                date_str = str(int(float(str(date_val).replace(',', ''))))
                date_str = date_str.zfill(8)
                
                # Extract components
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                
                # Basic range checks
                if not (1900 <= year <= 2100):
                    return False
                if not (1 <= month <= 12):
                    return False
                if not (1 <= day <= 31):
                    return False
                
                # Check month-specific day limits
                days_in_month = {
                    2: 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
                    4: 30, 6: 30, 9: 30, 11: 30
                }
                max_days = days_in_month.get(month, 31)
                if day > max_days:
                    return False
                
                # Check against max date
                if int(date_str) > max_date:
                    return False
                
                return True
            except (ValueError, TypeError):
                return False
        
        # Convert series to numeric, handling various formats
        result = pd.to_numeric(
            date_series.astype(str).str.replace(',', ''),
            errors='coerce'
        ).fillna(-1).astype('int64')
        
        # Apply validation
        mask = ~result.apply(is_valid_date)
        result[mask] = pd.NA
        
        # Log invalid dates
        invalid_count = mask.sum()
        if invalid_count > 0:
            total = len(date_series)
            pct = round(100 * invalid_count / total, 2)
            logging.warning(f"Invalid dates found: {invalid_count} ({pct}%)")
        
        return result

    def process_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process demographics data following R script order exactly.
        
        Args:
            df: Raw demographics DataFrame
            
        Returns:
            Processed demographics DataFrame
        """
        df = df.copy()
        
        # 1. Initial data cleaning
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        
        # 2. Standardize dates
        date_cols = ['init_fda_dt', 'fda_dt', 'event_dt', 'rept_dt']
        df = self.standardize_dates(df, date_cols)
        
        # 3. Standardize sex values
        df = self.standardize_sex(df)
        
        # 4. Standardize age values
        df = self.standardize_age(df)
        
        # 5. Standardize weight values
        df = self.standardize_weight(df)
        
        # 6. Standardize country codes
        df = self.standardize_country(df)
        
        # 7. Standardize occupation codes
        df = self.standardize_occupation(df)
        
        # 8. Handle manufacturer records
        df = self.standardize_manufacturer(df)
        
        # 9. Memory optimization
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')
        
        return df

    def process_drugs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process drug data following R script order exactly.
        
        Args:
            df: Raw drug DataFrame
            
        Returns:
            Processed drug DataFrame
        """
        df = df.copy()
        
        # 1. Initial data cleaning (exactly like R script)
        df['drugname'] = df['drugname'].fillna('').astype(str)
        df['drugname'] = df['drugname'].str.lower()
        df['drugname'] = df['drugname'].str.strip()
        df['drugname'] = df['drugname'].str.replace(r'\s+', ' ', regex=True)
        df['drugname'] = df['drugname'].str.replace(r'\.$', '', regex=True)
        df['drugname'] = df['drugname'].str.replace(r'\( ', '(', regex=True)
        df['drugname'] = df['drugname'].str.replace(r' \)', ')', regex=True)
        
        # 2. Map to standardized substances (exact match first)
        df['Substance'] = df['drugname'].map(self.drug_map).fillna('UNKNOWN')
        
        # 3. Handle multi-substance drugs (split into separate rows)
        multi_substance = df[df['Substance'].str.contains(';', na=False)]
        single_substance = df[~df['Substance'].str.contains(';', na=False)]
        
        if len(multi_substance) > 0:
            split_substances = []
            for _, row in multi_substance.iterrows():
                substances = row['Substance'].split(';')
                for substance in substances:
                    new_row = row.copy()
                    new_row['Substance'] = substance.strip()
                    split_substances.append(new_row)
            multi_substance_df = pd.DataFrame(split_substances)
            df = pd.concat([single_substance, multi_substance_df], ignore_index=True)
        
        # 4. Mark trial drugs (after splitting)
        df['trial'] = df['Substance'].str.contains(', trial', na=False)
        df['Substance'] = df['Substance'].str.replace(', trial', '', regex=False)
        
        # 5. Standardize routes (if present)
        if 'route' in df.columns:
            df['route'] = self.standardize_route(df['route'])
        
        # 6. Standardize dose forms (if present)
        if 'dose_form' in df.columns:
            df['dose_form'] = self.standardize_dose_form(df['dose_form'])
        
        # 7. Convert dosages to standard units (if present)
        if all(col in df.columns for col in ['dose_amt', 'dose_unit']):
            df['dose_std'] = self.standardize_dose(df['dose_amt'], df['dose_unit'])
        
        # 8. Memory optimization (after all processing)
        df['drugname'] = df['drugname'].astype('category')
        df['Substance'] = df['Substance'].astype('category')
        if 'prod_ai' in df.columns:
            df['prod_ai'] = df['prod_ai'].astype('category')
        
        # 9. Log statistics
        logging.info(f"Drug processing statistics:")
        logging.info(f"  Total drugs: {len(df)}")
        logging.info(f"  Unique substances: {df['Substance'].nunique()}")
        logging.info(f"  Trial drugs: {df['trial'].sum()}")
        if 'route' in df.columns:
            logging.info(f"  Unique routes: {df['route'].nunique()}")
        
        return df

    def process_reactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process reaction data following R script order exactly.
        
        Args:
            df: Raw reaction DataFrame
            
        Returns:
            Processed reaction DataFrame
        """
        df = df.copy()
        
        # 1. Clean and standardize PT terms
        if 'pt' in df.columns:
            # First try direct PT matches
            df['pt'] = df['pt'].str.lower().str.strip()
            df['reaction_term'] = df['pt'].map(lambda x: x if x in self.pt_data['pt_name'].str.lower().values else None)
            
            # Then try LLT translations for non-standard terms
            mask = df['reaction_term'].isna()
            df.loc[mask, 'reaction_term'] = df.loc[mask, 'pt'].map(self.pt_to_llt_map)
            
            # Finally try manual fixes
            still_missing = df['reaction_term'].isna()
            df.loc[still_missing, 'reaction_term'] = df.loc[still_missing, 'pt'].map(self.manual_pt_fixes)
            
            # Log standardization results
            total = len(df)
            standardized = df['reaction_term'].notna().sum()
            logging.info(f"Reaction standardization results:")
            logging.info(f"  Total terms: {total}")
            logging.info(f"  Standardized: {standardized} ({100*standardized/total:.1f}%)")
            logging.info(f"  Unstandardized: {total-standardized} ({100*(total-standardized)/total:.1f}%)")
        
        # 2. Add severity if available (after PT standardization)
        if 'outc_cod' in df.columns:
            df['severity'] = self.standardize_outcome(df['outc_cod'])
            severity_dist = df['severity'].value_counts()
            logging.info("Severity distribution:")
            for sev, count in severity_dist.items():
                logging.info(f"  {sev}: {count} ({100*count/len(df):.1f}%)")
        
        # 3. Memory optimization
        if 'reaction_term' in df.columns:
            df['reaction_term'] = df['reaction_term'].astype('category')
        if 'severity' in df.columns:
            df['severity'] = df['severity'].astype('category')
        
        return df

    def process_file(self, file_path: Path, data_type: str) -> pd.DataFrame:
        """Process a single FAERS file.
        
        Args:
            file_path: Path to the file
            data_type: Type of data ('demographics', 'drugs', 'reactions')
            
        Returns:
            Processed DataFrame
        """
        try:
            # Read file with optimized settings
            df = pd.read_csv(
                file_path,
                sep='$',
                dtype=str,
                na_values=['', 'NA', 'NULL'],
                keep_default_na=True,
                low_memory=False,
                encoding='utf-8'
            )
            
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
            
            return df
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            return pd.DataFrame()

    def standardize_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize demographics data.
        
        Args:
            df: Raw demographics DataFrame
            
        Returns:
            Standardized demographics DataFrame
        """
        # Convert columns to lowercase
        df.columns = df.columns.str.lower()
        
        # Standardize column names
        column_map = {
            'primaryid': 'primaryid',
            'caseid': 'caseid', 
            'caseversion': 'caseversion',
            'i_f_cod': 'i_f_code',
            'sex': 'sex',
            'age': 'age',
            'age_cod': 'age_code',
            'age_grp': 'age_group',
            'wt': 'weight',
            'wt_cod': 'weight_code',
            'reporter_country': 'reporter_country',
            'occr_country': 'occurrence_country',
            'event_dt': 'event_date',
            'rept_dt': 'report_date'
        }
        df = df.rename(columns=column_map)
        
        # Convert dates
        for date_col in ['event_date', 'report_date']:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], format='%Y%m%d', errors='coerce')
        
        return df
        
    def standardize_drugs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize drug data.
        
        Args:
            df: Raw drug DataFrame
            
        Returns:
            Standardized drug DataFrame
        """
        # Convert columns to lowercase
        df.columns = df.columns.str.lower()
        
        # Standardize column names
        column_map = {
            'primaryid': 'primaryid',
            'drug_seq': 'drug_seq',
            'role_cod': 'role_code',
            'drugname': 'drug_name',
            'prod_ai': 'active_ingredient',
            'val_vbm': 'verbatim_indication',
            'route': 'route',
            'dose_vbm': 'verbatim_dose',
            'dechal': 'dechallenge',
            'rechal': 'rechallenge',
            'lot_num': 'lot_number',
            'nda_num': 'nda_number',
            'exp_dt': 'expiration_date'
        }
        df = df.rename(columns=column_map)
        
        # Convert dates
        if 'expiration_date' in df.columns:
            df['expiration_date'] = pd.to_datetime(df['expiration_date'], format='%Y%m%d', errors='coerce')
            
        return df
        
    def standardize_reactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize reaction data.
        
        Args:
            df: Raw reaction DataFrame
            
        Returns:
            Standardized reaction DataFrame
        """
        # Convert columns to lowercase
        df.columns = df.columns.str.lower()
        
        # Standardize column names
        column_map = {
            'primaryid': 'primaryid',
            'pt': 'preferred_term',
            'drug_rec_act': 'drug_reaction_action'
        }
        df = df.rename(columns=column_map)
        
        return df
