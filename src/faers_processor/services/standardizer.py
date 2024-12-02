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
        self.external_dir = external_dir
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
                df = pd.read_csv(dict_path)
                self._drug_dictionary = dict(zip(df['original'], df['standard']))
            else:
                logging.warning(f"Drug dictionary not found at {dict_path}")
                self._drug_dictionary = {}
        return self._drug_dictionary

    def standardize_dates(self, data: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
        """Standardize date columns to datetime format.
        
        Args:
            data: Input DataFrame
            date_columns: List of column names containing dates
        
        Returns:
            DataFrame with standardized dates
        """
        df = data.copy()
        
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            for col in date_columns:
                if col in df.columns:
                    try:
                        # First try standard parsing
                        temp_series = pd.to_datetime(df[col], errors='coerce')
                        # Only assign if successful
                        if isinstance(temp_series, pd.Series):
                            df[col] = temp_series
                        else:
                            logging.warning(f"Skipping {col} - unexpected type after date parsing")
                    except Exception as e:
                        logging.warning(f"Error converting {col} to datetime: {str(e)}")
                        # Try custom date parsing for problematic formats
                        df[col] = df[col].apply(self._parse_custom_date)
        
        return df

    def _parse_custom_date(self, date_str: str) -> Optional[pd.Timestamp]:
        """Parse dates with custom formats.
        
        Args:
            date_str: Date string to parse
            
        Returns:
            Parsed timestamp or None if parsing fails
        """
        if pd.isna(date_str):
            return None

        try:
            # Try common FAERS date formats
            formats = [
                '%Y%m%d',
                '%Y-%m-%d',
                '%m/%d/%Y',
                '%d-%b-%Y',
                '%Y-%b-%d'
            ]

            for fmt in formats:
                try:
                    return pd.Timestamp(datetime.strptime(str(date_str), fmt))
                except ValueError:
                    continue

            return None

        except Exception as e:
            logging.debug(f"Could not parse date {date_str}: {str(e)}")
            return None

    def standardize_sex(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize sex values exactly as in R script."""
        df = data.copy()
        if 'sex' not in df.columns:
            return df
        
        # Exact R script logic: Demo[!sex %in% c("F","M")]$sex<- NA
        df.loc[~df['sex'].isin(['F', 'M']), 'sex'] = np.nan
        return df

    def standardize_age(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize age values exactly as in R script."""
        df = data.copy()
        
        # Exact R age corrector mapping from R script
        age_corrector_map = {
            'DEC': 3650,
            'YR': 365,
            'MON': 30.41667,
            'WK': 7,
            'DY': 1,
            'HR': 0.00011415525114155251,
            'SEC': 3.1709791983764586e-08,
            'MIN': 1.9025875190259e-06
        }
        
        # Create age_corrector column
        df['age_corrector'] = df['age_cod'].map(age_corrector_map)
        
        # Calculate age_in_days exactly as R script
        df['age_in_days'] = np.abs(pd.to_numeric(df['age'], errors='coerce')) * df['age_corrector']
        
        # Apply R script's plausibility check
        mask = (df['age_in_days'] > 122*365) & (df['age_cod'] != 'DEC')
        df.loc[mask, 'age_in_days'] = np.nan
        
        # Calculate age in years
        df['age_in_years'] = np.round(df['age_in_days'] / 365)
        
        # Create age groups exactly as R script
        df['age_grp_st'] = np.nan
        df.loc[df['age_in_years'].notna(), 'age_grp_st'] = 'E'
        df.loc[df['age_in_years'] < 65, 'age_grp_st'] = 'A'
        df.loc[df['age_in_years'] < 18, 'age_grp_st'] = 'T'
        df.loc[df['age_in_years'] < 12, 'age_grp_st'] = 'C'
        df.loc[df['age_in_years'] < 2, 'age_grp_st'] = 'I'
        df.loc[df['age_in_days'] < 28, 'age_grp_st'] = 'N'
        
        # Clean up temporary columns
        df = df.drop(['age_corrector', 'age', 'age_cod'], axis=1)
        df = df.rename(columns={'age_grp_st': 'age_grp'})
        
        return df

    def standardize_weight(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize weight values exactly as in R script."""
        df = data.copy()
        
        # Exact R weight corrector mapping
        wt_corrector_map = {
            'LBS': 0.453592,
            'IB': 0.453592,
            'KG': 1,
            'KGS': 1,
            'GMS': 0.001,
            'MG': 1e-06
        }
        
        # Set default corrector to 1 for NA values as in R
        df['wt_corrector'] = df['wt_cod'].map(wt_corrector_map).fillna(1)
        
        # Calculate weight in kgs exactly as R script
        df['wt_in_kgs'] = np.round(np.abs(pd.to_numeric(df['wt'], errors='coerce')) * df['wt_corrector'])
        
        # Apply R script's plausibility check
        df.loc[df['wt_in_kgs'] > 635, 'wt_in_kgs'] = np.nan
        
        # Clean up temporary columns
        df = df.drop(['wt_corrector', 'wt', 'wt_cod'], axis=1)
        
        return df

    def standardize_country(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize country codes exactly as in R script."""
        df = data.copy()
        
        # Load country mappings
        countries_df = pd.read_csv(self.external_dir / 'Manual_fix/countries.csv', sep=';')
        countries_df.loc[countries_df['country'].isna(), 'country'] = 'NA'  # Handle Namibia case
        
        # Create mapping dictionary
        country_map = dict(zip(countries_df['country'], countries_df['Country_Name']))
        
        # Apply mappings exactly as R script
        for col in ['occr_country', 'reporter_country']:
            if col in df.columns:
                df[col] = df[col].map(country_map)
        
        return df

    def standardize_occupation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize occupation codes exactly as in R script."""
        df = data.copy()
        
        # Exact R script logic for occupation codes
        valid_codes = ['MD', 'CN', 'OT', 'PH', 'HP', 'LW', 'RN']
        df.loc[~df['occp_cod'].isin(valid_codes), 'occp_cod'] = np.nan
        
        return df

    def remove_special_chars(self, text: str) -> str:
        """Remove special characters from text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return text
        # Use word characters, whitespace, and hyphen only
        return re.sub(r'[^\w\s-]', '', str(text))

    def standardize_boolean(self, value: Union[str, int]) -> Optional[bool]:
        """Standardize boolean values.
        
        Args:
            value: Input value
            
        Returns:
            Standardized boolean or None
        """
        if pd.isna(value):
            return None
        value = str(value).lower().strip()
        true_values = {'y', 'yes', 'true', '1', 't'}
        false_values = {'n', 'no', 'false', '0', 'f'}

        if value in true_values:
            return True
        if value in false_values:
            return False
        return None

    def handle_manufacturer_records(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process manufacturer records in demographics data.
        
        Args:
            data: Input demographics DataFrame
            
        Returns:
            Processed DataFrame
        """
        df = data.copy()

        # Sort by FDA date
        df = df.sort_values('fda_dt')

        # Group by case ID and get last record for each manufacturer
        last_mfr_records = df.groupby(['caseid', 'mfr_sndr']).tail(1)

        # Count records with missing manufacturer
        missing_mfr_sndr = df[df['mfr_sndr'].isna()]

        # Log statistics
        total_cases = len(df)
        unique_cases = df['caseid'].nunique()
        logging.info(f"Total records: {total_cases}")
        logging.info(f"Unique cases: {unique_cases}")
        logging.info(f"Records with missing mfr_sndr: {len(missing_mfr_sndr)}")
        logging.info(f"Unique manufacturer combinations: {len(last_mfr_records)}")

        return df

    def check_date_validity(self, date_str: str, min_year: int = 1900) -> bool:
        """Check if a date string is valid.
        
        Args:
            date_str: Date string to check
            min_year: Minimum valid year (default: 1900)
            
        Returns:
            bool: True if date is valid, False otherwise
        """
        if pd.isna(date_str):
            return False

        try:
            date_str = str(date_str).zfill(8)
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])

            # Check basic ranges
            if not (min_year <= year <= datetime.now().year):
                return False
            if not (1 <= month <= 12):
                return False
            if not (1 <= day <= 31):
                return False

            # Check month-specific day ranges
            days_in_month = {
                2: 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
                4: 30, 6: 30, 9: 30, 11: 30
            }
            max_days = days_in_month.get(month, 31)
            if day > max_days:
                return False

            return True

        except (ValueError, TypeError):
            return False

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
                df[col] = self.check_date(df[col], max_date)

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
            self.country_map = pd.read_csv(country_file, sep=';', dtype=str).set_index('country')['Country_Name'].to_dict()

        # Load occupation codes (matching R script)
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

        # Load SOC data
        soc_file = meddra_dir / 'soc.asc'
        if soc_file.exists():
            self.soc_data = pd.read_csv(soc_file, sep='$', dtype=str, usecols=[0,1,2])
            self.soc_data.columns = ['soc_code', 'soc_name', 'soc_abbrev']

        # Load PT data
        pt_file = meddra_dir / 'pt.asc'
        if pt_file.exists():
            self.pt_data = pd.read_csv(pt_file, sep='$', dtype=str, usecols=[0,1,3])
            self.pt_data.columns = ['pt_code', 'pt_name', 'pt_soc_code']

        # Load LLT data
        llt_file = meddra_dir / 'llt.asc'
        if llt_file.exists():
            self.llt_data = pd.read_csv(llt_file, sep='$', dtype=str, usecols=[0,1,2])
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
            manual_fixes = pd.read_csv(manual_fix_file)
            self.manual_pt_fixes = dict(zip(manual_fixes['original'].str.lower(), 
                                          manual_fixes['standardized']))

    def _load_diana_dictionary(self):
        """Load and prepare the DiAna drug dictionary."""
        try:
            dict_path = self.external_dir / 'DiAna_dictionary' / 'drugnames_standardized.csv'
            if not dict_path.exists():
                logging.error(f"DiAna dictionary not found at {dict_path}")
                return
            
            # Read with error_bad_lines=False to skip problematic rows
            self.diana_dict = pd.read_csv(dict_path, 
                                        dtype={'drugname': str, 'Substance': str},
                                        on_bad_lines='skip',  # Skip problematic lines
                                        delimiter=';')  # Use semicolon delimiter
            
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
        """Standardize PT terms exactly as in R script."""
        df = df.copy()
        
        # Load MedDRA PT list
        meddra_df = pd.read_csv(self.external_dir / 'Dictionaries/MedDRA/meddra.csv', sep=';')
        pt_list = pd.Series(meddra_df['pt'].unique()).str.lower().str.strip().unique()
        
        # Calculate PT frequencies
        df[pt_variable] = df[pt_variable].str.lower().str.strip()
        pt_freq = (df[~df[pt_variable].isna()]
                   .groupby(pt_variable).size()
                   .reset_index(name='N')
                   .sort_values('N', ascending=False))
        
        # Check if PTs are standardized
        pt_freq['standard_pt'] = np.where(pt_freq[pt_variable].isin(pt_list), 
                                         pt_freq[pt_variable], 
                                         np.nan)
        pt_freq['freq'] = np.round(pt_freq['N'] / pt_freq['N'].sum() * 100, 2)
        
        # Get unstandardized PTs
        not_pts = pt_freq[pt_freq['standard_pt'].isna()][[pt_variable, 'N', 'freq']]
        
        # Calculate initial non-standardized percentage
        initial_nonstd_pct = np.round(not_pts['N'].sum() * 100 / len(df[~df[pt_variable].isna()]), 3)
        logging.info(f"Initial non-standardized PT percentage: {initial_nonstd_pct}%")
        
        # Try to translate through LLTs
        llt_mappings = meddra_df[['pt', 'llt']].copy()
        llt_mappings.columns = ['standard_pt', pt_variable]
        not_pts = not_pts.merge(llt_mappings, on=pt_variable, how='left')
        not_llts = not_pts[not_pts['standard_pt'].isna()].drop('standard_pt', axis=1)
        
        # Load and apply manual fixes
        manual_fixes = pd.read_csv(self.external_dir / 'Manual_fix/pt_fixed.csv', sep=';')
        manual_fixes = manual_fixes[[pt_variable, 'standard_pt']]
        
        not_llts = not_llts.merge(manual_fixes, on=pt_variable, how='left')
        still_unstandardized = not_llts[not_llts['standard_pt'].isna()][[pt_variable, 'standard_pt']]
        
        # Combine all standardization sources
        pt_fixed = pd.concat([
            manual_fixes,
            still_unstandardized,
            not_pts[~not_pts['standard_pt'].isna()][[pt_variable, 'standard_pt']]
        ]).drop_duplicates()
        
        # Check for duplicates
        duplicates = pt_fixed[pt_fixed[pt_variable].duplicated()]
        if not duplicates.empty:
            logging.warning(f"Duplicate PT mappings found: {duplicates[pt_variable].tolist()}")
        
        # Apply standardization to original data
        df['pt_temp'] = df[pt_variable].str.lower().str.strip()
        df = df.merge(pt_fixed, left_on='pt_temp', right_on=pt_variable, how='left')
        df[pt_variable] = df['standard_pt'].fillna(df['pt_temp'])
        df = df.drop(['pt_temp', 'standard_pt'], axis=1)
        
        # Calculate final standardization percentage
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
        
        # Ensure drugname column exists
        if 'drugname' not in df.columns and 'drug_name' in df.columns:
            df = df.rename(columns={'drug_name': 'drugname'})
        
        if 'drugname' not in df.columns:
            logging.error("No drugname or drug_name column found in DataFrame")
            return df
        
        # Clean drug names
        df['drugname'] = df['drugname'].fillna('').astype(str).apply(self._clean_drugname)
        
        # Map to standardized substances using the precomputed mapping
        df['Substance'] = df['drugname'].str.lower().str.strip().map(self.drug_map).fillna('UNKNOWN')
        
        # Handle multi-substance drugs
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
        
        # Mark trial drugs
        df['trial'] = df['Substance'].str.contains(', trial', na=False)
        df['Substance'] = df['Substance'].str.replace(', trial', '', regex=False)
        
        # Convert to categorical for memory efficiency
        df['drugname'] = df['drugname'].astype('category')
        df['Substance'] = df['Substance'].astype('category')
        if 'prod_ai' in df.columns:
            df['prod_ai'] = df['prod_ai'].astype('category')
        
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
        
        # Define age unit conversion factors (matching R script)
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
        df['age_in_days'] = pd.to_numeric(df['age'].abs()) * df['age_corrector']
        
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
        
        # Apply age group rules (matching R script)
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

    def process_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process demographics data.
        
        Args:
            df: Raw demographics DataFrame
            
        Returns:
            Processed demographics DataFrame
        """
        df = df.copy()
        
        # Standardize dates
        date_cols = ['init_fda_dt', 'fda_dt', 'event_dt', 'mfr_dt']
        for col in date_cols:
            if col in df.columns:
                df[col] = self.standardize_dates(df[col])
        
        # Standardize sex
        if 'sex' in df.columns:
            df['sex'] = self.standardize_sex(df['sex'])
            
        # Standardize age
        if 'age' in df.columns and 'age_cod' in df.columns:
            df['age'] = self.standardize_age(df['age'], df['age_cod'])
            
        # Standardize weight
        if 'wt' in df.columns and 'wt_cod' in df.columns:
            df['weight'] = self.standardize_weight(df['wt'], df['wt_cod'])
            
        # Standardize country codes
        if 'reporter_country' in df.columns:
            df['reporter_country'] = self.standardize_country(df['reporter_country'])
            
        # Standardize occupations
        if 'occp_cod' in df.columns:
            df['occupation'] = self.standardize_occupation(df['occp_cod'])
            
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

    def process_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process demographics data following R script order exactly.
        
        Args:
            df: Raw demographics DataFrame
            
        Returns:
            Processed demographics DataFrame
        """
        df = df.copy()
        
        # 1. Standardize dates first (as they may affect age calculations)
        date_cols = ['init_fda_dt', 'fda_dt', 'event_dt', 'mfr_dt']
        for col in date_cols:
            if col in df.columns:
                df[col] = self.standardize_dates(df[col])
        
        # 2. Standardize sex (before age groups as it may be used in age-sex analysis)
        if 'sex' in df.columns:
            df = self.standardize_sex(df)
        
        # 3. Process age and create age groups
        if 'age' in df.columns and 'age_cod' in df.columns:
            df = self.standardize_age(df)
            df = self.standardize_age_groups(df)
        
        # 4. Standardize country codes
        if 'reporter_country' in df.columns:
            df = self.standardize_country(df)
        
        # 5. Standardize occupation codes (last as it's least critical)
        if 'occp_cod' in df.columns:
            df = self.standardize_occupation(df)
        
        # 6. Memory optimization
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category')
        
        # 7. Log processing results
        logging.info("Demographics processing results:")
        if 'sex' in df.columns:
            sex_dist = df['sex'].value_counts(dropna=False)
            logging.info("Sex distribution:")
            for sex, count in sex_dist.items():
                logging.info(f"  {sex}: {count} ({100*count/len(df):.1f}%)")
        
        if 'age_grp' in df.columns:
            age_dist = df['age_grp'].value_counts(dropna=False)
            logging.info("Age group distribution:")
            for grp, count in age_dist.items():
                logging.info(f"  {grp}: {count} ({100*count/len(df):.1f}%)")
        
        return df

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
                df[col] = self.check_date(df[col], max_date)

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
            self.country_map = pd.read_csv(country_file, sep=';', dtype=str).set_index('country')['Country_Name'].to_dict()

        # Load occupation codes (matching R script)
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

        # Load SOC data
        soc_file = meddra_dir / 'soc.asc'
        if soc_file.exists():
            self.soc_data = pd.read_csv(soc_file, sep='$', dtype=str, usecols=[0,1,2])
            self.soc_data.columns = ['soc_code', 'soc_name', 'soc_abbrev']

        # Load PT data
        pt_file = meddra_dir / 'pt.asc'
        if pt_file.exists():
            self.pt_data = pd.read_csv(pt_file, sep='$', dtype=str, usecols=[0,1,3])
            self.pt_data.columns = ['pt_code', 'pt_name', 'pt_soc_code']

        # Load LLT data
        llt_file = meddra_dir / 'llt.asc'
        if llt_file.exists():
            self.llt_data = pd.read_csv(llt_file, sep='$', dtype=str, usecols=[0,1,2])
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
            manual_fixes = pd.read_csv(manual_fix_file)
            self.manual_pt_fixes = dict(zip(manual_fixes['original'].str.lower(), 
                                          manual_fixes['standardized']))

    def _load_diana_dictionary(self):
        """Load and prepare the DiAna drug dictionary."""
        try:
            dict_path = self.external_dir / 'DiAna_dictionary' / 'drugnames_standardized.csv'
            if not dict_path.exists():
                logging.error(f"DiAna dictionary not found at {dict_path}")
                return
            
            # Read with error_bad_lines=False to skip problematic rows
            self.diana_dict = pd.read_csv(dict_path, 
                                        dtype={'drugname': str, 'Substance': str},
                                        on_bad_lines='skip',  # Skip problematic lines
                                        delimiter=';')  # Use semicolon delimiter
            
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
        """Standardize PT terms exactly as in R script."""
        df = df.copy()
        
        # Load MedDRA PT list
        meddra_df = pd.read_csv(self.external_dir / 'Dictionaries/MedDRA/meddra.csv', sep=';')
        pt_list = pd.Series(meddra_df['pt'].unique()).str.lower().str.strip().unique()
        
        # Calculate PT frequencies
        df[pt_variable] = df[pt_variable].str.lower().str.strip()
        pt_freq = (df[~df[pt_variable].isna()]
                   .groupby(pt_variable).size()
                   .reset_index(name='N')
                   .sort_values('N', ascending=False))
        
        # Check if PTs are standardized
        pt_freq['standard_pt'] = np.where(pt_freq[pt_variable].isin(pt_list), 
                                         pt_freq[pt_variable], 
                                         np.nan)
        pt_freq['freq'] = np.round(pt_freq['N'] / pt_freq['N'].sum() * 100, 2)
        
        # Get unstandardized PTs
        not_pts = pt_freq[pt_freq['standard_pt'].isna()][[pt_variable, 'N', 'freq']]
        
        # Calculate initial non-standardized percentage
        initial_nonstd_pct = np.round(not_pts['N'].sum() * 100 / len(df[~df[pt_variable].isna()]), 3)
        logging.info(f"Initial non-standardized PT percentage: {initial_nonstd_pct}%")
        
        # Try to translate through LLTs
        llt_mappings = meddra_df[['pt', 'llt']].copy()
        llt_mappings.columns = ['standard_pt', pt_variable]
        not_pts = not_pts.merge(llt_mappings, on=pt_variable, how='left')
        not_llts = not_pts[not_pts['standard_pt'].isna()].drop('standard_pt', axis=1)
        
        # Load and apply manual fixes
        manual_fixes = pd.read_csv(self.external_dir / 'Manual_fix/pt_fixed.csv', sep=';')
        manual_fixes = manual_fixes[[pt_variable, 'standard_pt']]
        
        not_llts = not_llts.merge(manual_fixes, on=pt_variable, how='left')
        still_unstandardized = not_llts[not_llts['standard_pt'].isna()][[pt_variable, 'standard_pt']]
        
        # Combine all standardization sources
        pt_fixed = pd.concat([
            manual_fixes,
            still_unstandardized,
            not_pts[~not_pts['standard_pt'].isna()][[pt_variable, 'standard_pt']]
        ]).drop_duplicates()
        
        # Check for duplicates
        duplicates = pt_fixed[pt_fixed[pt_variable].duplicated()]
        if not duplicates.empty:
            logging.warning(f"Duplicate PT mappings found: {duplicates[pt_variable].tolist()}")
        
        # Apply standardization to original data
        df['pt_temp'] = df[pt_variable].str.lower().str.strip()
        df = df.merge(pt_fixed, left_on='pt_temp', right_on=pt_variable, how='left')
        df[pt_variable] = df['standard_pt'].fillna(df['pt_temp'])
        df = df.drop(['pt_temp', 'standard_pt'], axis=1)
        
        # Calculate final standardization percentage
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
        
        # Ensure drugname column exists
        if 'drugname' not in df.columns and 'drug_name' in df.columns:
            df = df.rename(columns={'drug_name': 'drugname'})
        
        if 'drugname' not in df.columns:
            logging.error("No drugname or drug_name column found in DataFrame")
            return df
        
        # Clean drug names
        df['drugname'] = df['drugname'].fillna('').astype(str).apply(self._clean_drugname)
        
        # Map to standardized substances using the precomputed mapping
        df['Substance'] = df['drugname'].str.lower().str.strip().map(self.drug_map).fillna('UNKNOWN')
        
        # Handle multi-substance drugs
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
        
        # Mark trial drugs
        df['trial'] = df['Substance'].str.contains(', trial', na=False)
        df['Substance'] = df['Substance'].str.replace(', trial', '', regex=False)
        
        # Convert to categorical for memory efficiency
        df['drugname'] = df['drugname'].astype('category')
        df['Substance'] = df['Substance'].astype('category')
        if 'prod_ai' in df.columns:
            df['prod_ai'] = df['prod_ai'].astype('category')
        
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
        
        # Define age unit conversion factors (matching R script)
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
        df['age_in_days'] = pd.to_numeric(df['age'].abs()) * df['age_corrector']
        
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
        
        # Apply age group rules (matching R script)
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

    def process_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process demographics data.
        
        Args:
            df: Raw demographics DataFrame
            
        Returns:
            Processed demographics DataFrame
        """
        df = df.copy()
        
        # Standardize dates
        date_cols = ['init_fda_dt', 'fda_dt', 'event_dt', 'mfr_dt']
        for col in date_cols:
            if col in df.columns:
                df[col] = self.standardize_dates(df[col])
        
        # Standardize sex
        if 'sex' in df.columns:
            df['sex'] = self.standardize_sex(df['sex'])
            
        # Standardize age
        if 'age' in df.columns and 'age_cod' in df.columns:
            df['age'] = self.standardize_age(df['age'], df['age_cod'])
            
        # Standardize weight
        if 'wt' in df.columns and 'wt_cod' in df.columns:
            df['weight'] = self.standardize_weight(df['wt'], df['wt_cod'])
            
        # Standardize country codes
        if 'reporter_country' in df.columns:
            df['reporter_country'] = self.standardize_country(df['reporter_country'])
            
        # Standardize occupations
        if 'occp_cod' in df.columns:
            df['occupation'] = self.standardize_occupation(df['occp_cod'])
            
        return df
        
    def process_drugs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process drug data.
        
        Args:
            df: Raw drug DataFrame
            
        Returns:
            Processed drug DataFrame
        """
        df = df.copy()
        
        # Standardize drug names and substances
        df = self.standardize_drugs(df)
        
        # Standardize routes
        if 'route' in df.columns:
            df['route'] = self.standardize_route(df['route'])
            
        # Standardize dose forms
        if 'dose_form' in df.columns:
            df['dose_form'] = self.standardize_dose_form(df['dose_form'])
            
        # Convert dosages to standard units
        if all(col in df.columns for col in ['dose_amt', 'dose_unit']):
            df['dose_std'] = self.standardize_dose(df['dose_amt'], df['dose_unit'])
            
        return df
        
    def process_reactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process reaction data.
        
        Args:
            df: Raw reaction DataFrame
            
        Returns:
            Processed reaction DataFrame
        """
        df = df.copy()
        
        # Standardize reaction terms using MedDRA
        if 'pt' in df.columns:
            df['reaction_term'] = self.standardize_reaction(df['pt'])
            
        # Add reaction severity if available
        if 'outc_cod' in df.columns:
            df['severity'] = self.standardize_outcome(df['outc_cod'])
            
        return df

    def process_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process demographics data.
        
        Args:
            df: Raw demographics DataFrame
            
        Returns:
            Processed demographics DataFrame
        """
        df = df.copy()
        
        # Standardize dates
        date_cols = ['init_fda_dt', 'fda_dt', 'event_dt', 'mfr_dt']
        for col in date_cols:
            if col in df.columns:
                df[col] = self.standardize_dates(df[col])
        
        # Standardize sex
        if 'sex' in df.columns:
            df['sex'] = self.standardize_sex(df['sex'])
            
        # Standardize age
        if 'age' in df.columns and 'age_cod' in df.columns:
            df['age'] = self.standardize_age(df['age'], df['age_cod'])
            
        # Standardize weight
        if 'wt' in df.columns and 'wt_cod' in df.columns:
            df['weight'] = self.standardize_weight(df['wt'], df['wt_cod'])
            
        # Standardize country codes
        if 'reporter_country' in df.columns:
            df['reporter_country'] = self.standardize_country(df['reporter_country'])
            
        # Standardize occupations
        if 'occp_cod' in df.columns:
            df['occupation'] = self.standardize_occupation(df['occp_cod'])
            
        return df
        
    def process_drugs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process drug data.
        
        Args:
            df: Raw drug DataFrame
            
        Returns:
            Processed drug DataFrame
        """
        df = df.copy()
        
        # Standardize drug names and substances
        df = self.standardize_drugs(df)
        
        # Standardize routes
        if 'route' in df.columns:
            df['route'] = self.standardize_route(df['route'])
            
        # Standardize dose forms
        if 'dose_form' in df.columns:
            df['dose_form'] = self.standardize_dose_form(df['dose_form'])
            
        # Convert dosages to standard units
        if all(col in df.columns for col in ['dose_amt', 'dose_unit']):
            df['dose_std'] = self.standardize_dose(df['dose_amt'], df['dose_unit'])
            
        return df
        
    def process_reactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process reaction data.
        
        Args:
            df: Raw reaction DataFrame
            
        Returns:
            Processed reaction DataFrame
        """
        df = df.copy()
        
        # Standardize reaction terms using MedDRA
        if 'pt' in df.columns:
            df['reaction_term'] = self.standardize_reaction(df['pt'])
            
        # Add reaction severity if available
        if 'outc_cod' in df.columns:
            df['severity'] = self.standardize_outcome(df['outc_cod'])
            
        return df

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
                df[col] = self.check_date(df[col], max_date)

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
            self.country_map = pd.read_csv(country_file, sep=';', dtype=str).set_index('country')['Country_Name'].to_dict()

        # Load occupation codes (matching R script)
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

        # Load SOC data
        soc_file = meddra_dir / 'soc.asc'
        if soc_file.exists():
            self.soc_data = pd.read_csv(soc_file, sep='$', dtype=str, usecols=[0,1,2])
            self.soc_data.columns = ['soc_code', 'soc_name', 'soc_abbrev']

        # Load PT data
        pt_file = meddra_dir / 'pt.asc'
        if pt_file.exists():
            self.pt_data = pd.read_csv(pt_file, sep='$', dtype=str, usecols=[0,1,3])
            self.pt_data.columns = ['pt_code', 'pt_name', 'pt_soc_code']

        # Load LLT data
        llt_file = meddra_dir / 'llt.asc'
        if llt_file.exists():
            self.llt_data = pd.read_csv(llt_file, sep='$', dtype=str, usecols=[0,1,2])
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
            manual_fixes = pd.read_csv(manual_fix_file)
            self.manual_pt_fixes = dict(zip(manual_fixes['original'].str.lower(), 
                                          manual_fixes['standardized']))

    def _load_diana_dictionary(self):
        """Load and prepare the DiAna drug dictionary."""
        try:
            dict_path = self.external_dir / 'DiAna_dictionary' / 'drugnames_standardized.csv'
            if not dict_path.exists():
                logging.error(f"DiAna dictionary not found at {dict_path}")
                return
            
            # Read with error_bad_lines=False to skip problematic rows
            self.diana_dict = pd.read_csv(dict_path, 
                                        dtype={'drugname': str, 'Substance': str},
                                        on_bad_lines='skip',  # Skip problematic lines
                                        delimiter=';')  # Use semicolon delimiter
            
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
        """Standardize PT terms exactly as in R script."""
        df = df.copy()
        
        # Load MedDRA PT list
        meddra_df = pd.read_csv(self.external_dir / 'Dictionaries/MedDRA/meddra.csv', sep=';')
        pt_list = pd.Series(meddra_df['pt'].unique()).str.lower().str.strip().unique()
        
        # Calculate PT frequencies
        df[pt_variable] = df[pt_variable].str.lower().str.strip()
        pt_freq = (df[~df[pt_variable].isna()]
                   .groupby(pt_variable).size()
                   .reset_index(name='N')
                   .sort_values('N', ascending=False))
        
        # Check if PTs are standardized
        pt_freq['standard_pt'] = np.where(pt_freq[pt_variable].isin(pt_list), 
                                         pt_freq[pt_variable], 
                                         np.nan)
        pt_freq['freq'] = np.round(pt_freq['N'] / pt_freq['N'].sum() * 100, 2)
        
        # Get unstandardized PTs
        not_pts = pt_freq[pt_freq['standard_pt'].isna()][[pt_variable, 'N', 'freq']]
        
        # Calculate initial non-standardized percentage
        initial_nonstd_pct = np.round(not_pts['N'].sum() * 100 / len(df[~df[pt_variable].isna()]), 3)
        logging.info(f"Initial non-standardized PT percentage: {initial_nonstd_pct}%")
        
        # Try to translate through LLTs
        llt_mappings = meddra_df[['pt', 'llt']].copy()
        llt_mappings.columns = ['standard_pt', pt_variable]
        not_pts = not_pts.merge(llt_mappings, on=pt_variable, how='left')
        not_llts = not_pts[not_pts['standard_pt'].isna()].drop('standard_pt', axis=1)
        
        # Load and apply manual fixes
        manual_fixes = pd.read_csv(self.external_dir / 'Manual_fix/pt_fixed.csv', sep=';')
        manual_fixes = manual_fixes[[pt_variable, 'standard_pt']]
        
        not_llts = not_llts.merge(manual_fixes, on=pt_variable, how='left')
        still_unstandardized = not_llts[not_llts['standard_pt'].isna()][[pt_variable, 'standard_pt']]
        
        # Combine all standardization sources
        pt_fixed = pd.concat([
            manual_fixes,
            still_unstandardized,
            not_pts[~not_pts['standard_pt'].isna()][[pt_variable, 'standard_pt']]
        ]).drop_duplicates()
        
        # Check for duplicates
        duplicates = pt_fixed[pt_fixed[pt_variable].duplicated()]
        if not duplicates.empty:
            logging.warning(f"Duplicate PT mappings found: {duplicates[pt_variable].tolist()}")
        
        # Apply standardization to original data
        df['pt_temp'] = df[pt_variable].str.lower().str.strip()
        df = df.merge(pt_fixed, left_on='pt_temp', right_on=pt_variable, how='left')
        df[pt_variable] = df['standard_pt'].fillna(df['pt_temp'])
        df = df.drop(['pt_temp', 'standard_pt'], axis=1)
        
        # Calculate final standardization percentage
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
        
        # Ensure drugname column exists
        if 'drugname' not in df.columns and 'drug_name' in df.columns:
            df = df.rename(columns={'drug_name': 'drugname'})
        
        if 'drugname' not in df.columns:
            logging.error("No drugname or drug_name column found in DataFrame")
            return df
        
        # Clean drug names
        df['drugname'] = df['drugname'].fillna('').astype(str).apply(self._clean_drugname)
        
        # Map to standardized substances using the precomputed mapping
        df['Substance'] = df['drugname'].str.lower().str.strip().map(self.drug_map).fillna('UNKNOWN')
        
        # Handle multi-substance drugs
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
        
        # Mark trial drugs
        df['trial'] = df['Substance'].str.contains(', trial', na=False)
        df['Substance'] = df['Substance'].str.replace(', trial', '', regex=False)
        
        # Convert to categorical for memory efficiency
        df['drugname'] = df['drugname'].astype('category')
        df['Substance'] = df['Substance'].astype('category')
        if 'prod_ai' in df.columns:
            df['prod_ai'] = df['prod_ai'].astype('category')
        
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
        
        # Define age unit conversion factors (matching R script)
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
        df['age_in_days'] = pd.to_numeric(df['age'].abs()) * df['age_corrector']
        
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
        
        # Apply age group rules (matching R script)
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

    def process_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process demographics data.
        
        Args:
            df: Raw demographics DataFrame
            
        Returns:
            Processed demographics DataFrame
        """
        df = df.copy()
        
        # Standardize dates
        date_cols = ['init_fda_dt', 'fda_dt', 'event_dt', 'mfr_dt']
        for col in date_cols:
            if col in df.columns:
                df[col] = self.standardize_dates(df[col])
        
        # Standardize sex
        if 'sex' in df.columns:
            df['sex'] = self.standardize_sex(df['sex'])
            
        # Standardize age
        if 'age' in df.columns and 'age_cod' in df.columns:
            df['age'] = self.standardize_age(df['age'], df['age_cod'])
            
        # Standardize weight
        if 'wt' in df.columns and 'wt_cod' in df.columns:
            df['weight'] = self.standardize_weight(df['wt'], df['wt_cod'])
            
        # Standardize country codes
        if 'reporter_country' in df.columns:
            df['reporter_country'] = self.standardize_country(df['reporter_country'])
            
        # Standardize occupations
        if 'occp_cod' in df.columns:
            df['occupation'] = self.standardize_occupation(df['occp_cod'])
            
        return df
        
    def process_drugs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process drug data.
        
        Args:
            df: Raw drug DataFrame
            
        Returns:
            Processed drug DataFrame
        """
        df = df.copy()
        
        # Standardize drug names and substances
        df = self.standardize_drugs(df)
        
        # Standardize routes
        if 'route' in df.columns:
            df['route'] = self.standardize_route(df['route'])
            
        # Standardize dose forms
        if 'dose_form' in df.columns:
            df['dose_form'] = self.standardize_dose_form(df['dose_form'])
            
        # Convert dosages to standard units
        if all(col in df.columns for col in ['dose_amt', 'dose_unit']):
            df['dose_std'] = self.standardize_dose(df['dose_amt'], df['dose_unit'])
            
        return df
        
    def process_reactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process reaction data.
        
        Args:
            df: Raw reaction DataFrame
            
        Returns:
            Processed reaction DataFrame
        """
        df = df.copy()
        
        # Standardize reaction terms using MedDRA
        if 'pt' in df.columns:
            df['reaction_term'] = self.standardize_reaction(df['pt'])
            
        # Add reaction severity if available
        if 'outc_cod' in df.columns:
            df['severity'] = self.standardize_outcome(df['outc_cod'])
            
        return df

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
                df[col] = self.check_date(df[col], max_date)

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
            self.country_map = pd.read_csv(country_file, sep=';', dtype=str).set_index('country')['Country_Name'].to_dict()

        # Load occupation codes (matching R script)
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

        # Load SOC data
        soc_file = meddra_dir / 'soc.asc'
        if soc_file.exists():
            self.soc_data = pd.read_csv(soc_file, sep='$', dtype=str, usecols=[0,1,2])
            self.soc_data.columns = ['soc_code', 'soc_name', 'soc_abbrev']

        # Load PT data
        pt_file = meddra_dir / 'pt.asc'
        if pt_file.exists():
            self.pt_data = pd.read_csv(pt_file, sep='$', dtype=str, usecols=[0,1,3])
            self.pt_data.columns = ['pt_code', 'pt_name', 'pt_soc_code']

        # Load LLT data
        llt_file = meddra_dir / 'llt.asc'
        if llt_file.exists():
            self.llt_data = pd.read_csv(llt_file, sep='$', dtype=str, usecols=[0,1,2])
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
            manual_fixes = pd.read_csv(manual_fix_file)
            self.manual_pt_fixes = dict(zip(manual_fixes['original'].str.lower(), 
                                          manual_fixes['standardized']))

    def _load_diana_dictionary(self):
        """Load and prepare the DiAna drug dictionary."""
        try:
            dict_path = self.external_dir / 'DiAna_dictionary' / 'drugnames_standardized.csv'
            if not dict_path.exists():
                logging.error(f"DiAna dictionary not found at {dict_path}")
                return
            
            # Read with error_bad_lines=False to skip problematic rows
            self.diana_dict = pd.read_csv(dict_path, 
                                        dtype={'drugname': str, 'Substance': str},
                                        on_bad_lines='skip',  # Skip problematic lines
                                        delimiter=';')  # Use semicolon delimiter
            
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
        """Standardize PT terms exactly as in R script."""
        df = df.copy()
        
        # Load MedDRA PT list
        meddra_df = pd.read_csv(self.external_dir / 'Dictionaries/MedDRA/meddra.csv', sep=';')
        pt_list = pd.Series(meddra_df['pt'].unique()).str.lower().str.strip().unique()
        
        # Calculate PT frequencies
        df[pt_variable] = df[pt_variable].str.lower().str.strip()
        pt_freq = (df[~df[pt_variable].isna()]
                   .groupby(pt_variable).size()
                   .reset_index(name='N')
                   .sort_values('N', ascending=False))
        
        # Check if PTs are standardized
        pt_freq['standard_pt'] = np.where(pt_freq[pt_variable].isin(pt_list), 
                                         pt_freq[pt_variable], 
                                         np.nan)
        pt_freq['freq'] = np.round(pt_freq['N'] / pt_freq['N'].sum() * 100, 2)
        
        # Get unstandardized PTs
        not_pts = pt_freq[pt_freq['standard_pt'].isna()][[pt_variable, 'N', 'freq']]
        
        # Calculate initial non-standardized percentage
        initial_nonstd_pct = np.round(not_pts['N'].sum() * 100 / len(df[~df[pt_variable].isna()]), 3)
        logging.info(f"Initial non-standardized PT percentage: {initial_nonstd_pct}%")
        
        # Try to translate through LLTs
        llt_mappings = meddra_df[['pt', 'llt']].copy()
        llt_mappings.columns = ['standard_pt', pt_variable]
        not_pts = not_pts.merge(llt_mappings, on=pt_variable, how='left')
        not_llts = not_pts[not_pts['standard_pt'].isna()].drop('standard_pt', axis=1)
        
        # Load and apply manual fixes
        manual_fixes = pd.read_csv(self.external_dir / 'Manual_fix/pt_fixed.csv', sep=';')
        manual_fixes = manual_fixes[[pt_variable, 'standard_pt']]
        
        not_llts = not_llts.merge(manual_fixes, on=pt_variable, how='left')
        still_unstandardized = not_llts[not_llts['standard_pt'].isna()][[pt_variable, 'standard_pt']]
        
        # Combine all standardization sources
        pt_fixed = pd.concat([
            manual_fixes,
            still_unstandardized,
            not_pts[~not_pts['standard_pt'].isna()][[pt_variable, 'standard_pt']]
        ]).drop_duplicates()
        
        # Check for duplicates
        duplicates = pt_fixed[pt_fixed[pt_variable].duplicated()]
        if not duplicates.empty:
            logging.warning(f"Duplicate PT mappings found: {duplicates[pt_variable].tolist()}")
        
        # Apply standardization to original data
        df['pt_temp'] = df[pt_variable].str.lower().str.strip()
        df = df.merge(pt_fixed, left_on='pt_temp', right_on=pt_variable, how='left')
        df[pt_variable] = df['standard_pt'].fillna(df['pt_temp'])
        df = df.drop(['pt_temp', 'standard_pt'], axis=1)
        
        # Calculate final standardization percentage
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
        
        # Ensure drugname column exists
        if 'drugname' not in df.columns and 'drug_name' in df.columns:
            df = df.rename(columns={'drug_name': 'drugname'})
        
        if 'drugname' not in df.columns:
            logging.error("No drugname or drug_name column found in DataFrame")
            return df
        
        # Clean drug names
        df['drugname'] = df['drugname'].fillna('').astype(str).apply(self._clean_drugname)
        
        # Map to standardized substances using the precomputed mapping
        df['Substance'] = df['drugname'].str.lower().str.strip().map(self.drug_map).fillna('UNKNOWN')
        
        # Handle multi-substance drugs
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
        
        # Mark trial drugs
        df['trial'] = df['Substance'].str.contains(', trial', na=False)
        df['Substance'] = df['Substance'].str.replace(', trial', '', regex=False)
        
        # Convert to categorical for memory efficiency
        df['drugname'] = df['drugname'].astype('category')
        df['Substance'] = df['Substance'].astype('category')
        if 'prod_ai' in df.columns:
            df['prod_ai'] = df['prod_ai'].astype('category')
        
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
        
        # Define age unit conversion factors (matching R script)
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
        df['age_in_days'] = pd.to_numeric(df['age'].abs()) * df['age_corrector']
        
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
        
        # Apply age group rules (matching R script)
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

    def process_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process demographics data.
        
        Args:
            df: Raw demographics DataFrame
            
        Returns:
            Processed demographics DataFrame
        """
        df = df.copy()
        
        # Standardize dates
        date_cols = ['init_fda_dt', 'fda_dt', 'event_dt', 'mfr_dt']
        for col in date_cols:
            if col in df.columns:
                df[col] = self.standardize_dates(df[col])
        
        # Standardize sex
        if 'sex' in df.columns:
            df['sex'] = self.standardize_sex(df['sex'])
            
        # Standardize age
        if 'age' in df.columns and 'age_cod' in df.columns:
            df['age'] = self.standardize_age(df['age'], df['age_cod'])
            
        # Standardize weight
        if 'wt' in df.columns and 'wt_cod' in df.columns:
            df['weight'] = self.standardize_weight(df['wt'], df['wt_cod'])
            
        # Standardize country codes
        if 'reporter_country' in df.columns:
            df['reporter_country'] = self.standardize_country(df['reporter_country'])
            
        # Standardize occupations
        if 'occp_cod' in df.columns:
            df['occupation'] = self.standardize_occupation(df['occp_cod'])
            
        return df
        
    def process_drugs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process drug data.
        
        Args:
            df: Raw drug DataFrame
            
        Returns:
            Processed drug DataFrame
        """
        df = df.copy()
        
        # Standardize drug names and substances
        df = self.standardize_drugs(df)
        
        # Standardize routes
        if 'route' in df.columns:
            df['route'] = self.standardize_route(df['route'])
            
        # Standardize dose forms
        if 'dose_form' in df.columns:
            df['dose_form'] = self.standardize_dose_form(df['dose_form'])
            
        # Convert dosages to standard units
        if all(col in df.columns for col in ['dose_amt', 'dose_unit']):
            df['dose_std'] = self.standardize_dose(df['dose_amt'], df['dose_unit'])
            
        return df
        
    def process_reactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process reaction data.
        
        Args:
            df: Raw reaction DataFrame
            
        Returns:
            Processed reaction DataFrame
        """
        df = df.copy()
        
        # Standardize reaction terms using MedDRA
        if 'pt' in df.columns:
            df['reaction_term'] = self.standardize_reaction(df['pt'])
            
        # Add reaction severity if available
        if 'outc_cod' in df.columns:
            df['severity'] = self.standardize_outcome(df['outc_cod'])
            
        return df

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
                df[col] = self.check_date(df[col], max_date)

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
            self.country_map = pd.read_csv(country_file, sep=';', dtype=str).set_index('country')['Country_Name'].to_dict()

        # Load occupation codes (matching R script)
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

        # Load SOC data
        soc_file = meddra_dir / 'soc.asc'
        if soc_file.exists():
            self.soc_data = pd.read_csv(soc_file, sep='$', dtype=str, usecols=[0,1,2])
            self.soc_data.columns = ['soc_code', 'soc_name', 'soc_abbrev']

        # Load PT data
        pt_file = meddra_dir / 'pt.asc'
        if pt_file.exists():
            self.pt_data = pd.read_csv(pt_file, sep='$', dtype=str, usecols=[0,1,3])
            self.pt_data.columns = ['pt_code', 'pt_name', 'pt_soc_code']

        # Load LLT data
        llt_file = meddra_dir / 'llt.asc'
        if llt_file.exists():
            self.llt_data = pd.read_csv(llt_file, sep='$', dtype=str, usecols=[0,1,2])
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
            manual_fixes = pd.read_csv(manual_fix_file)
            self.manual_pt_fixes = dict(zip(manual_fixes['original'].str.lower(), 
                                          manual_fixes['standardized']))

    def _load_diana_dictionary(self):
        """Load and prepare the DiAna drug dictionary."""
        try:
            dict_path = self.external_dir / 'DiAna_dictionary' / 'drugnames_standardized.csv'
            if not dict_path.exists():
                logging.error(f"DiAna dictionary not found at {dict_path}")
                return
            
            # Read with error_bad_lines=False to skip problematic rows
            self.diana_dict = pd.read_csv(dict_path, 
                                        dtype={'drugname': str, 'Substance': str},
                                        on_bad_lines='skip',  # Skip problematic lines
                                        delimiter=';')  # Use semicolon delimiter
            
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
        """Standardize PT terms exactly as in R script."""
        df = df.copy()
        
        # Load MedDRA PT list
        meddra_df = pd.read_csv(self.external_dir / 'Dictionaries/MedDRA/meddra.csv', sep=';')
        pt_list = pd.Series(meddra_df['pt'].unique()).str.lower().str.strip().unique()
        
        # Calculate PT frequencies
        df[pt_variable] = df[pt_variable].str.lower().str.strip()
        pt_freq = (df[~df[pt_variable].isna()]
                   .groupby(pt_variable).size()
                   .reset_index(name='N')
                   .sort_values('N', ascending=False))
        
        # Check if PTs are standardized
        pt_freq['standard_pt'] = np.where(pt_freq[pt_variable].isin(pt_list), 
                                         pt_freq[pt_variable], 
                                         np.nan)
        pt_freq['freq'] = np.round(pt_freq['N'] / pt_freq['N'].sum() * 100, 2)
        
        # Get unstandardized PTs
        not_pts = pt_freq[pt_freq['standard_pt'].isna()][[pt_variable, 'N', 'freq']]
        
        # Calculate initial non-standardized percentage
        initial_nonstd_pct = np.round(not_pts['N'].sum() * 100 / len(df[~df[pt_variable].isna()]), 3)
        logging.info(f"Initial non-standardized PT percentage: {initial_nonstd_pct}%")
        
        # Try to translate through LLTs
        llt_mappings = meddra_df[['pt', 'llt']].copy()
        llt_mappings.columns = ['standard_pt', pt_variable]
        not_pts = not_pts.merge(llt_mappings, on=pt_variable, how='left')
        not_llts = not_pts[not_pts['standard_pt'].isna()].drop('standard_pt', axis=1)
        
        # Load and apply manual fixes
        manual_fixes = pd.read_csv(self.external_dir / 'Manual_fix/pt_fixed.csv', sep=';')
        manual_fixes = manual_fixes[[pt_variable, 'standard_pt']]
        
        not_llts = not_llts.merge(manual_fixes, on=pt_variable, how='left')
        still_unstandardized = not_llts[not_llts['standard_pt'].isna()][[pt_variable, 'standard_pt']]
        
        # Combine all standardization sources
        pt_fixed = pd.concat([
            manual_fixes,
            still_unstandardized,
            not_pts[~not_pts['standard_pt'].isna()][[pt_variable, 'standard_pt']]
        ]).drop_duplicates()
        
        # Check for duplicates
        duplicates = pt_fixed[pt_fixed[pt_variable].duplicated()]
        if not duplicates.empty:
            logging.warning(f"Duplicate PT mappings found: {duplicates[pt_variable].tolist()}")
        
        # Apply standardization to original data
        df['pt_temp'] = df[pt_variable].str.lower().str.strip()
        df = df.merge(pt_fixed, left_on='pt_temp', right_on=pt_variable, how='left')
        df[pt_variable] = df['standard_pt'].fillna(df['pt_temp'])
        df = df.drop(['pt_temp', 'standard_pt'], axis=1)
        
        # Calculate final standardization percentage
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
        
        # Ensure drugname column exists
        if 'drugname' not in df.columns and 'drug_name' in df.columns:
            df = df.rename(columns={'drug_name': 'drugname'})
        
        if 'drugname' not in df.columns:
            logging.error("No drugname or drug_name column found in DataFrame")
            return df
        
        # Clean drug names
        df['drugname'] = df['drugname'].fillna('').astype(str).apply(self._clean_drugname)
        
        # Map to standardized substances using the precomputed mapping
        df['Substance'] = df['drugname'].str.lower().str.strip().map(self.drug_map).fillna('UNKNOWN')
        
        # Handle multi-substance drugs
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
        
        # Mark trial drugs
        df['trial'] = df['Substance'].str.contains(', trial', na=False)
        df['Substance'] = df['Substance'].str.replace(', trial', '', regex=False)
        
        # Convert to categorical for memory efficiency
        df['drugname'] = df['drugname'].astype('category')
        df['Substance'] = df['Substance'].astype('category')
        if 'prod_ai' in df.columns:
            df['prod_ai'] = df['prod_ai'].astype('category')
        
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
        
        # Define age unit conversion factors (matching R script)
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
        df['age_in_days'] = pd.to_numeric(df['age'].abs()) * df['age_corrector']
        
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
        
        # Apply age group rules (matching R script)
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

    def process_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process demographics data.
        
        Args:
            df: Raw demographics DataFrame
            
        Returns:
            Processed demographics DataFrame
        """
        df = df.copy()
        
        # Standardize dates
        date_cols = ['init_fda_dt', 'fda_dt', 'event_dt', 'mfr_dt']
        for col in date_cols:
            if col in df.columns:
                df[col] = self.standardize_dates(df[col])
        
        # Standardize sex
        if 'sex' in df.columns:
            df['sex'] = self.standardize_sex(df['sex'])
            
        # Standardize age
        if 'age' in df.columns and 'age_cod' in df.columns:
            df['age'] = self.standardize_age(df['age'], df['age_cod'])
            
        # Standardize weight
        if 'wt' in df.columns and 'wt_cod' in df.columns:
            df['weight'] = self.standardize_weight(df['wt'], df['wt_cod'])
            
        # Standardize country codes
        if 'reporter_country' in df.columns:
            df['reporter_country'] = self.standardize_country(df['reporter_country'])
            
        # Standardize occupations
        if 'occp_cod' in df.columns:
            df['occupation'] = self.standardize_occupation(df['occp_cod'])
            
        return df
        
    def process_drugs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process drug data.
        
        Args:
            df: Raw drug DataFrame
            
        Returns:
            Processed drug DataFrame
        """
        df = df.copy()
        
        # Standardize drug names and substances
        df = self.standardize_drugs(df)
        
        # Standardize routes
        if 'route' in df.columns:
            df['route'] = self.standardize_route(df['route'])
            
        # Standardize dose forms
        if 'dose_form' in df.columns:
            df['dose_form'] = self.standardize_dose_form(df['dose_form'])
            
        # Convert dosages to standard units
        if all(col in df.columns for col in ['dose_amt', 'dose_unit']):
            df['dose_std'] = self.standardize_dose(df['dose_amt'], df['dose_unit'])
            
        return df
        
    def process_reactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process reaction data.
        
        Args:
            df: Raw reaction DataFrame
            
        Returns:
            Processed reaction DataFrame
        """
        df = df.copy()
        
        # Standardize reaction terms using MedDRA
        if 'pt' in df.columns:
            df['reaction_term'] = self.standardize_reaction(df['pt'])
            
        # Add reaction severity if available
        if 'outc_cod' in df.columns:
            df['severity'] = self.standardize_outcome(df['outc_cod'])
            
        return df

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
                df[col] = self.check_date(df[col], max_date)

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
            self.country_map = pd.read_csv(country_file, sep=';', dtype=str).set_index('country')['Country_Name'].to_dict()

        # Load occupation codes (matching R script)
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

        # Load SOC data
        soc_file = meddra_dir / 'soc.asc'
        if soc_file.exists():
            self.soc_data = pd.read_csv(soc_file, sep='$', dtype=str, usecols=[0,1,2])
            self.soc_data.columns = ['soc_code', 'soc_name', 'soc_abbrev']

        # Load PT data
        pt_file = meddra_dir / 'pt.asc'
        if pt_file.exists():
            self.pt_data = pd.read_csv(pt_file, sep='$', dtype=str, usecols=[0,1,3])
            self.pt_data.columns = ['pt_code', 'pt_name', 'pt_soc_code']

        # Load LLT data
        llt_file = meddra_dir / 'llt.asc'
        if llt_file.exists():
            self.llt_data = pd.read_csv(llt_file, sep='$', dtype=str, usecols=[0,1,2])
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
            manual_fixes = pd.read_csv(manual_fix_file)
            self.manual_pt_fixes = dict(zip(manual_fixes['original'].str.lower(), 
                                          manual_fixes['standardized']))

    def _load_diana_dictionary(self):
        """Load and prepare the DiAna drug dictionary."""
        try:
            dict_path = self.external_dir / 'DiAna_dictionary' / 'drugnames_standardized.csv'
            if not dict_path.exists():
                logging.error(f"DiAna dictionary not found at {dict_path}")
                return
            
            # Read with error_bad_lines=False to skip problematic rows
            self.diana_dict = pd.read_csv(dict_path, 
                                        dtype={'drugname': str, 'Substance': str},
                                        on_bad_lines='skip',  # Skip problematic lines
                                        delimiter=';')  # Use semicolon delimiter
            
            # Clean dictionary entries (matching R script)
            self.diana_dict['drugname'] = self.diana_dict['drugname'].apply(self._clean_drugname)
            self.diana_dict['Substance'] = self.diana_dict['Substance'].fillna('UNKNOWN')
            
            # Create mapping dictionary
            self.drug_map = dict(zip(
