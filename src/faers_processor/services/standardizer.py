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
    
    def __init__(self, external_dir: Path, output_dir: Path):
        """Initialize the standardizer.
        
        Args:
            external_dir: Path to external data directory
            output_dir: Path to output directory
        """
        self.external_dir = external_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        logging.info(f"Using external data from: {self.external_dir}")
        logging.info(f"Saving processed data to: {self.output_dir}")
        
        # Load reference data
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

    def standardize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize date columns in demographics data.
        
        Args:
            df: DataFrame containing date columns
            
        Returns:
            DataFrame with standardized dates
        """
        if df.empty:
            return df
            
        df = df.copy()
        
        # Process date columns in order matching R script
        date_columns = ['event_dt', 'mfr_dt', 'fda_dt', 'rept_dt']
        
        # FAERS date formats we need to handle
        date_formats = [
            '%Y%m%d',      # YYYYMMDD
            '%Y%m',        # YYYYMM
            '%Y',          # YYYY
            '%d%b%Y',      # DDMONYYYY
            '%b%Y',        # MONYYYY
            '%d/%m/%Y',    # DD/MM/YYYY
            '%m/%d/%Y',    # MM/DD/YYYY
            '%Y-%m-%d',    # YYYY-MM-DD
            '%Y%m%d.0',    # YYYYMMDD.0 (from float conversion)
        ]
        
        def try_parse_date(date_str):
            """Try to parse a date string using multiple formats."""
            if pd.isna(date_str):
                return pd.NaT
                
            # Convert to string and clean
            date_str = str(date_str).strip()
            
            # Handle empty strings
            if not date_str or date_str == 'nan':
                return pd.NaT
                
            # Remove .0 if present (from float conversion)
            if date_str.endswith('.0'):
                date_str = date_str[:-2]
                
            # Try each format
            for fmt in date_formats:
                try:
                    return pd.to_datetime(date_str, format=fmt)
                except (ValueError, TypeError):
                    continue
                    
            # If all formats fail, try dateutil parser
            try:
                return pd.to_datetime(date_str)
            except (ValueError, TypeError):
                return pd.NaT
        
        # Valid date range
        min_date = pd.Timestamp('1960-01-01')
        max_date = pd.Timestamp.now()
        
        for col in date_columns:
            if col not in df.columns:
                continue
                
            try:
                # Convert column to string first to handle numeric dates
                df[col] = df[col].astype(str)
                
                # Convert to datetime using our custom parser
                df[col] = df[col].apply(try_parse_date)
                
                # Validate dates are within reasonable range
                mask = df[col].notna()
                invalid_dates = (
                    (df[col] < min_date) | 
                    (df[col] > max_date)
                )
                df.loc[mask & invalid_dates, col] = pd.NaT
                
                # Log statistics
                total_rows = len(df)
                na_count = df[col].isna().sum()
                if na_count > 0:
                    logging.warning(
                        f"{na_count}/{total_rows} rows ({na_count/total_rows*100:.1f}%) "
                        f"had invalid dates in {col}"
                    )
                    
            except Exception as e:
                logging.error(f"Error processing {col}: {str(e)}")
                # Keep original values if conversion fails
                continue
        
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
        try:
            # Manual fix files are under external_data/manual_fixes
            manual_fix_dir = self.external_dir / 'manual_fixes'
            
            if not manual_fix_dir.exists():
                raise FileNotFoundError(f"Manual fix directory not found at {manual_fix_dir}")
            
            # Load country mappings
            countries_file = manual_fix_dir / 'countries.csv'
            if not countries_file.exists():
                raise FileNotFoundError(f"Countries file not found at {countries_file}")
            self.country_map = pd.read_csv(countries_file, sep=';', low_memory=False)
            logging.info(f"Loaded {len(self.country_map)} country mappings")
            
            # Load route standardization
            route_file = manual_fix_dir / 'route_st.csv'
            if not route_file.exists():
                raise FileNotFoundError(f"Route file not found at {route_file}")
            self.route_map = pd.read_csv(route_file, sep=';', low_memory=False)
            logging.info(f"Loaded {len(self.route_map)} route mappings")
            
            # Load dose form standardization
            dose_form_file = manual_fix_dir / 'dose_form_st.csv'
            if not dose_form_file.exists():
                raise FileNotFoundError(f"Dose form file not found at {dose_form_file}")
            self.dose_form_map = pd.read_csv(dose_form_file, sep=';', low_memory=False)
            logging.info(f"Loaded {len(self.dose_form_map)} dose form mappings")
            
            # Load dose frequency standardization
            dose_freq_file = manual_fix_dir / 'dose_freq_st.csv'
            if not dose_freq_file.exists():
                raise FileNotFoundError(f"Dose frequency file not found at {dose_freq_file}")
            self.dose_freq_map = pd.read_csv(dose_freq_file, sep=';', low_memory=False)
            logging.info(f"Loaded {len(self.dose_freq_map)} dose frequency mappings")
            
            # Load PT fixes
            pt_file = manual_fix_dir / 'pt_fixed.csv'
            if not pt_file.exists():
                raise FileNotFoundError(f"PT file not found at {pt_file}")
            self.pt_fixes = pd.read_csv(pt_file, sep=';', low_memory=False)
            logging.info(f"Loaded {len(self.pt_fixes)} PT fixes")
            
            # Load route form standardization
            route_form_file = manual_fix_dir / 'route_form_st.csv'
            if not route_form_file.exists():
                raise FileNotFoundError(f"Route form file not found at {route_form_file}")
            self.route_form_map = pd.read_csv(route_form_file, sep=';', low_memory=False)
            logging.info(f"Loaded {len(self.route_form_map)} route form mappings")
            
        except Exception as e:
            logging.error(f"Error loading reference data: {str(e)}")
            raise

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
        
        # 1. Load MedDRA PT list and convert to lowercase for matching
        meddra_df = pd.read_csv(self.external_dir / 'Dictionaries/MedDRA/meddra.csv', 
                               sep=';', low_memory=False)
        pt_list = pd.Series(meddra_df['pt'].unique()).str.strip().str.lower().unique()
        
        # 2. Clean and standardize PT terms (matching R script)
        if pt_variable in df.columns:
            # Clean terms: lowercase and strip whitespace
            df[pt_variable] = df[pt_variable].str.strip().str.lower()
            
            # Calculate frequencies before standardization
            pt_freq = (df[~df[pt_variable].isna()]
                      .groupby(pt_variable).size()
                      .reset_index(name='N')
                      .sort_values('N', ascending=False))
            
            # Mark standard PTs
            pt_freq['standard_pt'] = np.where(pt_freq[pt_variable].isin(pt_list), 
                                            pt_freq[pt_variable], 
                                            np.nan)
            pt_freq['freq'] = np.round(pt_freq['N'] / pt_freq['N'].sum() * 100, 2)
            
            # Get unstandardized PTs
            not_pts = pt_freq[pt_freq['standard_pt'].isna()][[pt_variable, 'N', 'freq']]
            
            # Calculate initial non-standardized percentage
            initial_nonstd_pct = np.round(not_pts['N'].sum() * 100 / len(df[~df[pt_variable].isna()]), 3)
            logging.info(f"Initial non-standardized PT percentage: {initial_nonstd_pct}%")
            
            # Try LLT translations
            llt_mappings = meddra_df[['pt', 'llt']].copy()
            llt_mappings.columns = ['standard_pt', pt_variable]
            llt_mappings[pt_variable] = llt_mappings[pt_variable].str.strip().str.lower()
            llt_mappings['standard_pt'] = llt_mappings['standard_pt'].str.strip()
            
            not_pts = not_pts.merge(llt_mappings, on=pt_variable, how='left')
            not_llts = not_pts[not_pts['standard_pt'].isna()].drop('standard_pt', axis=1)
            
            # Try manual fixes
            manual_fixes = pd.read_csv(self.external_dir / 'Manual_fix/pt_fixed.csv', 
                                     sep=';', low_memory=False)
            manual_fixes = manual_fixes[[pt_variable, 'standard_pt']]
            manual_fixes[pt_variable] = manual_fixes[pt_variable].str.strip().str.lower()
            manual_fixes['standard_pt'] = manual_fixes['standard_pt'].str.strip()
            
            # Apply standardization in order: direct PT, LLT, manual fixes
            df['standard_pt'] = df[pt_variable].map(lambda x: x if x in pt_list else None)
            mask = df['standard_pt'].isna()
            df.loc[mask, 'standard_pt'] = df.loc[mask, pt_variable].map(
                dict(zip(llt_mappings[pt_variable], llt_mappings['standard_pt']))
            )
            still_missing = df['standard_pt'].isna()
            df.loc[still_missing, 'standard_pt'] = df.loc[still_missing, pt_variable].map(
                dict(zip(manual_fixes[pt_variable], manual_fixes['standard_pt']))
            )
            
            # Log standardization results
            total = len(df)
            standardized = df['standard_pt'].notna().sum()
            logging.info(f"PT standardization results:")
            logging.info(f"  Total terms: {total}")
            logging.info(f"  Standardized: {standardized} ({100*standardized/total:.1f}%)")
            logging.info(f"  Unstandardized: {total-standardized} ({100*(total-standardized)/total:.1f}%)")
            
            # Replace original column with standardized version
            df[pt_variable] = df['standard_pt']
            df = df.drop('standard_pt', axis=1)
        
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
        
        # Handle column name variations (both upper and lower case)
        sex_cols = ['sex', 'SEX', 'gndr_cod']
        for col in sex_cols:
            if col in df.columns:
                df = df.rename(columns={col: 'sex'})
                break
        
        if 'sex' not in df.columns:
            logging.warning("Sex column not found in DataFrame")
            return df
        
        # Convert to string and uppercase for standardization
        df['sex'] = df['sex'].astype(str).str.upper()
        
        # Map values
        sex_map = {
            'M': 'M',     # Male
            'F': 'F',     # Female
            'U': pd.NA,   # Unknown
            'UNK': pd.NA, # Unknown
            'NS': pd.NA,  # Not Specified
            'NAN': pd.NA, # Missing
            'NA': pd.NA,  # Missing
            'NR': pd.NA,  # Not Reported
            'UNKNOWN': pd.NA # Unknown
        }
        
        # Apply mapping and set non-mapped values to NA
        df['sex'] = df['sex'].map(sex_map)
        df.loc[~df['sex'].isin(['F', 'M']), 'sex'] = pd.NA
        
        # Log distribution
        sex_dist = df['sex'].value_counts(dropna=False)
        total = len(df)
        logging.info("Sex distribution:")
        for sex, count in sex_dist.items():
            percent = round(100 * count / total, 2)
            logging.info(f"  {sex if pd.notna(sex) else 'Unknown'}: {count} ({percent}%)")
        
        # Convert to category
        df['sex'] = df['sex'].astype('category')
        
        return df

    def standardize_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize age values to days and years.
        
        Args:
            df: DataFrame with age columns
            
        Returns:
            DataFrame with standardized age values
        """
        df = df.copy()
        
        # Ensure required columns exist
        if 'age' not in df.columns:
            df['age'] = pd.NA
            logging.warning("Age column not found, added with NA values")
        
        if 'age_cod' not in df.columns:
            df['age_cod'] = 'YR'  # Default to years if missing
            logging.info("Age code column not found, defaulting to years (YR)")
        
        # Convert age to numeric, handling various formats
        df['age'] = pd.to_numeric(
            df['age'].astype(str).str.replace(',', ''),
            errors='coerce'
        )
        
        # Define age unit conversion factors to days
        age_factors = {
            'DEC': 3650,       # Decades to days
            'YR': 365,         # Years to days
            'MON': 30.41667,   # Months to days (average)
            'WK': 7,           # Weeks to days
            'DY': 1,           # Days (no conversion needed)
            'HR': 1/24,        # Hours to days
            'MIN': 1/1440,     # Minutes to days
            'SEC': 1/86400     # Seconds to days
        }
        
        # Fill missing age codes with YR
        df['age_cod'] = df['age_cod'].fillna('YR')
        
        # Convert age to days
        df['age_corrector'] = df['age_cod'].str.upper().map(age_factors)
        df.loc[df['age_corrector'].isna(), 'age_corrector'] = age_factors['YR']
        df['age_in_days'] = df['age'].abs() * df['age_corrector']
        
        # Handle plausible age range
        max_age_days = 122 * 365  # Maximum recorded human age
        df.loc[df['age_in_days'] > max_age_days, 'age_in_days'] = pd.NA
        
        # Convert to years for convenience
        df['age_in_years'] = (df['age_in_days'] / 365).round()
        
        # Log age distribution
        age_stats = df['age_in_years'].describe()
        logging.info("Age distribution (years):")
        for stat, value in age_stats.items():
            logging.info(f"  {stat}: {value:.1f}")
        
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
        
        # Load route mappings (case-sensitive as in R script)
        route_map = pd.read_csv(self.external_dir / 'Manual_fix/routes.csv', 
                              sep=';', low_memory=False)
        
        # Create route mapping dictionary preserving original case
        route_dict = dict(zip(route_map['route'], route_map['Route_std']))
        
        # First try exact matches (case-sensitive)
        if 'route' in df.columns:
            df['route_std'] = df['route'].map(route_dict)
            
            # For non-matches, try case-insensitive matching
            mask = df['route_std'].isna()
            if mask.any():
                # Create case-insensitive mapping
                lower_dict = {k.lower(): v for k, v in route_dict.items()}
                df.loc[mask, 'route_std'] = df.loc[mask, 'route'].str.lower().map(lower_dict)
            
            # Replace original with standardized and convert to category
            df['route'] = df['route_std']
            df = df.drop('route_std', axis=1)
            df['route'] = df['route'].astype('category')
            
            # Log standardization results
            total = len(df)
            standardized = df['route'].notna().sum()
            logging.info(f"Route standardization results:")
            logging.info(f"  Total routes: {total}")
            logging.info(f"  Standardized: {standardized} ({100*standardized/total:.1f}%)")
            logging.info(f"  Unstandardized: {total-standardized} ({100*(total-standardized)/total:.1f}%)")
        
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
        
        # 1. Convert to string and lowercase (matching R's tolower)
        name = str(name).lower()
        
        # 2. Remove trailing dots (matching R's gsub("\\.$",""))
        name = re.sub(r'\.$', '', name)
        
        # 3. Trim whitespace (matching R's trimws)
        name = name.strip()
        
        # 4. Replace multiple spaces with single space (matching R's gsub("\\s+", " "))
        name = re.sub(r'\s+', ' ', name)
        
        # 5. Remove remaining special characters but keep hyphens and spaces
        name = re.sub(r'[^\w\s-]', '', name)
        
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
        if df.empty:
            return df
            
        df = df.copy()
        
        # Basic column mapping (handle both upper and lower case)
        column_map = {
            'isr': 'primaryid',
            'ISR': 'primaryid',
            'case': 'caseid',
            'CASE': 'caseid',
            'CASEID': 'caseid',
            'CASE_ID': 'caseid',
            'i_f_code': 'i_f_code',
            'i_f_cod': 'i_f_code',
            'event_dt': 'event_dt',
            'EVENT_DT': 'event_dt',
            'mfr_dt': 'mfr_dt',
            'MFR_DT': 'mfr_dt',
            'fda_dt': 'fda_dt',
            'FDA_DT': 'fda_dt',
            'rept_dt': 'rept_dt',
            'REPT_DT': 'rept_dt',
            'sex': 'sex',
            'SEX': 'sex',
            'gndr_cod': 'sex',
            'age': 'age',
            'age_cod': 'age_cod',
            'AGE_COD': 'age_cod',
            'age_grp': 'age_grp',
            'AGE_GRP': 'age_grp',
            'wt': 'wt',
            'wt_cod': 'wt_cod',
            'WT_COD': 'wt_cod',
            'rept_cod': 'rept_cod',
            'REPT_COD': 'rept_cod',
            'occp_cod': 'occp_cod',
            'OCCP_COD': 'occp_cod',
            'reporter_country': 'reporter_country',
            'REPORTER_COUNTRY': 'reporter_country',
            'occr_country': 'occr_country',
            'OCCR_COUNTRY': 'occr_country'
        }
        
        # Rename columns based on mapping
        df = df.rename(columns=column_map)
        
        # Ensure required columns exist
        required_cols = ['primaryid', 'caseid', 'i_f_code', 'event_dt', 'mfr_dt', 'fda_dt', 'rept_dt']
        for col in required_cols:
            if col not in df.columns:
                logging.warning(f"Required column '{col}' not found, adding with NA values")
                df[col] = pd.NA
        
        # Convert IDs to numeric and handle missing values
        for id_col in ['primaryid', 'caseid']:
            if id_col in df.columns:
                # Convert to numeric, coercing errors to NaN
                df[id_col] = pd.to_numeric(df[id_col], errors='coerce')
                # Fill NaN with -1 and convert to int64
                df[id_col] = df[id_col].fillna(-1).astype('int64')
        
        # Convert numeric fields
        numeric_cols = ['age', 'wt']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Process dates
        date_columns = ['event_dt', 'mfr_dt', 'fda_dt', 'rept_dt']
        if any(col in df.columns for col in date_columns):
            df = self.standardize_dates(df)
        
        # Process other fields in order
        df = self.standardize_sex(df)
        df = self.standardize_age(df)
        df = self.standardize_age_groups(df)
        df = self.standardize_weight(df)
        df = self.standardize_country(df)
        df = self.standardize_occupation(df)
        df = self.standardize_manufacturer(df)
        
        # Optimize memory usage
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
        
        # Basic column mapping
        column_map = {
            'isr': 'primaryid',
            'case': 'caseid',
            'drug_seq': 'drug_seq',
            'DRUGNAME': 'drugname',  # Handle both upper and lower case
            'drugname': 'drugname',
            'prod_ai': 'prod_ai'
        }
        df = df.rename(columns=column_map)
        
        # Convert numeric fields
        numeric_cols = ['primaryid', 'caseid', 'drug_seq']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if col in ['primaryid', 'caseid', 'drug_seq']:
                    df[col] = df[col].fillna(-1).astype('int64')
        
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
        
        # Basic column mapping
        column_map = {
            'isr': 'primaryid',
            'case': 'caseid',
            'pt': 'pt'
        }
        df = df.rename(columns=column_map)
        
        # Convert numeric fields
        numeric_cols = ['primaryid', 'caseid']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if col in ['primaryid', 'caseid']:
                    df[col] = df[col].fillna(-1).astype('int64')
        
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
            # First try standard parsing
            try:
                df = pd.read_csv(
                    file_path,
                    sep='$',
                    dtype=str,
                    na_values=['', 'NA', 'NULL'],
                    keep_default_na=True,
                    low_memory=False,
                    encoding='utf-8',
                    on_bad_lines='warn' if data_type == 'drugs' else 'error'
                )
            except pd.errors.ParserError as e:
                if data_type == 'drugs':
                    logging.warning(f"Standard parsing failed for {file_path}, attempting manual fix: {str(e)}")
                    
                    # Read file manually and fix lines with incorrect field counts
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Get header and expected field count
                    header = lines[0].strip().split('$')
                    expected_fields = len(header)
                    
                    # Fix problematic lines
                    fixed_lines = [lines[0]]  # Keep header
                    for i, line in enumerate(lines[1:], 1):
                        fields = line.strip().split('$')
                        if len(fields) != expected_fields:
                            # Try to fix common issues
                            line = line.replace('$$', '$NA$')  # Fix empty fields
                            line = re.sub(r'\${2,}', '$', line)  # Remove multiple delimiters
                            fields = line.strip().split('$')
                            
                            # If still incorrect, log and skip
                            if len(fields) != expected_fields:
                                logging.warning(f"Skipping line {i} in {file_path}: incorrect field count")
                                continue
                        fixed_lines.append(line)
                    
                    # Create temporary file with fixed data
                    temp_file = file_path.parent / f"temp_{file_path.name}"
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        f.writelines(fixed_lines)
                    
                    try:
                        # Try reading fixed file
                        df = pd.read_csv(
                            temp_file,
                            sep='$',
                            dtype=str,
                            na_values=['', 'NA', 'NULL'],
                            keep_default_na=True,
                            low_memory=False,
                            encoding='utf-8'
                        )
                        temp_file.unlink()  # Clean up temp file
                    except Exception as e2:
                        temp_file.unlink()  # Clean up temp file
                        raise Exception(f"Failed to process {file_path} even after fixes: {str(e2)}")
                else:
                    raise e
            
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
        if df.empty:
            return df
            
        df = df.copy()
        
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
            'rept_dt': 'report_date',
            'fda_dt': 'fda_date',
            'mfr_dt': 'manufacturer_date',
            'init_fda_dt': 'initial_fda_date'
        }
        df = df.rename(columns=column_map)
        
        # Note: Date standardization is already done in process_demographics
        # Just rename the columns here
        
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

    def standardize_route(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize route values using route_st.csv mapping.
        
        Args:
            df: DataFrame with route column
            
        Returns:
            DataFrame with standardized route values
        """
        if 'route' not in df.columns:
            logging.warning("Route column not found, skipping route standardization")
            return df
            
        df = df.copy()
        
        # Convert to uppercase for consistent matching
        df['route'] = df['route'].str.upper()
        
        # Create route mapping dictionary
        route_dict = dict(zip(
            self.route_map['route'].str.upper(),
            self.route_map['route_st']
        ))
        
        # Apply standardization
        df['route_st'] = df['route'].map(route_dict)
        
        # Log standardization results
        total_routes = len(df['route'].dropna())
        mapped_routes = len(df['route_st'].dropna())
        logging.info(f"Route standardization: {mapped_routes}/{total_routes} routes mapped ({mapped_routes/total_routes*100:.1f}%)")
        
        # Log unmapped values
        unmapped = df[df['route_st'].isna() & df['route'].notna()]['route'].unique()
        if len(unmapped) > 0:
            logging.warning(f"Unmapped routes: {', '.join(unmapped)}")
        
        return df

    def standardize_dose_form(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize dose form values using dose_form_st.csv mapping.
        
        Args:
            df: DataFrame with dose_form column
            
        Returns:
            DataFrame with standardized dose form values
        """
        if 'dose_form' not in df.columns:
            logging.warning("Dose form column not found, skipping dose form standardization")
            return df
            
        df = df.copy()
        
        # Convert to uppercase for consistent matching
        df['dose_form'] = df['dose_form'].str.upper()
        
        # Create dose form mapping dictionary
        dose_form_dict = dict(zip(
            self.dose_form_map['dose_form'].str.upper(),
            self.dose_form_map['dose_form_st']
        ))
        
        # Apply standardization
        df['dose_form_st'] = df['dose_form'].map(dose_form_dict)
        
        # Log standardization results
        total_forms = len(df['dose_form'].dropna())
        mapped_forms = len(df['dose_form_st'].dropna())
        logging.info(f"Dose form standardization: {mapped_forms}/{total_forms} forms mapped ({mapped_forms/total_forms*100:.1f}%)")
        
        # Log unmapped values
        unmapped = df[df['dose_form_st'].isna() & df['dose_form'].notna()]['dose_form'].unique()
        if len(unmapped) > 0:
            logging.warning(f"Unmapped dose forms: {', '.join(unmapped)}")
        
        return df

    def standardize_dose_freq(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize dose frequency values using dose_freq_st.csv mapping.
        
        Args:
            df: DataFrame with dose_freq column
            
        Returns:
            DataFrame with standardized dose frequency values
        """
        if 'dose_freq' not in df.columns:
            logging.warning("Dose frequency column not found, skipping dose frequency standardization")
            return df
            
        df = df.copy()
        
        # Convert to uppercase for consistent matching
        df['dose_freq'] = df['dose_freq'].str.upper()
        
        # Create dose frequency mapping dictionary
        dose_freq_dict = dict(zip(
            self.dose_freq_map['dose_freq'].str.upper(),
            self.dose_freq_map['dose_freq_st']
        ))
        
        # Apply standardization
        df['dose_freq_st'] = df['dose_freq'].map(dose_freq_dict)
        
        # Log standardization results
        total_freqs = len(df['dose_freq'].dropna())
        mapped_freqs = len(df['dose_freq_st'].dropna())
        logging.info(f"Dose frequency standardization: {mapped_freqs}/{total_freqs} frequencies mapped ({mapped_freqs/total_freqs*100:.1f}%)")
        
        # Log unmapped values
        unmapped = df[df['dose_freq_st'].isna() & df['dose_freq'].notna()]['dose_freq'].unique()
        if len(unmapped) > 0:
            logging.warning(f"Unmapped dose frequencies: {', '.join(unmapped)}")
        
        return df

    def standardize_pt(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize preferred terms using pt_fixed.csv mapping.
        
        Args:
            df: DataFrame with pt column
            
        Returns:
            DataFrame with standardized preferred terms
        """
        if 'pt' not in df.columns:
            logging.warning("PT column not found, skipping PT standardization")
            return df
            
        df = df.copy()
        
        # Create PT mapping dictionary
        pt_dict = dict(zip(
            self.pt_fixes['pt_original'],
            self.pt_fixes['pt_fixed']
        ))
        
        # Apply standardization
        df['pt_st'] = df['pt'].map(pt_dict).fillna(df['pt'])
        
        # Log standardization results
        total_pts = len(df['pt'].dropna())
        fixed_pts = len(df[df['pt'] != df['pt_st']].dropna())
        logging.info(f"PT standardization: {fixed_pts} terms fixed out of {total_pts} total terms")
        
        return df

    def standardize_country(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize country values using countries.csv mapping.
        
        Args:
            df: DataFrame with country column
            
        Returns:
            DataFrame with standardized country values
        """
        if 'country' not in df.columns:
            logging.warning("Country column not found, skipping country standardization")
            return df
            
        df = df.copy()
        
        # Convert to uppercase for consistent matching
        df['country'] = df['country'].str.upper()
        
        # Create country mapping dictionary
        country_dict = dict(zip(
            self.country_map['country'].str.upper(),
            self.country_map['Country_Name']
        ))
        
        # Apply standardization
        df['country_st'] = df['country'].map(country_dict)
        
        # Log standardization results
        total_countries = len(df['country'].dropna())
        mapped_countries = len(df['country_st'].dropna())
        logging.info(f"Country standardization: {mapped_countries}/{total_countries} countries mapped ({mapped_countries/total_countries*100:.1f}%)")
        
        # Log unmapped values
        unmapped = df[df['country_st'].isna() & df['country'].notna()]['country'].unique()
        if len(unmapped) > 0:
            logging.warning(f"Unmapped countries: {', '.join(unmapped)}")
        
        return df
