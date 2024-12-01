"""Data standardization utilities for FAERS processing."""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple, Union

class DataStandardizer:
    """Handles standardization of FAERS data fields."""
    
    def __init__(self, external_dir: Path):
        """Initialize standardizer with external data directory."""
        self.external_dir = external_dir
        self._load_reference_data()
    
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
    
    def standardize_sex(self, df: pd.DataFrame, col: str = 'sex') -> pd.DataFrame:
        """Standardize sex values to F/M."""
        df = df.copy()
        df.loc[~df[col].isin(['F', 'M']), col] = np.nan
        return df
    
    def standardize_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize age values and add age groups."""
        df = df.copy()
        
        # Age unit conversion factors to days
        age_factors = {
            'DEC': 3650,
            'YR': 365,
            'MON': 30.41667,
            'WK': 7,
            'DY': 1,
            'HR': 0.00011415525114155251,
            'SEC': 3.1709791983764586e-08,
            'MIN': 1.9025875190259e-06
        }
        
        # Convert age to days
        df['age_corrector'] = df['age_cod'].map(age_factors)
        df['age_corrector'] = df['age_corrector'].fillna(365)  # Default to years if missing
        df['age_in_days'] = np.abs(pd.to_numeric(df['age'], errors="coerce")) * df['age_corrector']
        
        # Handle plausibility
        max_age_days = 122 * 365  # Max recorded human age
        df.loc[df['age_in_days'] > max_age_days, 'age_in_days'] = np.nan
        df.loc[(df['age_in_days'] > max_age_days) & (df['age_cod'] == 'DEC'), 'age_in_days'] = \
            df.loc[(df['age_in_days'] > max_age_days) & (df['age_cod'] == 'DEC'), 'age_in_days'] / \
            df.loc[(df['age_in_days'] > max_age_days) & (df['age_cod'] == 'DEC'), 'age_corrector']
        
        # Calculate age in years
        df['age_in_years'] = np.round(df['age_in_days'] / 365)
        
        # Assign age groups
        df['age_grp'] = 'E'  # Default to Elderly
        df.loc[df['age_in_years'] < 65, 'age_grp'] = 'A'  # Adult
        df.loc[df['age_in_years'] < 18, 'age_grp'] = 'T'  # Teen
        df.loc[df['age_in_years'] < 12, 'age_grp'] = 'C'  # Child
        df.loc[df['age_in_years'] < 2, 'age_grp'] = 'I'   # Infant
        df.loc[df['age_in_days'] < 28, 'age_grp'] = 'N'   # Neonate
        
        # Clean up temporary columns
        df = df.drop(columns=['age_corrector', 'age', 'age_cod'])
        
        return df
    
    def standardize_weight(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize weight values to kilograms."""
        df = df.copy()
        
        # Weight conversion factors to kg
        weight_factors = {
            'LBS': 0.453592,
            'IB': 0.453592,
            'KG': 1,
            'KGS': 1,
            'GMS': 0.001,
            'MG': 1e-06
        }
        
        # Convert weight to kg
        df['wt_corrector'] = df['wt_cod'].map(weight_factors)
        df['wt_corrector'] = df['wt_corrector'].fillna(1)  # Default to kg if missing
        df['wt_in_kgs'] = np.round(np.abs(pd.to_numeric(df['wt'], errors="coerce")) * df['wt_corrector'])
        
        # Handle implausible values (>635 kg)
        df.loc[df['wt_in_kgs'] > 635, 'wt_in_kgs'] = np.nan
        
        # Clean up temporary columns
        df = df.drop(columns=['wt_corrector', 'wt', 'wt_cod'])
        
        return df
    
    def standardize_country(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize country names using reference data."""
        df = df.copy()
        
        for col in ['occr_country', 'reporter_country']:
            if col in df.columns:
                df[col] = df[col].map(self.country_map)
        
        return df
    
    def standardize_occupation(self, df: pd.DataFrame, col: str = 'occp_cod') -> pd.DataFrame:
        """Standardize occupation codes."""
        df = df.copy()
        valid_codes = ["MD", "CN", "OT", "PH", "HP", "LW", "RN"]
        df.loc[~df[col].isin(valid_codes), col] = np.nan
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
