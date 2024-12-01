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
            self.country_map = pd.read_csv(country_file, sep=';').set_index('country')['Country_Name'].to_dict()
        
        # Load occupation codes
        self.valid_occupations = {'MD', 'CN', 'OT', 'PH', 'HP', 'LW', 'RN'}
        
        # Load route standardization
        route_file = self.external_dir / 'manual_fix' / 'route_st.csv'
        if route_file.exists():
            self.route_map = pd.read_csv(route_file, sep=';').set_index('route')['route_st'].to_dict()
        
        # Load dose form standardization
        dose_form_file = self.external_dir / 'manual_fix' / 'dose_form_st.csv'
        if dose_form_file.exists():
            self.dose_form_map = pd.read_csv(dose_form_file, sep=';').set_index('dose_form')['dose_form_st'].to_dict()
        
        # Load dose frequency standardization
        dose_freq_file = self.external_dir / 'manual_fix' / 'dose_freq_st.csv'
        if dose_freq_file.exists():
            self.dose_freq_map = pd.read_csv(dose_freq_file, sep=';').set_index('dose_freq')['dose_freq_st'].to_dict()
    
    def standardize_sex(self, df: pd.DataFrame, col: str = 'sex') -> pd.DataFrame:
        """Standardize sex values to F/M."""
        df = df.copy()
        df[col] = df[col].where(df[col].isin(['F', 'M']), None)
        return df
    
    def standardize_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize age values and add age groups."""
        df = df.copy()
        
        # Age conversion factors
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
        df['age_corrector'] = df['age_cod'].map(lambda x: age_factors.get(x, 365))
        df['age_in_days'] = df.apply(
            lambda x: round(abs(float(x['age'])) * x['age_corrector']) 
            if pd.notna(x['age']) and pd.notna(x['age_corrector']) 
            else None, 
            axis=1
        )
        
        # Handle plausible compilation error
        df['age_in_days'] = df.apply(
            lambda x: x['age_in_days'] if x['age_in_days'] <= 122*365 
            else (x['age_in_days']/x['age_corrector'] if x['age_cod'] == 'DEC' else None)
            if pd.notna(x['age_in_days']) else None,
            axis=1
        )
        
        # Calculate age in years
        df['age_in_years'] = df['age_in_days'].apply(lambda x: round(x/365) if pd.notna(x) else None)
        
        # Assign age groups
        df['age_grp'] = None
        df.loc[df['age_in_years'].notna(), 'age_grp'] = 'E'
        df.loc[df['age_in_years'] < 65, 'age_grp'] = 'A'
        df.loc[df['age_in_years'] < 18, 'age_grp'] = 'T'
        df.loc[df['age_in_years'] < 12, 'age_grp'] = 'C'
        df.loc[df['age_in_years'] < 2, 'age_grp'] = 'I'
        df.loc[df['age_in_days'] < 28, 'age_grp'] = 'N'
        
        # Clean up
        df = df.drop(['age_corrector', 'age', 'age_cod'], axis=1)
        return df
    
    def standardize_weight(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize weight values to kilograms."""
        df = df.copy()
        
        # Weight conversion factors
        weight_factors = {
            'LBS': 0.453592,
            'IB': 0.453592,
            'KG': 1,
            'KGS': 1,
            'GMS': 0.001,
            'MG': 1e-06
        }
        
        # Convert weight to kg
        df['wt_corrector'] = df['wt_cod'].map(lambda x: weight_factors.get(x, 1))
        df['wt_in_kgs'] = df.apply(
            lambda x: round(abs(float(x['wt'])) * x['wt_corrector'])
            if pd.notna(x['wt']) and pd.notna(x['wt_corrector'])
            else None,
            axis=1
        )
        
        # Remove implausible values
        df.loc[df['wt_in_kgs'] > 635, 'wt_in_kgs'] = None
        
        # Clean up
        df = df.drop(['wt_corrector', 'wt', 'wt_cod'], axis=1)
        return df
    
    def standardize_country(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize country names using reference data."""
        df = df.copy()
        
        for col in ['occr_country', 'reporter_country']:
            if col in df.columns:
                df[col] = df[col].map(lambda x: self.country_map.get(x, x))
        
        return df
    
    def standardize_occupation(self, df: pd.DataFrame, col: str = 'occp_cod') -> pd.DataFrame:
        """Standardize occupation codes."""
        df = df.copy()
        df[col] = df[col].where(df[col].isin(self.valid_occupations), None)
        return df
    
    def standardize_route(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize administration routes."""
        df = df.copy()
        df['route'] = df['route'].str.lower().str.strip()
        df['route'] = df['route'].map(lambda x: self.route_map.get(x, None) if pd.notna(x) else None)
        return df
    
    def standardize_dose_form(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize dose forms."""
        df = df.copy()
        df['dose_form'] = df['dose_form'].str.lower().str.strip()
        df['dose_form'] = df['dose_form'].map(lambda x: self.dose_form_map.get(x, None) if pd.notna(x) else None)
        return df
    
    def standardize_dose_freq(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize dose frequencies."""
        df = df.copy()
        df['dose_freq'] = df['dose_freq'].map(lambda x: self.dose_freq_map.get(x, None) if pd.notna(x) else None)
        return df
    
    def standardize_dates(self, df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
        """Standardize date fields."""
        df = df.copy()
        
        def check_date(dt):
            if pd.isna(dt):
                return None
                
            n = len(str(dt))
            year = int(str(dt)[:4])
            
            if n == 4:
                if 1985 <= year <= datetime.now().year:
                    return dt
            elif n == 6:
                if 198500 <= dt <= int(f"{datetime.now().year}12"):
                    return dt
            elif n == 8:
                if 19850000 <= dt <= int(f"{datetime.now().year}1231"):
                    return dt
            return None
        
        for col in date_cols:
            if col in df.columns:
                df[col] = df[col].apply(check_date)
        
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
