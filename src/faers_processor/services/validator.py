"""FAERS data validation service."""
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

@dataclass
class ValidationResult:
    """Results of data validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class DataValidator:
    """Validates FAERS data according to FDA specifications."""

    def __init__(self):
        """Initialize the validator."""
        self.logger = logging.getLogger(__name__)
        
        # Valid values
        self.valid_sex = {'M', 'F', 'UNK'}
        self.valid_age_codes = {'YR', 'MON', 'WK', 'DY', 'HR'}
        self.valid_routes = {'ORAL', 'INTRAVENOUS', 'SUBCUTANEOUS', 'INTRAMUSCULAR', 'TOPICAL'}
        self.valid_roles = {'PS', 'SS', 'C', 'I'}
        self.valid_outcomes = {'DE', 'LT', 'HO', 'DS', 'CA', 'RI', 'OT'}
        self.valid_occupations = {'MD', 'CN', 'OT', 'PH', 'HP', 'LW', 'RN'}
        
    def validate_demographics(self, df: pd.DataFrame) -> ValidationResult:
        """Validate demographics data.
        
        Args:
            df: Demographics DataFrame
            
        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult(valid=True)
        
        # Required columns
        required_cols = {'primaryid', 'i_f_code', 'event_dt', 'sex'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            result.valid = False
            result.errors.append(f"Missing required columns: {missing_cols}")
            
        if not result.valid:
            return result
            
        # Validate sex values
        invalid_sex = set(df['sex'].dropna().unique()) - self.valid_sex
        if invalid_sex:
            result.warnings.append(f"Invalid sex values found: {invalid_sex}")
            
        # Validate age codes if present
        if 'age_cod' in df.columns:
            invalid_age_codes = set(df['age_cod'].dropna().unique()) - self.valid_age_codes
            if invalid_age_codes:
                result.warnings.append(f"Invalid age codes found: {invalid_age_codes}")
                
        # Validate dates
        for date_col in ['event_dt', 'fda_dt', 'rept_dt']:
            if date_col in df.columns:
                try:
                    pd.to_datetime(df[date_col], format='%Y%m%d', errors='raise')
                except ValueError as e:
                    result.warnings.append(f"Invalid dates in {date_col}: {str(e)}")
                    
        return result
        
    def validate_drug_info(self, df: pd.DataFrame) -> ValidationResult:
        """Validate drug information data.
        
        Args:
            df: Drug information DataFrame
            
        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult(valid=True)
        
        # Required columns
        required_cols = {'primaryid', 'drug_seq', 'drugname'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            result.valid = False
            result.errors.append(f"Missing required columns: {missing_cols}")
            
        if not result.valid:
            return result
            
        # Validate role codes if present
        if 'role_cod' in df.columns:
            invalid_roles = set(df['role_cod'].dropna().unique()) - self.valid_roles
            if invalid_roles:
                result.warnings.append(f"Invalid role codes found: {invalid_roles}")
                
        # Validate routes if present
        if 'route' in df.columns:
            invalid_routes = set(df['route'].dropna().str.upper().unique()) - self.valid_routes
            if invalid_routes:
                result.warnings.append(f"Invalid routes found: {invalid_routes}")
                
        return result
        
    def validate_reactions(self, df: pd.DataFrame) -> ValidationResult:
        """Validate reaction data.
        
        Args:
            df: Reaction DataFrame
            
        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult(valid=True)
        
        # Required columns
        required_cols = {'primaryid', 'pt'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            result.valid = False
            result.errors.append(f"Missing required columns: {missing_cols}")
            
        return result
        
    def validate_outcomes(self, df: pd.DataFrame) -> ValidationResult:
        """Validate outcome data.
        
        Args:
            df: Outcome DataFrame
            
        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult(valid=True)
        
        # Required columns
        required_cols = {'primaryid', 'outc_cod'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            result.valid = False
            result.errors.append(f"Missing required columns: {missing_cols}")
            
        if not result.valid:
            return result
            
        # Validate outcome codes
        invalid_outcomes = set(df['outc_cod'].dropna().str.upper().unique()) - self.valid_outcomes
        if invalid_outcomes:
            result.warnings.append(f"Invalid outcome codes found: {invalid_outcomes}")
            
        return result
        
    def validate_indications(self, df: pd.DataFrame) -> ValidationResult:
        """Validate indication data.
        
        Args:
            df: Indication DataFrame
            
        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult(valid=True)
        
        # Required columns
        required_cols = {'primaryid', 'drug_seq', 'indi_pt'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            result.valid = False
            result.errors.append(f"Missing required columns: {missing_cols}")
            
        return result
        
    def validate_therapies(self, df: pd.DataFrame) -> ValidationResult:
        """Validate therapy data.
        
        Args:
            df: Therapy DataFrame
            
        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult(valid=True)
        
        # Required columns
        required_cols = {'primaryid', 'drug_seq', 'start_dt'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            result.valid = False
            result.errors.append(f"Missing required columns: {missing_cols}")
            
        if not result.valid:
            return result
            
        # Validate dates
        for date_col in ['start_dt', 'end_dt']:
            if date_col in df.columns:
                try:
                    pd.to_datetime(df[date_col], format='%Y%m%d', errors='raise')
                except ValueError as e:
                    result.warnings.append(f"Invalid dates in {date_col}: {str(e)}")
                    
        return result
        
    def validate_data(self, df: pd.DataFrame, data_type: str) -> ValidationResult:
        """Validate data based on its type.
        
        Args:
            df: DataFrame to validate
            data_type: Type of data ('demo', 'drug', 'reac', etc.)
            
        Returns:
            ValidationResult with validation details
        """
        validation_methods = {
            'demo': self.validate_demographics,
            'drug': self.validate_drug_info,
            'reac': self.validate_reactions,
            'outc': self.validate_outcomes,
            'indi': self.validate_indications,
            'ther': self.validate_therapies
        }
        
        if data_type not in validation_methods:
            return ValidationResult(
                valid=False,
                errors=[f"Unknown data type: {data_type}"]
            )
            
        return validation_methods[data_type](df)
