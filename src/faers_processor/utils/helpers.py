"""Utility functions for FAERS data processing."""
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

def standardize_dates(date_str: Optional[str]) -> Optional[datetime]:
    """Standardize date formats from FAERS data."""
    if not date_str or pd.isna(date_str):
        return None
        
    try:
        # Try different date formats
        for fmt in ('%Y%m%d', '%Y-%m-%d', '%d/%m/%Y'):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None
    except Exception:
        return None

def standardize_numeric(value: Any, unit: Optional[str] = None) -> Optional[float]:
    """Standardize numeric values with optional unit conversion."""
    if pd.isna(value):
        return None
        
    try:
        value = float(value)
        
        # Apply unit conversions if needed
        if unit:
            unit = unit.lower()
            if unit in ['kg', 'kgs']:
                return value
            elif unit in ['g', 'grams']:
                return value / 1000
            elif unit in ['mg', 'milligrams']:
                return value / 1000000
        return value
    except (ValueError, TypeError):
        return None

def clean_text(text: Optional[str]) -> Optional[str]:
    """Clean and standardize text fields."""
    if not text or pd.isna(text):
        return None
        
    # Convert to string and clean
    text = str(text).strip()
    
    # Remove multiple spaces
    text = ' '.join(text.split())
    
    return text if text else None

def get_quarter_from_filename(file_path: Path) -> str:
    """Extract quarter information from FAERS filename."""
    filename = file_path.stem.upper()
    
    # Extract year and quarter (e.g., "18Q1" from "DEMO18Q1.txt")
    year_quarter = ''.join(filter(str.isalnum, filename))
    year_quarter = ''.join(c for c in year_quarter if c.isdigit() or c in 'Qq')
    
    if len(year_quarter) >= 3:
        year = year_quarter[:2]
        quarter = year_quarter[-1]
        
        # Convert 2-digit year to 4-digit year
        full_year = '19' + year if int(year) > 50 else '20' + year
        
        return f"{full_year}Q{quarter}"
    return ''

def deduplicate_records(df: pd.DataFrame, 
                       key_columns: List[str],
                       similarity_threshold: float = 0.9) -> pd.DataFrame:
    """Deduplicate records based on key columns and similarity."""
    if df.empty or not key_columns:
        return df
        
    # Create a similarity score for each group of records
    def calculate_similarity(group: pd.DataFrame) -> pd.Series:
        if len(group) == 1:
            return pd.Series([1.0], index=group.index)
            
        # Calculate similarity based on non-key columns
        non_key_cols = [col for col in group.columns if col not in key_columns]
        similarities = []
        
        for idx1, row1 in group.iterrows():
            max_sim = 0
            for idx2, row2 in group.iterrows():
                if idx1 == idx2:
                    continue
                    
                # Calculate similarity for each non-key column
                col_sims = []
                for col in non_key_cols:
                    val1, val2 = row1[col], row2[col]
                    if pd.isna(val1) and pd.isna(val2):
                        col_sims.append(1.0)
                    elif pd.isna(val1) or pd.isna(val2):
                        col_sims.append(0.0)
                    else:
                        col_sims.append(1.0 if val1 == val2 else 0.0)
                
                sim = np.mean(col_sims) if col_sims else 0.0
                max_sim = max(max_sim, sim)
            
            similarities.append(max_sim)
            
        return pd.Series(similarities, index=group.index)
    
    # Group by key columns and calculate similarities
    groups = df.groupby(key_columns, dropna=False)
    similarities = groups.apply(calculate_similarity)
    
    # Keep records above similarity threshold
    return df[similarities >= similarity_threshold]
