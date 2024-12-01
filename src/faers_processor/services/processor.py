"""Service for processing FAERS data files."""
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from ..models.faers_data import Demographics, Drug, Reaction, Outcome

class DataProcessor(ABC):
    """Abstract base class for data processing."""
    
    @abstractmethod
    def process_file(self, file_path: Path) -> pd.DataFrame:
        """Process a single data file."""
        pass
    
    @abstractmethod
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and formats."""
        pass

class FAERSProcessor(DataProcessor):
    """FAERS specific data processor implementation."""
    
    COLUMN_MAPPINGS = {
        'ISR': 'primary_id',
        'CASE': 'case_id',
        'DRUG_SEQ': 'drug_seq',
        'ROLE_COD': 'role_code',
        'DRUGNAME': 'drug_name',
        'PT': 'pt',
        'OUTC_COD': 'outcome_code'
    }
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        
    def process_file(self, file_path: Path) -> pd.DataFrame:
        """Process a single FAERS data file."""
        # Read the header to get column names
        with open(file_path, 'r') as f:
            header = f.readline().strip().split('$')
            
        # Read the data with the correct delimiter
        df = pd.read_csv(file_path, sep='$', names=header, skiprows=1,
                        dtype=str, na_values=[''], keep_default_na=False)
        return self.standardize_columns(df)
        
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and formats."""
        # Rename columns according to mapping
        df = df.rename(columns=self.COLUMN_MAPPINGS)
        
        # Convert empty strings to None
        df = df.replace('', None)
        
        return df
        
    def standardize_pt(self, df: pd.DataFrame, pt_column: str = 'pt') -> pd.DataFrame:
        """Standardize Preferred Terms (PT)."""
        if pt_column not in df.columns:
            return df
            
        # Convert to lowercase and strip whitespace
        df[pt_column] = df[pt_column].str.lower().str.strip()
        
        # Remove special characters and standardize spacing
        df[pt_column] = df[pt_column].str.replace(r'[^\w\s]', ' ', regex=True)
        df[pt_column] = df[pt_column].str.replace(r'\s+', ' ', regex=True)
        
        return df
        
    def standardize_occupation(self, df: pd.DataFrame, occp_column: str = 'occp_cod') -> pd.DataFrame:
        """Standardize occupation codes."""
        if occp_column not in df.columns:
            return df
            
        # Define occupation mappings based on R script
        occupation_map = {
            'MD': 'MD',
            'PH': 'PH',
            'OT': 'OT',
            'CN': 'CN',
            'LW': 'OT',  # Lawyer mapped to Other
            'CO': 'CN'   # Consumer mapped to Consumer
        }
        
        # Apply mapping and set unknown values to None
        df[occp_column] = df[occp_column].map(occupation_map)
        return df
        
    def standardize_country(self, df: pd.DataFrame, country_column: str = 'reporter_country') -> pd.DataFrame:
        """Standardize country codes."""
        if country_column not in df.columns:
            return df
            
        # Remove special characters and standardize spacing
        df[country_column] = df[country_column].str.upper().str.strip()
        
        # Map common variations (add more as needed)
        country_map = {
            'USA': 'US',
            'UNITED STATES': 'US',
            'UNITED STATES OF AMERICA': 'US',
            'UK': 'GB',
            'UNITED KINGDOM': 'GB',
            'GREAT BRITAIN': 'GB'
        }
        
        df[country_column] = df[country_column].map(country_map).fillna(df[country_column])
        return df

    def process_demographics(self, file_path: Path) -> List[Demographics]:
        """Process demographics data file with standardization."""
        df = self.process_file(file_path)
        
        # Apply standardizations
        df = self.standardize_occupation(df)
        df = self.standardize_country(df)
        
        # Convert age to days and weight to kg
        df['age_in_days'] = pd.to_numeric(df['age'], errors='coerce')
        df.loc[df['age_cod'] == 'YR', 'age_in_days'] *= 365.25
        df.loc[df['age_cod'] == 'MON', 'age_in_days'] *= 30.44
        df.loc[df['age_cod'] == 'WK', 'age_in_days'] *= 7
        
        df['wt_in_kgs'] = pd.to_numeric(df['wt'], errors='coerce')
        df.loc[df['wt_cod'] == 'LBS', 'wt_in_kgs'] *= 0.45359237
        
        return [
            Demographics(
                primary_id=row['primary_id'],
                case_id=row['case_id'],
                case_version=row.get('case_version'),
                sex=row.get('sex'),
                age=row.get('age_in_days'),
                weight=row.get('wt_in_kgs'),
                reporter_country=row.get('reporter_country'),
                occupation=row.get('occp_cod'),
                quarter=row.get('quarter')
            )
            for _, row in df.iterrows()
        ]
        
    def process_drugs(self, file_path: Path) -> List[Drug]:
        """Process drug data file."""
        df = self.process_file(file_path)
        return [
            Drug(
                primary_id=row['primary_id'],
                drug_seq=row['drug_seq'],
                role_code=row['role_code'],
                drug_name=row['drug_name'],
                substance=row.get('substance'),
                prod_ai=row.get('prod_ai')
            )
            for _, row in df.iterrows()
        ]
        
    def process_reactions(self, file_path: Path) -> List[Reaction]:
        """Process reaction data file."""
        df = self.process_file(file_path)
        df = self.standardize_pt(df)
        return [
            Reaction(
                primary_id=row['primary_id'],
                pt=row['pt'],
                drug_rec_act=row.get('drug_rec_act')
            )
            for _, row in df.iterrows()
        ]
        
    def process_outcomes(self, file_path: Path) -> List[Outcome]:
        """Process outcome data file."""
        df = self.process_file(file_path)
        return [
            Outcome(
                primary_id=row['primary_id'],
                outcome_code=row['outcome_code']
            )
            for _, row in df.iterrows()
        ]

    def validate_record(self, demo_df: pd.DataFrame, drug_df: pd.DataFrame, 
                       reac_df: pd.DataFrame) -> pd.DataFrame:
        """Validate records for completeness."""
        # Check for presence of both drugs and reactions
        valid_ids = set(demo_df['primary_id']) & set(drug_df['primary_id']) & set(reac_df['primary_id'])
        return demo_df[demo_df['primary_id'].isin(valid_ids)]

    def calculate_similarity_score(self, row1: pd.Series, row2: pd.Series) -> float:
        """Calculate similarity score between two records."""
        score = 0.0
        weights = {
            'sex': 0.1,
            'age_in_days': 0.2,
            'wt_in_kgs': 0.1,
            'reporter_country': 0.1,
            'pt': 0.3,
            'drug_name': 0.2
        }
        
        # Compare demographic fields
        if row1['sex'] == row2['sex'] and pd.notna(row1['sex']):
            score += weights['sex']
            
        # Compare age with tolerance
        if (pd.notna(row1['age_in_days']) and pd.notna(row2['age_in_days']) and 
            abs(row1['age_in_days'] - row2['age_in_days']) < 30):  # 30 days tolerance
            score += weights['age_in_days']
            
        # Compare weight with tolerance
        if (pd.notna(row1['wt_in_kgs']) and pd.notna(row2['wt_in_kgs']) and 
            abs(row1['wt_in_kgs'] - row2['wt_in_kgs']) < 2):  # 2 kg tolerance
            score += weights['wt_in_kgs']
            
        # Compare country
        if row1['reporter_country'] == row2['reporter_country'] and pd.notna(row1['reporter_country']):
            score += weights['reporter_country']
            
        # Compare reactions (PT)
        pt1 = set(str(row1['pt']).split(';'))
        pt2 = set(str(row2['pt']).split(';'))
        if pt1.intersection(pt2):
            score += weights['pt'] * len(pt1.intersection(pt2)) / max(len(pt1), len(pt2))
            
        # Compare drugs
        drug1 = set(str(row1['drug_name']).split(';'))
        drug2 = set(str(row2['drug_name']).split(';'))
        if drug1.intersection(drug2):
            score += weights['drug_name'] * len(drug1.intersection(drug2)) / max(len(drug1), len(drug2))
            
        return score

    def deduplicate_records(self, demo_df: pd.DataFrame, drug_df: pd.DataFrame, 
                          reac_df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
        """Deduplicate records using probabilistic record linkage."""
        # Merge data for comparison
        merged_df = demo_df.merge(
            drug_df.groupby('primary_id')['drug_name'].agg(';'.join).reset_index(),
            on='primary_id', how='left'
        ).merge(
            reac_df.groupby('primary_id')['pt'].agg(';'.join).reset_index(),
            on='primary_id', how='left'
        )
        
        # Find potential duplicates
        duplicates = []
        for i, row1 in merged_df.iterrows():
            for j, row2 in merged_df.iloc[i+1:].iterrows():
                score = self.calculate_similarity_score(row1, row2)
                if score >= threshold:
                    duplicates.append((row1['primary_id'], row2['primary_id']))
                    
        # Keep only one record from each duplicate group
        duplicate_ids = set([id2 for _, id2 in duplicates])
        return demo_df[~demo_df['primary_id'].isin(duplicate_ids)]

    def process_multi_substance_drugs(self, drug_df: pd.DataFrame) -> pd.DataFrame:
        """Process drugs with multiple substances."""
        # Split multi-substance drugs
        drug_df['substance_list'] = drug_df['substance'].str.split(';')
        
        # Explode the dataframe to create separate rows for each substance
        expanded_df = drug_df.explode('substance_list')
        
        # Clean up substance names
        expanded_df['substance_list'] = expanded_df['substance_list'].str.strip()
        
        # Remove empty substances
        expanded_df = expanded_df[expanded_df['substance_list'].notna() & 
                                (expanded_df['substance_list'] != '')]
        
        return expanded_df.rename(columns={'substance_list': 'substance'})
