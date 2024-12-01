"""Service for processing FAERS data files."""
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from ..models.faers_data import Demographics, Drug, Reaction, Outcome, DrugInfo, Indication, ReportSource, Therapy
import re
import datetime
import logging

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
        self.external_dir = Path('external_data')
        self._load_external_data()
        
    def _load_external_data(self):
        """Load external reference data for standardization."""
        # Load country codes
        country_file = self.external_dir / 'country_codes.csv'
        if country_file.exists():
            df = pd.read_csv(country_file)
            self.country_map = {}
            for _, row in df.iterrows():
                self.country_map[row['code']] = row['code']
                for var in row['common_variations'].split(','):
                    self.country_map[var.strip()] = row['code']
        
        # Load occupation codes
        occupation_file = self.external_dir / 'occupation_codes.csv'
        if occupation_file.exists():
            df = pd.read_csv(occupation_file)
            self.occupation_map = {}
            for _, row in df.iterrows():
                self.occupation_map[row['code']] = row['code']
                for var in row['variations'].split(','):
                    self.occupation_map[var.strip()] = row['code']
        
    def process_file(self, file_path: Path) -> pd.DataFrame:
        """Process a single FAERS data file."""
        try:
            # Read the header to get column names
            with open(file_path, 'r', encoding='utf-8') as f:
                header = f.readline().strip().split('$')
            
            # Read the data with the correct delimiter
            df = pd.read_csv(file_path, sep='$', names=header, skiprows=1,
                           dtype=str, na_values=[''], keep_default_na=False,
                           on_bad_lines='warn', encoding='utf-8')
            
            if df.empty:
                raise ValueError(f"Empty dataframe from file: {file_path}")
                
            return self.standardize_columns(df)
        except (IOError, pd.errors.EmptyDataError) as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            return pd.DataFrame()
        
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
        """Standardize occupation codes using external reference data."""
        if occp_column not in df.columns or not hasattr(self, 'occupation_map'):
            return df
            
        # Apply mapping from external file
        df[occp_column] = df[occp_column].map(self.occupation_map)
        return df
        
    def standardize_country(self, df: pd.DataFrame, country_column: str = 'reporter_country') -> pd.DataFrame:
        """Standardize country codes using external reference data."""
        if country_column not in df.columns or not hasattr(self, 'country_map'):
            return df
            
        # Remove special characters and standardize spacing
        df[country_column] = df[country_column].str.upper().str.strip()
        
        # Apply mapping from external file
        df[country_column] = df[country_column].map(self.country_map).fillna(df[country_column])
        return df

    def standardize_weight(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize weight to kilograms."""
        if 'wt' not in df.columns or 'wt_cod' not in df.columns:
            return df
            
        # Define weight conversion factors
        weight_map = {
            'LBS': 0.453592,  # Pounds to kg
            'IB': 0.453592,   # Pounds to kg
            'KG': 1.0,        # Already in kg
            'KGS': 1.0,       # Already in kg
            'GMS': 0.001,     # Grams to kg
            'MG': 1e-6        # Milligrams to kg
        }
        
        # Convert weight to numeric, handling invalid values
        df['wt'] = pd.to_numeric(df['wt'], errors='coerce')
        
        # Apply conversion factors
        df['wt_corrector'] = df['wt_cod'].map(weight_map).fillna(1.0)
        df['wt_in_kgs'] = abs(df['wt'] * df['wt_corrector']).round()
        
        # Remove implausible weights (>635 kg)
        df.loc[df['wt_in_kgs'] > 635, 'wt_in_kgs'] = None
        
        # Clean up temporary columns
        df = df.drop(['wt_corrector', 'wt', 'wt_cod'], axis=1)
        return df
        
    def standardize_sex(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize sex values."""
        if 'sex' not in df.columns:
            return df
            
        # Only keep valid sex codes
        df.loc[~df['sex'].isin(['F', 'M']), 'sex'] = None
        return df
        
    def standardize_dates(self, df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
        """Standardize date fields."""
        max_date = self._get_max_date()
        
        def check_date(dt):
            if pd.isna(dt):
                return dt
                
            dt_str = str(dt)
            n = len(dt_str)
            
            # Validate based on length
            if n == 4:  # Year only
                year = int(dt_str)
                if year < 1985 or year > int(str(max_date)[:4]):
                    return None
            elif n == 6:  # Year and month
                year = int(dt_str[:4])
                month = int(dt_str[4:6])
                if year < 1985 or year > int(str(max_date)[:4]) or month < 1 or month > 12:
                    return None
            elif n == 8:  # Full date
                year = int(dt_str[:4])
                month = int(dt_str[4:6])
                day = int(dt_str[6:8])
                if year < 1985 or int(dt_str) > max_date:
                    return None
                try:
                    datetime.datetime(year, month, day)
                except ValueError:
                    return None
            else:
                return None
                
            return dt
            
        for col in date_cols:
            if col in df.columns:
                df[col] = df[col].apply(check_date)
                
        return df
        
    def _get_max_date(self) -> int:
        """Get maximum valid date based on latest quarter."""
        # Find the latest quarter file
        latest_file = None
        latest_quarter = None
        
        for file in self.data_dir.rglob("*.[Tt][Xx][Tt]"):
            if "DELETE" in file.name.upper():
                quarter_match = re.search(r'(\d{2})[Qq](\d)', file.name)
                if quarter_match:
                    year, quarter = quarter_match.groups()
                    if not latest_quarter or (year, quarter) > latest_quarter:
                        latest_quarter = (year, quarter)
                        latest_file = file
                        
        if not latest_quarter:
            return 20500101  # Default if no quarter found
            
        year = f"20{latest_quarter[0]}"
        quarter_month = {
            '1': '0331',  # Q1 ends March 31
            '2': '0630',  # Q2 ends June 30
            '3': '0930',  # Q3 ends September 30
            '4': '1231'   # Q4 ends December 31
        }
        
        return int(f"{year}{quarter_month[latest_quarter[1]]}")
        
    def process_demographics(self, file_path: Path) -> List[Demographics]:
        """Process demographics data file with standardization."""
        df = self.process_file(file_path)
        
        # Apply standardizations
        df = self.standardize_occupation(df)
        df = self.standardize_country(df)
        df = self.standardize_sex(df)
        df = self.standardize_weight(df)
        
        # Convert age to days 
        df['age_in_days'] = pd.to_numeric(df['age'], errors='coerce')
        df.loc[df['age_cod'] == 'YR', 'age_in_days'] *= 365.25
        df.loc[df['age_cod'] == 'MON', 'age_in_days'] *= 30.44
        df.loc[df['age_cod'] == 'WK', 'age_in_days'] *= 7
        
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
                          reac_df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate records using rule-based and probabilistic methods."""
        # First pass: Rule-based deduplication
        complete_duplicates = ['event_dt', 'sex', 'reporter_country', 'age_in_days', 
                             'wt_in_kgs', 'pt', 'substance']
                             
        # Merge drug and reaction data
        merged_df = demo_df.merge(drug_df[['primary_id', 'substance']], 
                                on='primary_id', how='left')
        merged_df = merged_df.merge(reac_df[['primary_id', 'pt']], 
                                  on='primary_id', how='left')
                                  
        # Group by all duplicate fields
        grouped = merged_df.groupby(complete_duplicates)
        
        # Keep only the latest record from each group
        unique_records = grouped.apply(lambda x: x.sort_values('fda_dt').iloc[-1])
        
        # Mark duplicates
        demo_df['RB_duplicates'] = ~demo_df['primary_id'].isin(unique_records['primary_id'])
        
        # Second pass: Consider only suspect drugs
        suspect_drugs = drug_df[drug_df['role_cod'].isin(['PS', 'SS'])]
        merged_df = demo_df.merge(suspect_drugs[['primary_id', 'substance']], 
                                on='primary_id', how='left')
        
        grouped = merged_df.groupby(complete_duplicates)
        unique_records = grouped.apply(lambda x: x.sort_values('fda_dt').iloc[-1])
        
        # Mark duplicates considering only suspect drugs
        demo_df['RB_duplicates_only_susp'] = ~demo_df['primary_id'].isin(unique_records['primary_id'])
        
        return demo_df
        
    def process_multi_substance_drugs(self, drug_df: pd.DataFrame) -> pd.DataFrame:
        """Process drugs with multiple substances."""
        if 'substance' not in drug_df.columns:
            return drug_df
            
        # Split multi-substance drugs
        multi_mask = drug_df['substance'].str.contains(';', na=False)
        
        # Process single substance drugs
        single_drugs = drug_df[~multi_mask].copy()
        
        # Process multi substance drugs
        multi_drugs = drug_df[multi_mask].copy()
        if not multi_drugs.empty:
            # Split substances and create new rows
            expanded = multi_drugs.assign(
                substance=multi_drugs['substance'].str.split(';')
            ).explode('substance')
            
            # Combine single and multi substance results
            drug_df = pd.concat([single_drugs, expanded], ignore_index=True)
            
        return drug_df
        
    def correct_problematic_file(self, file_path: Path, old_line: str) -> None:
        """Correct files with missing newlines."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Replace problematic line with corrected version
            new_line = old_line.replace('$', '$\n')
            content = ''.join(lines).replace(old_line, new_line)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except IOError as e:
            logging.error(f"Error correcting file {file_path}: {str(e)}")

    def process_drug_info(self, file_path: Path) -> List[DrugInfo]:
        """Process drug information data file."""
        df = self.process_file(file_path)
        
        # Convert date fields
        date_cols = ['exp_dt']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Standardize dechal and rechal values
        for col in ['dechal', 'rechal']:
            if col in df.columns:
                df[col] = df[col].map({'Y': 'Y', 'N': 'N', 'D': 'D'})
        
        return [
            DrugInfo(
                primary_id=row['primary_id'],
                drug_seq=row['drug_seq'],
                val_vbm=row.get('val_vbm'),
                nda_num=row.get('nda_num'),
                lot_num=row.get('lot_num'),
                route=row.get('route'),
                dose_form=row.get('dose_form'),
                dose_freq=row.get('dose_freq'),
                exp_dt=row.get('exp_dt'),
                dose_amt=float(row['dose_amt']) if pd.notna(row.get('dose_amt')) else None,
                dose_unit=row.get('dose_unit'),
                dechal=row.get('dechal'),
                rechal=row.get('rechal')
            )
            for _, row in df.iterrows()
        ]

    def process_indications(self, file_path: Path) -> List[Indication]:
        """Process drug indications data file."""
        df = self.process_file(file_path)
        df = self.standardize_pt(df, 'indi_pt')
        
        # Remove rows with missing PT
        df = df[df['indi_pt'].notna()]
        
        return [
            Indication(
                primary_id=row['primary_id'],
                drug_seq=row['drug_seq'],
                indi_pt=row['indi_pt']
            )
            for _, row in df.iterrows()
        ]

    def process_report_sources(self, file_path: Path) -> List[ReportSource]:
        """Process report sources data file."""
        df = self.process_file(file_path)
        
        return [
            ReportSource(
                primary_id=row['primary_id'],
                rpsr_cod=row['rpsr_cod']
            )
            for _, row in df.iterrows()
        ]

    def process_therapy(self, file_path: Path) -> List[Therapy]:
        """Process drug therapy data file."""
        df = self.process_file(file_path)
        
        # Convert date fields
        date_cols = ['start_dt', 'end_dt']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return [
            Therapy(
                primary_id=row['primary_id'],
                drug_seq=row['drug_seq'],
                start_dt=row.get('start_dt'),
                end_dt=row.get('end_dt'),
                dur=float(row['dur']) if pd.notna(row.get('dur')) else None,
                dur_cod=row.get('dur_cod')
            )
            for _, row in df.iterrows()
        ]
