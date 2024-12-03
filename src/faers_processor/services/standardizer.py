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
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from tabulate import tabulate

@dataclass
class TableSummary:
    """Summary statistics for a single FAERS table."""
    total_rows: int = 0
    invalid_dates: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    missing_columns: List[str] = field(default_factory=list)
    parsing_errors: List[str] = field(default_factory=list)
    processing_time: float = 0.0

@dataclass
class QuarterSummary:
    """Summary statistics for a FAERS quarter."""
    quarter: str
    demo_summary: TableSummary = field(default_factory=TableSummary)
    drug_summary: TableSummary = field(default_factory=TableSummary)
    reac_summary: TableSummary = field(default_factory=TableSummary)
    processing_time: float = 0.0

class FAERSProcessingSummary:
    """Tracks and generates summary reports for FAERS data processing."""
    
    def __init__(self):
        self.quarter_summaries: Dict[str, QuarterSummary] = {}
        
    def add_quarter_summary(self, quarter: str, summary: QuarterSummary):
        """Add summary for a processed quarter."""
        self.quarter_summaries[quarter] = summary
        
    def generate_markdown_report(self) -> str:
        """Generate a detailed markdown report of all processing results."""
        report = ["# FAERS Processing Summary Report\n"]
        
        # Individual quarter summaries
        report.append("## Quarter-by-Quarter Summary\n")
        for quarter, summary in sorted(self.quarter_summaries.items()):
            report.append(f"### Quarter {quarter}\n")
            
            # Demographics table
            demo_data = [["Total Rows", summary.demo_summary.total_rows]]
            for col, count in summary.demo_summary.invalid_dates.items():
                demo_data.append([f"Invalid {col}", count])
            
            report.append("#### Demographics\n")
            report.append(tabulate(demo_data, headers=["Metric", "Value"], 
                                 tablefmt="pipe"))
            report.append("\n")
            
            # Drug table
            drug_data = [
                ["Total Rows", summary.drug_summary.total_rows],
                ["Parsing Errors", len(summary.drug_summary.parsing_errors)]
            ]
            report.append("#### Drug Data\n")
            report.append(tabulate(drug_data, headers=["Metric", "Value"], 
                                 tablefmt="pipe"))
            report.append("\n")
            
            # Reaction table
            reac_data = [
                ["Total Rows", summary.reac_summary.total_rows],
                ["Missing Columns", len(summary.reac_summary.missing_columns)]
            ]
            report.append("#### Reaction Data\n")
            report.append(tabulate(reac_data, headers=["Metric", "Value"], 
                                 tablefmt="pipe"))
            report.append("\n")
            
        # Grand summary by table type
        report.append("## Grand Summary\n")
        
        def table_totals(table_attr: str) -> List[List[Union[str, int]]]:
            """Calculate totals for a specific table type."""
            totals = defaultdict(int)
            for summary in self.quarter_summaries.values():
                table_summary = getattr(summary, table_attr)
                totals["Total Rows"] += table_summary.total_rows
                for date_col, count in table_summary.invalid_dates.items():
                    totals[f"Invalid {date_col}"] += count
            return [[k, v] for k, v in totals.items()]
        
        # Demographics totals
        report.append("### Demographics Totals\n")
        demo_totals = table_totals("demo_summary")
        report.append(tabulate(demo_totals, headers=["Metric", "Value"], 
                             tablefmt="pipe"))
        report.append("\n")
        
        # Drug totals
        report.append("### Drug Data Totals\n")
        drug_totals = table_totals("drug_summary")
        report.append(tabulate(drug_totals, headers=["Metric", "Value"], 
                             tablefmt="pipe"))
        report.append("\n")
        
        # Reaction totals
        report.append("### Reaction Data Totals\n")
        reac_totals = table_totals("reac_summary")
        report.append(tabulate(reac_totals, headers=["Metric", "Value"], 
                             tablefmt="pipe"))
        report.append("\n")
        
        return "\n".join(report)

class DataStandardizer:
    """Standardizes FAERS data according to FDA specifications."""
    
    def __init__(self, external_dir: Path, output_dir: Path):
        """Initialize the standardizer.
        
        Args:
            external_dir: Path to external data directory
            output_dir: Path to output directory
        """
        self.external_dir = external_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.logger = logging.getLogger(__name__)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Valid values for standardization
        self.valid_routes = {'ORAL', 'INTRAVENOUS', 'SUBCUTANEOUS', 'INTRAMUSCULAR', 'TOPICAL'}
        self.valid_roles = {'PS', 'SS', 'C', 'I'}
        self.valid_outcomes = {'DE', 'LT', 'HO', 'DS', 'CA', 'RI', 'OT'}
        
        # Initialize logging
        self.logger.info(f"Using external data from: {self.external_dir}")
        self.logger.info(f"Saving processed data to: {self.output_dir}")
        
        # Load reference data
        self._load_reference_data()
        self._load_meddra_data()
        self._load_diana_dictionary()
        
    def _get_column_case_insensitive(self, df: pd.DataFrame, column_name: str) -> Optional[str]:
        """Get the actual column name by checking uppercase first, then lowercase."""
        # Try uppercase first
        upper_name = column_name.upper()
        if upper_name in df.columns:
            return upper_name
            
        # Try lowercase
        lower_name = column_name.lower()
        if lower_name in df.columns:
            return lower_name
            
        return None

    def _has_column_case_insensitive(self, df: pd.DataFrame, column_name: str) -> bool:
        """Check if DataFrame has a column, trying exact, upper, and lower case."""
        df_cols = set(df.columns)
        return (column_name in df_cols or 
                column_name.upper() in df_cols or 
                column_name.lower() in df_cols)
        
    def standardize_demographics(self, df: pd.DataFrame, quarter_name: str, file_name: str) -> pd.DataFrame:
        """Standardize demographics information."""
        try:
            # Required columns exactly as defined in documentation.html
            required_columns = {
                'isr': ['ISR'],
                'age': ['AGE'],
                'age_cod': ['AGE_COD'],
                'sex': ['SEX'],
                'event_dt': ['EVENT_DT'],
                'i_f_cod': ['I_F_COD'],
                'rept_cod': ['REPT_COD'],
                'lit_ref': ['LIT_REF'],
                'age_grp': ['AGE_GRP']
            }
            
            # Process each required column
            for target_col, source_cols in required_columns.items():
                found = False
                for col in source_cols:
                    if col in df.columns:
                        if col != target_col:
                            df = df.rename(columns={col: target_col})
                        found = True
                        break
                
                if not found:
                    self.logger.warning(f"({quarter_name}) {file_name}: Required column '{target_col}' not found, adding with default value: <NA>")
                    df[target_col] = pd.NA
                    if target_col == 'i_f_cod':
                        df[target_col] = 'I'
            
            return df
            
        except Exception as e:
            self.logger.error(f"({quarter_name}) {file_name}: Error in standardize_demographics: {str(e)}")
            return df

    def standardize_drug_info(self, df: pd.DataFrame, quarter_name: str, file_name: str) -> pd.DataFrame:
        """Standardize drug information."""
        try:
            # Required columns exactly as defined in documentation.html
            required_columns = {
                'isr': ['ISR'],
                'drug_seq': ['DRUG_SEQ'],
                'role_cod': ['ROLE_COD'],
                'drugname': ['DRUGNAME'],
                'prod_ai': ['PROD_AI'],
                'val_vbm': ['VAL_VBM'],
                'route': ['ROUTE'],
                'dose_vbm': ['DOSE_VBM'],
                'cum_dose_chr': ['CUM_DOSE_CHR'],
                'cum_dose_unit': ['CUM_DOSE_UNIT'],
                'dechal': ['DECHAL'],
                'rechal': ['RECHAL'],
                'lot_num': ['LOT_NUM'],
                'exp_dt': ['EXP_DT'],
                'nda_num': ['NDA_NUM']
            }
            
            # Process each required column
            for target_col, source_cols in required_columns.items():
                found = False
                for col in source_cols:
                    if col in df.columns:
                        if col != target_col:
                            df = df.rename(columns={col: target_col})
                        found = True
                        break
                
                if not found:
                    self.logger.warning(f"({quarter_name}) {file_name}: Required column '{target_col}' not found, adding with default value: <NA>")
                    df[target_col] = pd.NA
            
            return df
            
        except Exception as e:
            self.logger.error(f"({quarter_name}) {file_name}: Error in standardize_drug_info: {str(e)}")
            return df

    def standardize_reactions(self, df: pd.DataFrame, quarter_name: str, file_name: str) -> pd.DataFrame:
        """Standardize reactions information."""
        try:
            # Required columns exactly as defined in documentation.html
            required_columns = {
                'isr': ['ISR'],
                'pt': ['PT'],
                'drug_rec_act': ['DRUG_REC_ACT']
            }
            
            # Process each required column
            for target_col, source_cols in required_columns.items():
                found = False
                for col in source_cols:
                    if col in df.columns:
                        if col != target_col:
                            df = df.rename(columns={col: target_col})
                        found = True
                        break
                
                if not found:
                    self.logger.warning(f"({quarter_name}) {file_name}: Required column '{target_col}' not found, adding with default value: <NA>")
                    df[target_col] = pd.NA
            
            return df
            
        except Exception as e:
            self.logger.error(f"({quarter_name}) {file_name}: Error in standardize_reactions: {str(e)}")
            return df

    def standardize_indications(self, df: pd.DataFrame, quarter_name: str, file_name: str) -> pd.DataFrame:
        """Standardize indication data."""
        try:
            # Column name mappings
            column_mappings = {
                'indi_pt': ['INDI_PT', 'indi_pt'],
                'drug_seq': ['DRUG_SEQ', 'drug_seq', 'INDI_DRUG_SEQ', 'indi_drug_seq'],
                'isr': ['ISR', 'isr', 'PRIMARYID', 'primaryid', 'CASEID', 'caseid']
            }
            
            # Standardize column names
            for std_name, possible_names in column_mappings.items():
                col_name = self._get_column_case_insensitive(df, possible_names[0])
                if col_name:
                    if col_name != std_name:
                        df = df.rename(columns={col_name: std_name})
                else:
                    self.logger.warning(f"{std_name} column not found - initialized with empty values")
                    df[std_name] = pd.NA
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in standardize_indications: {str(e)}")
            return df

    def standardize_outcomes(self, df: pd.DataFrame, quarter_name: str, file_name: str) -> pd.DataFrame:
        """Standardize outcome data."""
        try:
            # Column name mappings
            column_mappings = {
                'outc_cod': ['OUTC_COD', 'outc_cod'],
                'isr': ['ISR', 'isr', 'PRIMARYID', 'primaryid', 'CASEID', 'caseid']
            }
            
            # Standardize column names
            for std_name, possible_names in column_mappings.items():
                col_name = self._get_column_case_insensitive(df, possible_names[0])
                if col_name:
                    if col_name != std_name:
                        df = df.rename(columns={col_name: std_name})
                else:
                    self.logger.warning(f"{std_name} column not found - initialized with empty values")
                    df[std_name] = pd.NA
            
            # Standardize outcome codes
            if 'outc_cod' in df.columns:
                df['outc_cod'] = df['outc_cod'].str.upper()
                invalid_codes = set(df['outc_cod'].dropna().unique()) - self.valid_outcomes
                if invalid_codes:
                    self.logger.warning(f"Converted invalid outcome codes to NA: {invalid_codes}")
                    df.loc[df['outc_cod'].isin(invalid_codes), 'outc_cod'] = pd.NA
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in standardize_outcomes: {str(e)}")
            return df

    def standardize_therapies(self, df: pd.DataFrame, quarter_name: str, file_name: str) -> pd.DataFrame:
        """Standardize therapy data."""
        try:
            # Column name mappings
            column_mappings = {
                'drug_seq': ['DRUG_SEQ', 'drug_seq', 'DSG_DRUG_SEQ', 'dsg_drug_seq'],
                'start_dt': ['START_DT', 'start_dt'],
                'end_dt': ['END_DT', 'end_dt'],
                'dur': ['DUR', 'dur'],
                'dur_cod': ['DUR_COD', 'dur_cod'],
                'isr': ['ISR', 'isr', 'PRIMARYID', 'primaryid', 'CASEID', 'caseid']
            }
            
            # Standardize column names
            for std_name, possible_names in column_mappings.items():
                col_name = self._get_column_case_insensitive(df, possible_names[0])
                if col_name:
                    if col_name != std_name:
                        df = df.rename(columns={col_name: std_name})
                else:
                    self.logger.warning(f"{std_name} column not found - initialized with empty values")
                    df[std_name] = pd.NA
            
            # Standardize dates
            date_columns = ['start_dt', 'end_dt']
            for col in date_columns:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col], format='%Y%m%d', errors='coerce')
                        invalid_dates = df[col].isna().sum()
                        if invalid_dates > 0:
                            self.logger.warning(f"Found {invalid_dates} invalid dates in {col}")
                    except Exception as e:
                        self.logger.error(f"Error converting {col} to datetime: {str(e)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in standardize_therapies: {str(e)}")
            return df

    def remove_incomplete_cases(self, demo_df: pd.DataFrame, drug_df: pd.DataFrame,
                                reac_df: pd.DataFrame) -> pd.DataFrame:
        """
        Log statistics about cases that don't have valid drugs or reactions, but do not remove them.
        
        Args:
            demo_df: Demographics DataFrame
            drug_df: Drug DataFrame
            reac_df: Reactions DataFrame
        
        Returns:
            Original demographics DataFrame, unmodified
        """
        # Find cases without valid drugs
        valid_drug_cases = set(drug_df['primaryid'].unique())
        all_cases = set(demo_df['primaryid'].unique())
        cases_without_drugs = all_cases - valid_drug_cases

        # Find cases without valid reactions
        valid_reaction_cases = set(reac_df['primaryid'].unique())
        cases_without_reactions = all_cases - valid_reaction_cases

        # Calculate and log results
        logging.info(f"Total cases: {len(all_cases)}")
        logging.info(f"Cases without drugs: {len(cases_without_drugs)}")
        logging.info(f"Cases without reactions: {len(cases_without_reactions)}")

        return demo_df

    def _load_reference_data(self):
        """Load reference data for standardization."""
        try:
            # Load route standardization data
            route_path = self.external_dir / 'manual_fixes' / 'route_st.csv'
            if not route_path.exists():
                raise FileNotFoundError(f"Route standardization file not found at {route_path}")
            
            self.route_map = pd.read_csv(route_path, sep=';', dtype=str)
            self.route_map = dict(zip(
                self.route_map['route'].str.lower().str.strip(),
                self.route_map['route_st'].str.strip()
            ))
            
            # Load dose form standardization data
            dose_form_path = self.external_dir / 'manual_fixes' / 'dose_form_st.csv'
            if not dose_form_path.exists():
                raise FileNotFoundError(f"Dose form standardization file not found at {dose_form_path}")
            
            self.dose_form_map = pd.read_csv(dose_form_path, sep=';', dtype=str)
            self.dose_form_map = dict(zip(
                self.dose_form_map['dose_form'].str.lower().str.strip(),
                self.dose_form_map['dose_form_st'].str.strip()
            ))
            
            # Load dose frequency standardization data
            dose_freq_path = self.external_dir / 'manual_fixes' / 'dose_freq_st.csv'
            if not dose_freq_path.exists():
                raise FileNotFoundError(f"Dose frequency standardization file not found at {dose_freq_path}")
            
            self.dose_freq_map = pd.read_csv(dose_freq_path, sep=';', dtype=str)
            self.dose_freq_map = dict(zip(
                self.dose_freq_map['dose_freq'].str.lower().str.strip(),
                self.dose_freq_map['dose_freq_st'].str.strip()
            ))
            
            # Load country standardization data
            country_path = self.external_dir / 'manual_fixes' / 'countries.csv'
            if not country_path.exists():
                raise FileNotFoundError(f"Country standardization file not found at {country_path}")
            
            self.country_map = pd.read_csv(country_path, sep=';', dtype=str)
            self.country_map = dict(zip(
                self.country_map['country'].str.lower().str.strip(),
                self.country_map['Country_Name'].str.strip()
            ))
            
            self.logger.info("Successfully loaded reference data")
            
        except Exception as e:
            self.logger.error(f"Error loading reference data: {str(e)}")
            # Initialize empty mappings as fallback
            self.route_map = {}
            self.dose_form_map = {}
            self.dose_freq_map = {}
            self.country_map = {}

    def _load_meddra_data(self):
        """Load and process MedDRA data from ASC files."""
        try:
            meddra_dir = self.external_dir / 'meddra'
            if not meddra_dir.exists():
                raise FileNotFoundError(f"MedDRA directory not found at {meddra_dir}")
            
            # Read MedDRA files
            llt_path = meddra_dir / 'llt.asc'
            pt_path = meddra_dir / 'pt.asc'
            
            if not llt_path.exists():
                raise FileNotFoundError(f"LLT file not found at {llt_path}")
            if not pt_path.exists():
                raise FileNotFoundError(f"PT file not found at {pt_path}")
            
            # Read LLT data
            meddra = pd.read_csv(
                llt_path,
                sep='$',
                names=['llt_code', 'llt_name', 'pt_code', 'llt_whoart_code',
                      'llt_harts_code', 'llt_costart_sym', 'llt_icd9_code',
                      'llt_icd9cm_code', 'llt_icd10_code', 'llt_currency',
                      'llt_jart_code'],
                dtype=str,
                quoting=3
            )
            
            # Read PT data
            pt = pd.read_csv(
                pt_path,
                sep='$',
                names=['pt_code', 'pt_name', 'null_field', 'pt_whoart_code',
                      'pt_harts_code', 'pt_costart_sym', 'pt_icd9_code',
                      'pt_icd9cm_code', 'pt_icd10_code', 'pt_jart_code'],
                dtype=str,
                quoting=3
            )
            
            # Merge LLT and PT data
            meddra = meddra.merge(
                pt[['pt_code', 'pt_name']],
                on='pt_code',
                how='left'
            )
            
            # Create standardized columns
            meddra['llt'] = meddra['llt_name'].str.upper()
            meddra['pt'] = meddra['pt_name'].str.upper()
            
            # Get distinct PTs
            meddra_distinct = meddra[['pt']].drop_duplicates()
            
            # Save processed data
            meddra.to_csv(self.external_dir / 'meddra' / 'meddra.csv', 
                         sep=';', index=False)
            meddra_distinct.to_csv(self.external_dir / 'meddra' / 'meddra_primary.csv', 
                                 sep=';', index=False)
            
            # Create PT list for standardization
            self.pt_list = meddra_distinct['pt'].dropna().unique()
            self.llt_to_pt = dict(zip(meddra['llt'].str.lower(), meddra['pt'].str.lower()))
            
            self.logger.info(f"Loaded {len(self.pt_list):,} unique PTs from MedDRA")
            self.logger.info(f"Loaded {len(self.llt_to_pt):,} LLT to PT mappings")
            
        except Exception as e:
            self.logger.error(f"Error loading MedDRA data: {str(e)}")
            self.pt_list = []
            self.llt_to_pt = {}

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
            
            self.logger.info(f"Loaded DiAna dictionary with {len(self.diana_dict)} entries")
            
        except Exception as e:
            self.logger.error(f"Error loading DiAna dictionary: {str(e)}")
            self.diana_dict = pd.DataFrame(columns=['drugname', 'Substance'])
            self.drug_map = {}

    def _clean_drugname(self, name: str) -> str:
        """Clean drug name by removing special characters and standardizing format."""
        if pd.isna(name):
            return name
            
        # Convert to string and lowercase
        name = str(name).lower()
        
        # Remove trailing dots
        name = re.sub(r'\.$', '', name)
        
        # Trim whitespace
        name = name.strip()
        
        # Replace multiple spaces with single space
        name = re.sub(r'\s+', ' ', name)
        
        # Remove special characters but keep hyphens and spaces
        name = re.sub(r'[^\w\s-]', '', name)
        
        return name

    def standardize_pt(self, df: pd.DataFrame, pt_variable: str = 'pt') -> pd.DataFrame:
        """Standardize PT terms following R script exactly.
        
        Args:
            df: DataFrame with PT terms
            pt_variable: Name of the PT column
            
        Returns:
            DataFrame with standardized PT terms
        """
        try:
            if pt_variable not in df.columns:
                return df
                
            df = df.copy()
            
            # Clean and standardize PT terms
            df[pt_variable] = df[pt_variable].str.strip().str.lower()
            
            # Calculate initial frequencies
            pt_freq = (df[~df[pt_variable].isna()]
                      .groupby(pt_variable).size()
                      .reset_index(name='N')
                      .sort_values('N', ascending=False))
            
            # Mark standard PTs
            pt_freq['standard_pt'] = pt_freq[pt_variable].apply(
                lambda x: x if x in self.pt_list else None
            )
            pt_freq['freq'] = round(pt_freq['N'] / pt_freq['N'].sum() * 100, 2)
            
            # Get unstandardized PTs
            not_pts = pt_freq[pt_freq['standard_pt'].isna()][[pt_variable, 'N', 'freq']]
            
            # Calculate initial non-standardized percentage
            initial_nonstd_pct = round(
                not_pts['N'].sum() * 100 / len(df[~df[pt_variable].isna()]), 
                3
            )
            logging.info(f"Initial non-standardized PT percentage: {initial_nonstd_pct}%")
            
            # Try LLT translations
            not_pts['standard_pt'] = not_pts[pt_variable].map(self.llt_to_pt)
            not_llts = not_pts[not_pts['standard_pt'].isna()].drop('standard_pt', axis=1)
            
            # Try manual fixes
            manual_fixes = pd.read_csv(
                self.external_dir / 'manual_fixes' / 'pt_fixed.csv',  # Updated path
                sep=';', 
                low_memory=False
            )
            manual_fixes = manual_fixes[[pt_variable, 'standard_pt']]
            manual_fixes[pt_variable] = manual_fixes[pt_variable].str.strip().str.lower()
            manual_fixes['standard_pt'] = manual_fixes['standard_pt'].str.strip()
            
            # Apply standardization in order: direct PT, LLT, manual fixes
            df['standard_pt'] = df[pt_variable].apply(
                lambda x: x if x in self.pt_list else None
            )
            
            # Apply LLT translations
            mask = df['standard_pt'].isna()
            df.loc[mask, 'standard_pt'] = df.loc[mask, pt_variable].map(self.llt_to_pt)
            
            # Apply manual fixes
            still_missing = df['standard_pt'].isna()
            df.loc[still_missing, 'standard_pt'] = df.loc[still_missing, pt_variable].map(
                dict(zip(manual_fixes[pt_variable], manual_fixes['standard_pt']))
            )
            
            # Update unstandardized terms in manual fixes file
            unstandardized = df[df['standard_pt'].isna()][pt_variable].unique()
            if len(unstandardized) > 0:
                new_manual_fixes = pd.DataFrame({
                    pt_variable: unstandardized,
                    'standard_pt': pd.NA
                })
                manual_fixes = pd.concat([manual_fixes, new_manual_fixes]).drop_duplicates()
                manual_fixes.to_csv(
                    self.external_dir / 'manual_fixes' / 'pt_fixed.csv',  # Updated path
                    sep=';',
                    index=False
                )
                logging.warning(
                    f"{len(unstandardized)} terms not standardized. "
                    f"Added to pt_fixed.csv: {', '.join(unstandardized)}"
                )
            
            # Calculate final standardization percentage
            final_std_pct = round(
                df['standard_pt'].notna().sum() * 100 / len(df[~df[pt_variable].isna()]),
                3
            )
            logging.info(f"Final standardized PT percentage: {final_std_pct}%")
            
            # Replace original with standardized
            df[pt_variable] = df['standard_pt'].astype(str).replace('nan', '')  # Handle categorical
            df = df.drop('standard_pt', axis=1)
            
            return df
            
        except Exception as e:
            logging.error(f"Error standardizing {pt_variable}: {str(e)}")
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
        """Standardize sex values exactly matching R implementation.
        
        According to ASC_NTS.pdf, sex should only be 'F', 'M', or 'UNK'.
        However, analysis found additional values like 'NS', 'YR', 'P', 'I', 'T'.
        Following R implementation, we only keep 'F' and 'M', converting others to NA.
        
        Args:
            df: DataFrame with sex column
            
        Returns:
            DataFrame with standardized sex values
        """
        try:
            df = df.copy()
            
            # If sex column doesn't exist, add it with empty strings
            if 'sex' not in df.columns:
                df['sex'] = ''
                logging.warning("Sex column not found - initialized with empty strings")
                
            # Convert to string and lowercase
            df['sex'] = df['sex'].fillna('').astype(str).str.lower().str.strip()
            
            # Following R: Demo[!sex %in% c("F","M")]$sex <- NA
            # Only keep 'F' and 'M', convert others to NA
            df.loc[~df['sex'].isin(['f', 'm']), 'sex'] = np.nan
            
            # Convert to uppercase to match R
            df['sex'] = df['sex'].str.upper()
            
            # Log distribution
            value_counts = df['sex'].value_counts(dropna=False)
            logging.info("Sex value distribution:")
            for val, count in value_counts.items():
                logging.info(f"  {val}: {count} ({count/len(df)*100:.1f}%)")
            
            return df
            
        except Exception as e:
            logging.error(f"Error standardizing sex values: {str(e)}")
            # On any error, return original DataFrame
            return df

    def standardize_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize age values exactly matching R implementation.
        
        Handles three fields:
        - age (numeric value)
        - age_cod (unit: YR, MON, WK, DY, HR)
        - age_group (category)
        
        Converts all ages to days and years, handling special cases:
        - Converts character values to NA
        - Converts negative numbers to positive (e.g., "-08" to "8")
        - Handles decimal ages > 122 as compilation errors
        - Validates against max age of 122 years
        
        Args:
            df: DataFrame with age columns
            
        Returns:
            DataFrame with standardized age values
        """
        try:
            df = df.copy()
            
            # If age column doesn't exist, add it with empty strings
            if 'age' not in df.columns:
                df['age'] = ''
                logging.warning("Age column not found - initialized with empty strings")
                
            if 'age_cod' not in df.columns:
                df['age_cod'] = ''
                logging.warning("Age code column not found - initialized with empty strings")
                
            # Convert age to numeric, handling special cases
            df['age'] = df['age'].fillna('')
            
            # Clean age values before conversion:
            # 1. Strip whitespace
            # 2. Handle negative values by removing minus sign (e.g., "-08" to "8")
            # 3. Remove any commas
            df['age'] = (df['age'].astype(str)
                        .str.strip()
                        .str.replace('-', '')  # Remove minus signs
                        .str.replace(',', '')  # Remove commas
                        .str.replace(r'[^\d\.]', '', regex=True))  # Keep only digits and decimal points
            
            # Convert to numeric, coercing errors to NaN
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            
            # Create age corrector exactly as in R
            conditions = [
                df['age_cod'] == 'DEC',
                df['age_cod'].isin(['YR', '']) | df['age_cod'].isna(),
                df['age_cod'] == 'MON',
                df['age_cod'] == 'WK',
                df['age_cod'] == 'DY',
                df['age_cod'] == 'HR',
                df['age_cod'] == 'SEC',
                df['age_cod'] == 'MIN'
            ]
            
            choices = [
                3650,                          # DEC
                365,                           # YR or NA
                30.41667,                      # MON
                7,                             # WK
                1,                             # DY
                0.00011415525114155251,        # HR
                3.1709791983764586e-08,        # SEC
                1.9025875190259e-06            # MIN
            ]
            
            df['age_corrector'] = np.select(conditions, choices, default=np.nan)
            
            # Calculate age in days
            df['age_in_days'] = np.round(df['age'] * df['age_corrector'])
            
            # Handle plausible compilation error (age > 122 years)
            max_age_days = 122 * 365
            df.loc[df['age_in_days'] > max_age_days, 'age_in_days'] = np.where(
                df.loc[df['age_in_days'] > max_age_days, 'age_cod'] == 'DEC',
                df.loc[df['age_in_days'] > max_age_days, 'age_in_days'] / 
                df.loc[df['age_in_days'] > max_age_days, 'age_corrector'],
                np.nan
            )
            
            # Calculate age in years
            df['age_in_years'] = np.round(df['age_in_days'] / 365)
            
            # Drop temporary columns
            df = df.drop(columns=['age_corrector', 'age', 'age_cod'])
            
            # Log age distribution
            logging.info("Age distribution (years):")
            age_stats = df['age_in_years'].describe()
            for stat, value in age_stats.items():
                logging.info(f"  {stat}: {value}")
                
            return df
            
        except Exception as e:
            logging.error(f"Error standardizing age values: {str(e)}")
            # On any error, return original DataFrame
            return df

    def standardize_age_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize age groups according to NIH guidelines.
        
        Age group thresholds (NIH style guide):
        - N (neonate): ≤28 days
        - I (infant): <2 years
        - C (child): <12 years
        - T (teenager): <18 years
        - A (adult): <65 years
        - E (elderly): ≥65 years
        
        Args:
            df: DataFrame with age_in_days and age_in_years columns
        
        Returns:
            DataFrame with standardized age groups
        """
        try:
            df = df.copy()
            
            # Initialize age_grp column with NA
            df['age_grp'] = np.nan
            
            # Only process rows with valid ages
            mask = df['age_in_years'].notna()
            if mask.any():
                # Start with oldest group and work backwards
                # Exactly matching R: 
                # Demo[!is.na(age_in_years)]$age_grp_st <- "E"
                # Demo[age_in_years < 65]$age_grp_st <- "A"
                # Demo[age_in_years < 18]$age_grp_st <- "T"
                # Demo[age_in_years < 12]$age_grp_st <- "C"
                # Demo[age_in_years < 2]$age_grp_st <- "I"
                # Demo[age_in_days <28]$age_grp_st <- "N"
                
                df.loc[mask, 'age_grp'] = 'E'
                df.loc[mask & (df['age_in_years'] < 65), 'age_grp'] = 'A'
                df.loc[mask & (df['age_in_years'] < 18), 'age_grp'] = 'T'
                df.loc[mask & (df['age_in_years'] < 12), 'age_grp'] = 'C'
                df.loc[mask & (df['age_in_years'] < 2), 'age_grp'] = 'I'
                df.loc[mask & (df['age_in_days'] < 28), 'age_grp'] = 'N'
            
            # Log distribution
            value_counts = df['age_grp'].value_counts(dropna=False)
            total = len(df)
            logging.info("Age group distribution:")
            for group, count in value_counts.items():
                group_name = {
                    'N': 'Neonate (≤28 days)',
                    'I': 'Infant (<2 years)',
                    'C': 'Child (<12 years)',
                    'T': 'Teenager (<18 years)',
                    'A': 'Adult (<65 years)',
                    'E': 'Elderly (≥65 years)',
                    np.nan: 'Unknown'
                }.get(group, str(group))
                percent = round(count/total * 100, 1)
                logging.info(f"  {group_name}: {count} ({percent}%)")
            
            return df
            
        except Exception as e:
            logging.error(f"Error creating age groups: {str(e)}")
            # On any error, return original DataFrame
            return df

    def standardize_weight(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize weight values exactly matching R implementation.
        
        According to ASC_NTS.pdf guidelines, only these units are valid:
        - KG/KGS - Kilograms
        - LBS/IB - Pounds
        - GMS - Grams
        - MG - Milligrams
        
        Missing values (NA) are considered as kilograms.
        Values > 635 kg (heaviest documented human weight) are set to NA.
        
        Args:
            df: DataFrame with weight columns (wt, wt_cod)
        
        Returns:
            DataFrame with standardized weight in kg
        """
        try:
            df = df.copy()
            
            # If weight columns don't exist, add them with empty strings
            if 'wt' not in df.columns:
                df['wt'] = ''
                logging.warning("Weight column not found - initialized with empty strings")
                
            if 'wt_cod' not in df.columns:
                df['wt_cod'] = ''
                logging.warning("Weight code column not found - initialized with empty strings")
                
            # Clean weight values before conversion:
            # 1. Strip whitespace
            # 2. Handle negative values by removing minus sign
            # 3. Remove any commas and units embedded in the value
            df['wt'] = (df['wt'].astype(str)
                       .str.strip()
                       .str.replace('-', '')  # Remove minus signs
                       .str.replace(',', '')  # Remove commas
                       .str.replace(r'[^\d\.]', '', regex=True))  # Keep only digits and decimal points
            
            # Convert to numeric, coercing errors to NaN
            df['wt'] = pd.to_numeric(df['wt'], errors='coerce')
            
            # Create weight corrector exactly as in R
            # Demo[wt_cod %in%c("LBS", "IB")]$wt_corrector <- 0.453592
            # Demo[wt_cod%in% c("KG", "KGS")]$wt_corrector <- 1
            # Demo[wt_cod=="GMS"]$wt_corrector <- 0.001
            # Demo[wt_cod=="MG"]$wt_corrector <- 1e-06
            # Demo[is.na(wt_cod)]$wt_corrector <- 1
            
            conditions = [
                df['wt_cod'].isin(['LBS', 'IB']),
                df['wt_cod'].isin(['KG', 'KGS']),
                df['wt_cod'] == 'GMS',
                df['wt_cod'] == 'MG',
                df['wt_cod'].isna() | (df['wt_cod'] == '')
            ]
            
            choices = [
                0.453592,  # LBS/IB to kg
                1.0,       # KG/KGS (already in kg)
                0.001,     # GMS to kg
                1e-06,     # MG to kg
                1.0        # NA/empty treated as kg
            ]
            
            df['wt_corrector'] = np.select(conditions, choices, default=np.nan)
            
            # Calculate weight in kg
            # Demo <- Demo[,wt_in_kgs:=round(abs(as.numeric(wt))*wt_corrector)]
            df['wt_in_kgs'] = np.round(df['wt'] * df['wt_corrector'])
            
            # Handle implausible weights (> 635 kg)
            # Demo[wt_in_kgs>635]$wt_in_kgs <- NA
            df.loc[df['wt_in_kgs'] > 635, 'wt_in_kgs'] = np.nan
            
            # Drop temporary columns
            df = df.drop(columns=['wt_corrector', 'wt', 'wt_cod'])
            
            # Log weight distribution
            logging.info("Weight distribution (kg):")
            weight_stats = df['wt_in_kgs'].describe()
            for stat, value in weight_stats.items():
                logging.info(f"  {stat}: {value}")
                
            return df
            
        except Exception as e:
            logging.error(f"Error standardizing weight values: {str(e)}")
            return df

    def standardize_country(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize country codes using ISO standards.
        
        Handles two country fields:
        - occr_country: Where the event occurred
        - reporter_country: Where the report was submitted from
        
        Uses ISO country codes mapping from countries.csv, with special cases:
        - "COUNTRY NOT SPECIFIED" -> NA
        - "A1" -> NA
        - Continent codes (e.g., XE -> Europe, QU -> Oceania)
        
        Note: Requires countries.csv in external_data/manual_fixes/ directory
        with format: country;Country_Name
        
        Args:
            df: DataFrame with country columns
            
        Returns:
            DataFrame with standardized country names
        """
        try:
            df = df.copy()
            
            # Check for required columns
            if 'occr_country' not in df.columns:
                df['occr_country'] = ''
                logging.warning("Occurrence country column not found - initialized with empty strings")
                
            if 'reporter_country' not in df.columns:
                df['reporter_country'] = ''
                logging.warning("Reporter country column not found - initialized with empty strings")
                
            try:
                # Read country mapping file using the correct path
                countries_file = self.external_dir / 'manual_fixes' / 'countries.csv'
                
                countries_df = pd.read_csv(
                    countries_file, 
                    sep=';',
                    dtype={'country': str, 'Country_Name': str},
                    keep_default_na=False,
                    na_values=['']
                )
                
                # Handle special case for Namibia (NA)
                countries_df.loc[countries_df['Country_Name'].isna(), 'Country_Name'] = 'NA'
                
                # Create mapping dictionary
                country_map = dict(zip(countries_df['country'], countries_df['Country_Name']))
                
                # Special mappings for continents and unspecified
                special_mappings = {
                    'XE': 'Europe',
                    'QU': 'Oceania',
                    'COUNTRY NOT SPECIFIED': np.nan,
                    'A1': np.nan,
                    '': np.nan
                }
                country_map.update(special_mappings)
                
                # Map occurrence country
                df['occr_country'] = df['occr_country'].map(country_map)
                
                # Map reporter country
                df['reporter_country'] = df['reporter_country'].map(country_map)
                
                # Log country distributions
                logging.info("Occurrence country distribution:")
                occr_stats = df['occr_country'].value_counts().head()
                for country, count in occr_stats.items():
                    logging.info(f"  {country}: {count}")
                    
                logging.info("Reporter country distribution:")
                reporter_stats = df['reporter_country'].value_counts().head()
                for country, count in reporter_stats.items():
                    logging.info(f"  {country}: {count}")
            
                # Check for unmapped countries and log them
                unmapped_occr = set(df[~df['occr_country'].isna()]['occr_country'].unique()) - set(country_map.values())
                unmapped_reporter = set(df[~df['reporter_country'].isna()]['reporter_country'].unique()) - set(country_map.values())
                unmapped = unmapped_occr | unmapped_reporter
                if unmapped:
                    logging.warning(f"Found unmapped countries: {unmapped}")
            
                return df
                
            except FileNotFoundError:
                logging.error(
                    "countries.csv not found in external_data/manual_fixes/. "
                    "Please ensure the file exists with format: country;Country_Name"
                )
                return df
                
        except Exception as e:
            logging.error(f"Error standardizing country values: {str(e)}")
            return df

    def standardize_occupation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize occupation codes for reporters.
        
        Valid occupation codes:
        - MD: Medical Doctor
        - CN: Consumer
        - OT: Other
        - PH: Pharmacist
        - HP: Health Practitioner
        - LW: Lawyer
        - RN: Registered Nurse
        
        All other values (including SALES, 20120210) are converted to NA.
        
        Args:
            df: DataFrame with occupation column (occp_cod)
        
        Returns:
            DataFrame with standardized occupation codes
        """
        try:
            df = df.copy()
            
            if 'occp_cod' not in df.columns:
                df['occp_cod'] = ''
                logging.warning("Occupation code column not found - initialized with empty strings")
            
            # Valid occupation codes matching R implementation
            valid_codes = {'MD', 'CN', 'OT', 'PH', 'HP', 'LW', 'RN'}
            
            # Log initial distribution
            logging.info("Initial occupation code distribution:")
            initial_dist = df['occp_cod'].value_counts()
            for code, count in initial_dist.items():
                logging.info(f"  {code}: {count}")
            
            # Convert invalid codes to NA (exactly matching R: Demo[!occp_cod%in%c("MD","CN","OT","PH","HP","LW", "RN")]$occp_cod <- NA)
            df.loc[~df['occp_cod'].isin(valid_codes), 'occp_cod'] = pd.NA
            
            # Log final distribution
            logging.info("Final occupation code distribution:")
            final_dist = df['occp_cod'].value_counts()
            for code, count in final_dist.items():
                logging.info(f"  {code}: {count}")
            
            # Log specific invalid codes found (e.g., SALES, 20120210)
            invalid_codes = set(initial_dist.index) - valid_codes
            if invalid_codes:
                logging.warning(f"Converted invalid occupation codes to NA: {invalid_codes}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error standardizing occupation codes: {str(e)}")
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
        route_map = pd.read_csv(self.external_dir / 'manual_fixes' / 'routes.csv', 
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
            total_routes = len(df)
            standardized = df['route'].notna().sum()
            logging.info(f"Route standardization results:")
            logging.info(f"  Total routes: {total_routes}")
            logging.info(f"  Standardized: {standardized} ({100*standardized/total_routes:.1f}%)")
            logging.info(f"  Unstandardized: {total_routes-standardized} ({100*(total_routes-standardized)/total_routes:.1f}%)")
        
        return df

    def standardize_dose_form(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize medication dose forms.
        
        Args:
            df: DataFrame with dose form information
            
        Returns:
            DataFrame with standardized dose forms
        """
        if 'dose_form' not in df.columns:
            return df
        
        # Clean dose form strings
        df['dose_form'] = df['dose_form'].str.lower().str.strip()
        
        # Load dose form standardization mapping
        dose_form_st = pd.read_csv(
            self.external_dir / 'manual_fixes' / 'dose_form_st.csv',
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
        if 'dose_form' in df.columns:
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
            self.external_dir / 'manual_fixes' / 'unit_st.csv',
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

    def standardize_sources(self, df: pd.DataFrame, quarter_name: str, file_name: str) -> pd.DataFrame:
        """Standardize report source information."""
        try:
            # Required columns exactly as defined in documentation.html
            required_columns = {
                'isr': ['ISR'],
                'rpsr_cod': ['RPSR_COD']
            }
            
            # Process each required column
            for target_col, source_cols in required_columns.items():
                found = False
                for col in source_cols:
                    if col in df.columns:
                        if col != target_col:
                            df = df.rename(columns={col: target_col})
                        found = True
                        break
                
                if not found:
                    self.logger.warning(f"({quarter_name}) {file_name}: Required column '{target_col}' not found, adding with default value: <NA>")
                    df[target_col] = pd.NA
            
            return df
            
        except Exception as e:
            self.logger.error(f"({quarter_name}) {file_name}: Error in standardize_sources: {str(e)}")
            return df

    def standardize_therapies(self, df: pd.DataFrame, quarter_name: str, file_name: str) -> pd.DataFrame:
        """Standardize therapy information."""
        try:
            # Required columns exactly as defined in documentation.html
            required_columns = {
                'isr': ['ISR'],
                'dsg_drug_seq': ['DSG_DRUG_SEQ'],
                'start_dt': ['START_DT'],
                'end_dt': ['END_DT'],
                'dur': ['DUR'],
                'dur_cod': ['DUR_COD']
            }
            
            # Process each required column
            for target_col, source_cols in required_columns.items():
                found = False
                for col in source_cols:
                    if col in df.columns:
                        if col != target_col:
                            df = df.rename(columns={col: target_col})
                        found = True
                        break
                
                if not found:
                    self.logger.warning(f"({quarter_name}) {file_name}: Required column '{target_col}' not found, adding with default value: <NA>")
                    df[target_col] = pd.NA
            
            return df
            
        except Exception as e:
            self.logger.error(f"({quarter_name}) {file_name}: Error in standardize_therapies: {str(e)}")
            return df

    def standardize_indications(self, df: pd.DataFrame, quarter_name: str, file_name: str) -> pd.DataFrame:
        """Standardize indications information."""
        try:
            # Required columns exactly as defined in documentation.html
            required_columns = {
                'isr': ['ISR'],
                'drug_seq': ['DRUG_SEQ'],
                'indi_pt': ['INDI_PT']
            }
            
            # Process each required column
            for target_col, source_cols in required_columns.items():
                found = False
                for col in source_cols:
                    if col in df.columns:
                        if col != target_col:
                            df = df.rename(columns={col: target_col})
                        found = True
                        break
                
                if not found:
                    self.logger.warning(f"({quarter_name}) {file_name}: Required column '{target_col}' not found, adding with default value: <NA>")
                    df[target_col] = pd.NA
            
            return df
            
        except Exception as e:
            self.logger.error(f"({quarter_name}) {file_name}: Error in standardize_indications: {str(e)}")
            return df

    def process_quarters(self, quarters_dir: Path, parallel: bool = False, n_workers: Optional[int] = None) -> str:
        """Process all FAERS quarters with summary reporting.
        
        Args:
            quarters_dir: Directory containing FAERS quarter data
            parallel: If True, process quarters in parallel using dask
            n_workers: Number of worker processes for parallel processing
            
        Returns:
            Markdown formatted summary report
        """
        try:
            # Ensure quarters_dir is a Path and exists
            quarters_dir = Path(quarters_dir).resolve()
            if not quarters_dir.exists():
                raise FileNotFoundError(f"Quarters directory not found at {quarters_dir}")
            
            # Get list of quarter directories
            quarter_dirs = [d for d in quarters_dir.iterdir() if d.is_dir()]
            if not quarter_dirs:
                raise ValueError(f"No quarter directories found in {quarters_dir}")
            
            self.logger.info(f"Found {len(quarter_dirs)} quarters to process")
            
            # Initialize summary tracker
            summary = FAERSProcessingSummary()
            
            if parallel and len(quarter_dirs) > 1:
                # Set up dask client for parallel processing
                n_workers = n_workers or min(len(quarter_dirs), os.cpu_count() or 1)
                cluster = LocalCluster(n_workers=n_workers)
                client = Client(cluster)
                self.logger.info(f"Processing {len(quarter_dirs)} quarters in parallel with {n_workers} workers")
                
                # Create dask bag of quarter directories
                quarters_bag = db.from_sequence(quarter_dirs)
                results = quarters_bag.map(self.process_quarter).compute()
                
                # Add results to summary
                for quarter_dir, result in zip(quarter_dirs, results):
                    if result:
                        summary.add_quarter_summary(quarter_dir.name, result)
                
                # Close dask client
                client.close()
                cluster.close()
                
            else:
                # Process quarters sequentially
                self.logger.info(f"Processing {len(quarter_dirs)} quarters sequentially")
                for quarter_dir in tqdm(quarter_dirs, desc="Processing quarters"):
                    try:
                        result = self.process_quarter(quarter_dir)
                        if result:
                            summary.add_quarter_summary(quarter_dir.name, result)
                    except Exception as e:
                        self.logger.error(f"Error processing quarter {quarter_dir.name}: {str(e)}")
            
            # Generate and return report
            return summary.generate_markdown_report()
            
        except Exception as e:
            self.logger.error(f"Error in process_quarters: {str(e)}")
            return f"Error processing quarters: {str(e)}"

    def process_quarter(self, quarter_dir: Path) -> Optional[QuarterSummary]:
        """Process a single quarter.
        
        Args:
            quarter_dir: Path to quarter directory
            
        Returns:
            QuarterSummary if successful, None if error
        """
        try:
            quarter_dir = Path(quarter_dir).resolve()
            if not quarter_dir.exists():
                raise FileNotFoundError(f"Quarter directory not found at {quarter_dir}")
            
            quarter_name = quarter_dir.name
            self.logger.info(f"Processing quarter {quarter_name}")
            
            # Initialize quarter summary
            summary = QuarterSummary(quarter=quarter_name)
            start_time = time.time()
            
            # Process each file type
            file_types = {
                'demo': ['DEMO*.txt', 'Demographics'],
                'drug': ['DRUG*.txt', 'Drug'],
                'reac': ['REAC*.txt', 'Reaction'],
                'indi': ['INDI*.txt', 'Indication'],
                'outc': ['OUTC*.txt', 'Outcome'],
                'ther': ['THER*.txt', 'Therapy']
            }
            
            processed_data = {}
            for file_type, (pattern, desc) in file_types.items():
                try:
                    # Find matching files (case-insensitive)
                    files = list(quarter_dir.glob(pattern))
                    files.extend(quarter_dir.glob(pattern.lower()))
                    
                    if not files:
                        self.logger.warning(f"No {desc} files found in {quarter_name}")
                        continue
                        
                    # Process each file
                    for file_path in files:
                        try:
                            df = read_and_clean_file(file_path)
                            if df is not None and not df.empty:
                                df = self.standardize_data(df, file_type, str(file_path), quarter_name)
                                processed_data[file_type] = df
                                
                                # Update summary statistics
                                if file_type == 'demo':
                                    summary.demo_summary.total_rows = len(df)
                                elif file_type == 'drug':
                                    summary.drug_summary.total_rows = len(df)
                                elif file_type == 'reac':
                                    summary.reac_summary.total_rows = len(df)
                                    
                        except Exception as e:
                            self.logger.error(f"Error processing {desc} file {file_path}: {str(e)}")
                            
                except Exception as e:
                    self.logger.error(f"Error processing {desc} files for {quarter_name}: {str(e)}")
            
            # Save processed data
            quarter_output_dir = self.output_dir / quarter_name
            quarter_output_dir.mkdir(parents=True, exist_ok=True)
            
            for file_type, df in processed_data.items():
                try:
                    output_path = quarter_output_dir / f"{file_type}.csv"
                    df.to_csv(output_path, index=False)
                    self.logger.info(f"Saved {file_type} data to {output_path}")
                except Exception as e:
                    self.logger.error(f"Error saving {file_type} data for {quarter_name}: {str(e)}")
            
            # Update summary timing
            summary.processing_time = time.time() - start_time
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error processing quarter {quarter_dir}: {str(e)}")
            return None

    def standardize_data(self, df: pd.DataFrame, data_type: str, quarter_name: str, file_name: str) -> pd.DataFrame:
        """Standardize data based on its type.
        
        Args:
            df: DataFrame to standardize
            data_type: Type of data (demo, drug, reac, etc.)
            quarter_name: Name of the quarter being processed
            file_name: Name of the file being processed
            
        Returns:
            Standardized DataFrame
        """
        try:
            # Apply type-specific standardization
            if data_type == 'demo':
                df = self.standardize_demographics(df, quarter_name, file_name)
            elif data_type == 'drug':
                df = self.standardize_drug_info(df, quarter_name, file_name)
            elif data_type == 'reac':
                df = self.standardize_reactions(df, quarter_name, file_name)
            elif data_type == 'outc':
                df = self.standardize_outcomes(df, quarter_name, file_name)
            elif data_type == 'rpsr':
                df = self.standardize_sources(df, quarter_name, file_name)
            elif data_type == 'ther':
                df = self.standardize_therapies(df, quarter_name, file_name)
            elif data_type == 'indi':
                df = self.standardize_indications(df, quarter_name, file_name)
            else:
                self.logger.warning(f"({quarter_name}) {file_name}: Unknown data type: {data_type}")
                return df
                
            # Final cleanup
            # Replace NaN with empty strings for consistency
            df = df.fillna('')
            
            # Log standardization results
            self.logger.info(f"({quarter_name}) {file_name}: Successfully standardized {len(df):,} rows of {data_type} data")
            
            return df
            
        except Exception as e:
            self.logger.error(f"({quarter_name}) {file_name}: Error standardizing {data_type} data: {str(e)}")
            return df

    def calculate_time_to_onset(self, demo_df: pd.DataFrame, ther_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time to onset between drug administration and event.
        
        Matches R implementation for calculating days between start_dt and event_dt.
        Special handling for dates before 2013 due to auto-completion issues:
        - Dates ending in '01' before 2013 are considered unreliable
        - Negative or zero time to onset before 2013 are set to NA
        
        Args:
            demo_df: Demographics DataFrame with event_dt
            ther_df: Therapy DataFrame with start_dt
        
        Returns:
            DataFrame with time_to_onset calculated
        """
        try:
            # Get event dates from demo data
            event_dates = demo_df[['primaryid', 'event_dt']].copy()
            event_dates = event_dates[event_dates['event_dt'].notna()]
            
            # Merge with therapy data
            df = pd.merge(event_dates, ther_df, on='primaryid', how='right')
            
            def to_date(x):
                if pd.isna(x) or len(str(x)) != 8:
                    return pd.NaT
                return pd.to_datetime(str(x), format='%Y%m%d')
            
            # Calculate time to onset
            df['time_to_onset'] = (to_date(df['event_dt']) - to_date(df['start_dt'])).dt.days + 1
            
            # Handle dates before 2013 with special rules
            mask_pre_2013 = (df['event_dt'] <= 20121231)
            mask_invalid = ((df['time_to_onset'] <= 0) & mask_pre_2013)
            
            # Set invalid time to onset to NA
            df.loc[mask_invalid, 'time_to_onset'] = np.nan
            
            # Log time to onset statistics
            valid_tto = df['time_to_onset'].dropna()
            if len(valid_tto) > 0:
                neg_tto = (valid_tto < 0).mean() * 100
                logging.info(f"Negative time to onset: {neg_tto:.1f}% of valid cases")
                
                # Log pre-2013 statistics
                pre_2013 = df[mask_pre_2013]
                if len(pre_2013) > 0:
                    pre_2013_neg = (pre_2013['time_to_onset'] < 0).mean() * 100
                    logging.info(f"Pre-2013 negative time to onset: {pre_2013_neg:.1f}%")
                    
                    # Check for dates ending in '01'
                    ends_01_mask = df['event_dt'].astype(str).str.endswith('01')
                    pre_2013_01 = (ends_01_mask & mask_pre_2013).mean() * 100
                    logging.info(f"Pre-2013 dates ending in '01': {pre_2013_01:.1f}%")
        
            return df
        
        except Exception as e:
            logging.error(f"Error calculating time to onset: {str(e)}")
            return ther_df

    def remove_duplicate_primaryids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicated primaryids keeping only the most recent quarter entry.
        
        Matches R implementation:
        Demo <- Demo[Demo[,.I[quarter==last(quarter)],by=primaryid]$V1]
        
        Args:
            df: DataFrame with potential duplicate primaryids
        
        Returns:
            DataFrame with duplicates removed, keeping most recent quarter
        """
        try:
            if 'primaryid' not in df.columns or 'quarter' not in df.columns:
                logging.warning("Cannot remove duplicates: missing required columns 'primaryid' or 'quarter'")
                return df
                
            # Get indices of rows to keep (most recent quarter for each primaryid)
            indices = df.groupby('primaryid')['quarter'].idxmax()
            
            # Keep only the selected rows
            df_deduped = df.loc[indices]
            
            # Log duplicate removal statistics
            total_rows = len(df)
            kept_rows = len(df_deduped)
            removed_rows = total_rows - kept_rows
            
            if removed_rows > 0:
                logging.info(f"Removed {removed_rows} duplicate primaryid entries, keeping {kept_rows} unique entries")
                
                # Log some examples of removed duplicates for verification
                dupes = df[df.duplicated(subset=['primaryid'], keep=False)].sort_values(['primaryid', 'quarter'])
                if not dupes.empty:
                    sample_dupes = dupes.groupby('primaryid').head(2).head(6)  # Show up to 3 pairs of duplicates
                    logging.debug("Sample of removed duplicates (showing primaryid, quarter, caseid):")
                    for _, group in sample_dupes.groupby('primaryid'):
                        logging.debug(f"\nPrimaryid: {group['primaryid'].iloc[0]}")
                        for _, row in group.iterrows():
                            logging.debug(f"Quarter: {row['quarter']}, Caseid: {row.get('caseid', 'N/A')}")
        
            return df_deduped
        
        except Exception as e:
            logging.error(f"Error removing duplicate primaryids: {str(e)}")
            return df

    def standardize_drug_info(self, df: pd.DataFrame, quarter_name: str, file_name: str) -> pd.DataFrame:
        """Standardize drug information."""
        try:
            # Print actual columns for debugging
            self.logger.info(f"Columns in {file_name}: {list(df.columns)}")
            
            # Map the actual column names to our target names
            column_map = {
                'ISR': 'isr',
                'DRUG_SEQ': 'drug_seq',
                'ROLE_COD': 'role_cod',
                'DRUGNAME': 'drugname',
                'VAL_VBM': 'prod_ai',
                'ROUTE': 'route',
                'DOSE_VBM': 'dose_amt',
                'DECHAL': 'dechal',
                'RECHAL': 'rechal',
                'LOT_NUM': 'lot_num',
                'EXP_DT': 'exp_dt',
                'NDA_NUM': 'nda_num'
            }
            
            # First rename any columns that exist
            existing_cols = set(df.columns) & set(column_map.keys())
            if existing_cols:
                df = df.rename(columns={col: column_map[col] for col in existing_cols})
            
            # Then add missing columns with NA
            missing_cols = set(column_map.values()) - set(df.columns)
            for col in missing_cols:
                self.logger.warning(f"({quarter_name}) {file_name}: Required column '{col}' not found, adding with default value: <NA>")
                df[col] = pd.NA
            
            return df
            
        except Exception as e:
            self.logger.error(f"({quarter_name}) {file_name}: Error in standardize_drug_info: {str(e)}")
            return df

    def standardize_reactions(self, df: pd.DataFrame, quarter_name: str, file_name: str) -> pd.DataFrame:
        """Standardize reactions information."""
        try:
            # Required columns exactly as defined in documentation.html
            required_columns = {
                'ISR': 'isr',
                'PT': 'pt',
                'DRUG_REC_ACT': 'drug_rec_act'
            }
            
            # Process each required column
            for target_col, source_cols in required_columns.items():
                found = False
                for col in source_cols:
                    if col in df.columns:
                        if col != target_col:
                            df = df.rename(columns={col: target_col})
                        found = True
                        break
                
                if not found:
                    self.logger.warning(f"({quarter_name}) {file_name}: Required column '{target_col}' not found, adding with default value: <NA>")
                    df[target_col] = pd.NA
            
            return df
            
        except Exception as e:
            self.logger.error(f"({quarter_name}) {file_name}: Error in standardize_reactions: {str(e)}")
            return df

    def standardize_indications(self, df: pd.DataFrame, quarter_name: str, file_name: str) -> pd.DataFrame:
        """Standardize indications information."""
        try:
            # Required columns exactly as defined in documentation.html
            required_columns = {
                'ISR': 'isr',
                'DRUG_SEQ': 'drug_seq',
                'INDI_PT': 'indi_pt'
            }
            
            # Process each required column
            for target_col, source_cols in required_columns.items():
                found = False
                for col in source_cols:
                    if col in df.columns:
                        if col != target_col:
                            df = df.rename(columns={col: target_col})
                        found = True
                        break
                
                if not found:
                    self.logger.warning(f"({quarter_name}) {file_name}: Required column '{target_col}' not found, adding with default value: <NA>")
                    df[target_col] = pd.NA
            
            return df
            
        except Exception as e:
            self.logger.error(f"({quarter_name}) {file_name}: Error in standardize_indications: {str(e)}")
            return df

    def standardize_outcomes(self, df: pd.DataFrame, quarter_name: str, file_name: str) -> pd.DataFrame:
        """Standardize outcomes information."""
        try:
            # Required columns exactly as defined in documentation.html
            required_columns = {
                'ISR': 'isr',
                'OUTC_COD': 'outc_cod'
            }
            
            # Process each required column
            for target_col, source_cols in required_columns.items():
                found = False
                for col in source_cols:
                    if col in df.columns:
                        if col != target_col:
                            df = df.rename(columns={col: target_col})
                        found = True
                        break
                
                if not found:
                    self.logger.warning(f"({quarter_name}) {file_name}: Required column '{target_col}' not found, adding with default value: <NA>")
                    df[target_col] = pd.NA
            
            return df
            
        except Exception as e:
            self.logger.error(f"({quarter_name}) {file_name}: Error in standardize_outcomes: {str(e)}")
            return df

    def standardize_therapies(self, df: pd.DataFrame, quarter_name: str, file_name: str) -> pd.DataFrame:
        """Standardize therapy information."""
        try:
            # Required columns exactly as defined in documentation.html
            required_columns = {
                'ISR': 'isr',
                'DSG_DRUG_SEQ': 'dsg_drug_seq',
                'START_DT': 'start_dt',
                'END_DT': 'end_dt',
                'DUR': 'dur',
                'DUR_COD': 'dur_cod'
            }
            
            # Process each required column
            for target_col, source_cols in required_columns.items():
                found = False
                for col in source_cols:
                    if col in df.columns:
                        if col != target_col:
                            df = df.rename(columns={col: target_col})
                        found = True
                        break
                
                if not found:
                    self.logger.warning(f"({quarter_name}) {file_name}: Required column '{target_col}' not found, adding with default value: <NA>")
                    df[target_col] = pd.NA
            
            return df
            
        except Exception as e:
            self.logger.error(f"({quarter_name}) {file_name}: Error in standardize_therapies: {str(e)}")
            return df

def read_and_clean_file(file_path: Path) -> Tuple[List[str], str]:
    """Read and clean the file, detecting delimiter."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        # Read sample for delimiter detection
        sample_lines = [next(f) for _ in range(min(1000, sum(1 for _ in f)))]
        
    # Detect delimiter
    delimiters = ['$', '|', '\t', ',']
    max_consistent_cols = 0
    best_delimiter = ','
    
    for delimiter in delimiters:
        cols_per_row = [len(line.split(delimiter)) for line in sample_lines]
        # Get most common column count
        from collections import Counter
        col_counts = Counter(cols_per_row)
        if col_counts:
            most_common_count = col_counts.most_common(1)[0][1]
            if most_common_count > max_consistent_cols:
                max_consistent_cols = most_common_count
                best_delimiter = delimiter
                
    # Clean all lines
    cleaned_lines = [clean_line(line) for line in sample_lines]
    
    return cleaned_lines, best_delimiter

def clean_line(line: str) -> str:
    """Clean problematic characters from a line."""
    # Remove null bytes
    line = line.replace('\0', '')
    # Normalize line endings
    line = line.strip('\r\n')
    # Handle escaped quotes
    line = re.sub(r'(?<!\\)"', '\\"', line)
    return line
