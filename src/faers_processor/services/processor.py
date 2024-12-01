"""Service for processing FAERS data files."""
import logging
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from tqdm import tqdm

from .standardizer import DataStandardizer


class FAERSProcessor:
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

    def __init__(self, data_dir: Path, external_dir: Path):
        """Initialize processor with data and external directories."""
        self.data_dir = data_dir
        self.standardizer = DataStandardizer(external_dir)

    def process_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process demographics data."""
        df = df.copy()

        # Standardize column names
        df = self.standardize_columns(df)

        # Apply standardizations
        df = self.standardizer.standardize_sex(df)
        df = self.standardizer.standardize_age(df)
        df = self.standardizer.standardize_weight(df)
        df = self.standardizer.standardize_country(df)
        df = self.standardizer.standardize_occupation(df)
        df = self.standardizer.standardize_dates(df, ['init_fda_dt', 'event_dt', 'rept_dt'])

        return df

    def process_drugs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process drug data."""
        df = df.copy()

        # Standardize column names
        df = self.standardize_columns(df)

        # Apply standardizations
        df = self.standardizer.standardize_route(df)
        df = self.standardizer.standardize_dose_form(df)
        df = self.standardizer.standardize_dose_freq(df)

        # MedDRA standardization for drug reactions
        if 'drug_rec_act' in df.columns:
            df = self.standardizer.standardize_pt(df, 'drug_rec_act')

        return df

    def process_reactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process reaction data."""
        df = df.copy()

        # Standardize column names
        df = self.standardize_columns(df)

        # MedDRA standardization for reactions
        if 'pt' in df.columns:
            df = self.standardizer.standardize_pt(df, 'pt')

        return df

    def process_indications(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process indication data."""
        df = df.copy()

        # Standardize column names
        df = self.standardize_columns(df)

        # MedDRA standardization for indications
        if 'indi_pt' in df.columns:
            df = self.standardizer.standardize_pt(df, 'indi_pt')

        return df

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names using COLUMN_MAPPINGS."""
        df = df.copy()
        df.columns = df.columns.str.upper()
        return df.rename(columns=self.COLUMN_MAPPINGS)

    def process_file(self, file_path: Path, file_type: str) -> pd.DataFrame:
        """Process a single FAERS file based on type."""
        logging.info(f"Processing {file_type} file: {file_path.name}")
        df = pd.read_csv(file_path, sep='$', dtype=str)
        
        processors = {
            'DEMO': self.process_demographics,
            'DRUG': self.process_drugs,
            'REAC': self.process_reactions,
            'INDI': self.process_indications
        }
        
        if file_type not in processors:
            raise ValueError(f"Unknown file type: {file_type}")
        
        with tqdm(total=1, desc=f"Processing {file_type}") as pbar:
            result = processors[file_type](df)
            pbar.update(1)
        
        return result

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

    def standardize_drug_names(self, df: pd.DataFrame, drug_column: str) -> pd.DataFrame:
        """Standardize drug names using the standardizer's drug dictionary."""
        if drug_column in df.columns:
            df[drug_column] = df[drug_column].str.lower()
            # Get drug dictionary from standardizer
            drug_dict = self.standardizer.get_drug_dictionary()
            df[drug_column] = df[drug_column].map(lambda x: drug_dict.get(x, x))
            
        return df

    def process_drug_info(self, file_path: Path) -> pd.DataFrame:
        """Process drug information from FAERS file."""
        df = self.process_file(file_path, 'DRUG')
        
        # Convert date fields
        date_cols = ['exp_dt']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Standardize dechal and rechal values
        for col in ['dechal', 'rechal']:
            if col in df.columns:
                df[col] = df[col].str.lower()
        
        return df

    def process_indication_info(self, file_path: Path) -> pd.DataFrame:
        """Process indication information from FAERS file."""
        df = self.process_file(file_path, 'INDI')
        
        # Standardize indication PT
        df = self.standardizer.standardize_pt(df, 'indi_pt')
        
        return df

    def unify_data(self, files_list: List[str], name_key: Dict[str, str],
                   column_subset: List[str], duplicated_cols_x: List[str] = None,
                   duplicated_cols_y: List[str] = None) -> pd.DataFrame:
        """
        Unify data from multiple FAERS files with standardized column names and handling duplicates.
        
        Args:
            files_list: List of file paths to process
            name_key: Dictionary mapping original column names to standardized names
            column_subset: List of columns to keep in final dataset
            duplicated_cols_x: List of duplicate columns from first file
            duplicated_cols_y: List of duplicate columns from second file
        
        Returns:
            Unified DataFrame with standardized columns
        """
        unified_data = None
        
        with tqdm(total=len(files_list), desc="Unifying files") as pbar:
            for file_path in files_list:
                # Read the file
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:  # Assume text file with pipe delimiter
                    df = pd.read_csv(file_path, sep='$')
                
                # Rename columns using name_key
                df = df.rename(columns={v: k for k, v in name_key.items()})
                
                if unified_data is None:
                    unified_data = df
                else:
                    # Handle duplicate columns if specified
                    if duplicated_cols_x and duplicated_cols_y:
                        # Remove spaces from column names in y
                        df.columns = df.columns.str.strip()

                        # Merge based on common columns excluding duplicates
                        common_cols = list(set(unified_data.columns) & set(df.columns))
                        common_cols = [col for col in common_cols
                                    if col not in duplicated_cols_x
                                    and col not in duplicated_cols_y]
                        
                        unified_data = pd.merge(unified_data, df, on=common_cols, how='outer')

                        # Handle duplicate columns
                        for x_col, y_col in zip(duplicated_cols_x, duplicated_cols_y):
                            # Use coalesce logic: take value from x if not null, otherwise from y
                            unified_data[x_col] = unified_data[x_col].combine_first(unified_data[y_col])
                            unified_data = unified_data.drop(columns=[y_col])
                    else:
                        # Simple outer merge on all common columns
                        unified_data = pd.merge(unified_data, df, how='outer')
                
                pbar.update(1)
        
        # Subset columns if specified
        if column_subset:
            # Only keep columns that exist in the data
            valid_columns = [col for col in column_subset if col in unified_data.columns]
            unified_data = unified_data[valid_columns]
        
        return unified_data

    def process_demo_files(self, faers_files: List[str], output_path: str):
        """
        Process and unify DEMO files from FAERS data.
        
        Args:
            faers_files: List of all FAERS files
            output_path: Path to save the processed DEMO data
        """
        # Filter for DEMO files
        demo_files = [f for f in faers_files if 'demo' in f.lower()]

        # Define column mappings
        name_key = {
            "ISR": "primaryid",
            "CASE": "caseid",
            "FOLL_SEQ": "caseversion",
            "I_F_COD": "i_f_cod",
            "EVENT_DT": "event_dt",
            "MFR_DT": "mfr_dt",
            "FDA_DT": "fda_dt",
            "REPT_COD": "rept_cod",
            "MFR_NUM": "mfr_num",
            "MFR_SNDR": "mfr_sndr",
            "AGE": "age",
            "AGE_COD": "age_cod",
            "GNDR_COD": "sex",
            "E_SUB": "e_sub",
            "WT": "wt",
            "WT_COD": "wt_cod",
            "REPT_DT": "rept_dt",
            "OCCP_COD": "occp_cod",
            "TO_MFR": "to_mfr",
            "REPORTER_COUNTRY": "reporter_country",
            "quarter": "quarter",
            "i_f_code": "i_f_cod"
        }

        # Define columns to keep
        column_subset = [
            "primaryid", "caseid", "caseversion", "i_f_cod", "sex", "age",
            "age_cod", "age_grp", "wt", "wt_cod", "reporter_country",
            "occr_country", "event_dt", "rept_dt", "mfr_dt", "init_fda_dt",
            "fda_dt", "rept_cod", "occp_cod", "mfr_num", "mfr_sndr", "to_mfr",
            "e_sub", "quarter", "auth_num", "lit_ref"
        ]

        # Define duplicate columns
        duplicated_cols_x = ["rept_dt", "sex"]
        duplicated_cols_y = [" rept_dt", "gndr_cod"]

        # Process the data
        demo_data = self.unify_data(
            files_list=demo_files,
            name_key=name_key,
            column_subset=column_subset,
            duplicated_cols_x=duplicated_cols_x,
            duplicated_cols_y=duplicated_cols_y
        )

        # Save the processed data
        demo_data.to_pickle(output_path)

        return demo_data

    def process_indi_files(self, faers_files: List[str], output_path: str) -> pd.DataFrame:
        """
        Process and unify INDI (Indications) files from FAERS data.
        
        Args:
            faers_files: List of all FAERS files
            output_path: Path to save the processed INDI data
        
        Returns:
            DataFrame containing processed indication data
        """
        # Filter for INDI files
        indi_files = [f for f in faers_files if 'indi' in f.lower()]

        # Define column mappings
        name_key = {
            "ISR": "primaryid",
            "DRUG_SEQ": "drug_seq",
            "indi_drug_seq": "drug_seq",
            "INDI_PT": "indi_pt"
        }

        # Define columns to keep
        column_subset = ["primaryid", "drug_seq", "indi_pt"]

        # Process the data
        indi_data = self.unify_data(
            files_list=indi_files,
            name_key=name_key,
            column_subset=column_subset,
            duplicated_cols_x=None,  # No duplicate columns to handle
            duplicated_cols_y=None
        )

        # Remove rows with missing indication terms
        indi_data = indi_data.dropna(subset=['indi_pt'])

        # Save the processed data
        indi_data.to_pickle(output_path)

        return indi_data

    def process_outc_files(self, faers_files: List[str], output_path: str) -> pd.DataFrame:
        """
        Process and unify OUTC (Outcomes) files from FAERS data.
        
        Args:
            faers_files: List of all FAERS files
            output_path: Path to save the processed OUTC data
        
        Returns:
            DataFrame containing processed outcome data
        """
        # Filter for OUTC files
        outc_files = [f for f in faers_files if 'outc' in f.lower()]

        # Define column mappings
        name_key = {
            "ISR": "primaryid",
            "OUTC_COD": "outc_cod"
        }

        # Define columns to keep
        column_subset = ["primaryid", "outc_cod"]

        # Define duplicate columns (handling outc_cod and outc_code)
        duplicated_cols_x = ["outc_cod"]
        duplicated_cols_y = ["outc_code"]

        # Process the data
        outc_data = self.unify_data(
            files_list=outc_files,
            name_key=name_key,
            column_subset=column_subset,
            duplicated_cols_x=duplicated_cols_x,
            duplicated_cols_y=duplicated_cols_y
        )

        # Remove rows with missing outcome codes
        outc_data = outc_data.dropna(subset=['outc_cod'])

        # Save the processed data
        outc_data.to_pickle(output_path)

        return outc_data

    def process_reac_files(self, faers_files: List[str], output_path: str) -> pd.DataFrame:
        """
        Process and unify REAC (Reactions) files from FAERS data.
        
        Args:
            faers_files: List of all FAERS files
            output_path: Path to save the processed REAC data
        
        Returns:
            DataFrame containing processed reaction data
        """
        # Filter for REAC files
        reac_files = [f for f in faers_files if 'reac' in f.lower()]

        # Define column mappings
        name_key = {
            "ISR": "primaryid",
            "PT": "pt"
        }

        # Define columns to keep
        column_subset = ["primaryid", "pt", "drug_rec_act"]

        # Process the data
        reac_data = self.unify_data(
            files_list=reac_files,
            name_key=name_key,
            column_subset=column_subset,
            duplicated_cols_x=None,  # No duplicate columns to handle
            duplicated_cols_y=None
        )

        # Remove rows with missing reaction terms
        reac_data = reac_data.dropna(subset=['pt'])

        # Save the processed data
        reac_data.to_pickle(output_path)

        return reac_data

    def process_rpsr_files(self, faers_files: List[str], output_path: str) -> pd.DataFrame:
        """
        Process and unify RPSR (Report Source) files from FAERS data.
        
        Args:
            faers_files: List of all FAERS files
            output_path: Path to save the processed RPSR data
        
        Returns:
            DataFrame containing processed report source data
        """
        # Filter for RPSR files
        rpsr_files = [f for f in faers_files if 'rpsr' in f.lower()]

        # Define column mappings
        name_key = {
            "ISR": "primaryid",
            "RPSR_COD": "rpsr_cod"
        }

        # Define columns to keep
        column_subset = ["primaryid", "rpsr_cod"]

        # Process the data
        rpsr_data = self.unify_data(
            files_list=rpsr_files,
            name_key=name_key,
            column_subset=column_subset,
            duplicated_cols_x=None,  # No duplicate columns to handle
            duplicated_cols_y=None
        )

        # Save the processed data
        rpsr_data.to_pickle(output_path)

        return rpsr_data

    def process_ther_files(self, faers_files: List[str], output_path: str) -> pd.DataFrame:
        """
        Process and unify THER (Therapy) files from FAERS data.
        
        Args:
            faers_files: List of all FAERS files
            output_path: Path to save the processed THER data
        
        Returns:
            DataFrame containing processed therapy data
        """
        # Filter for THER files
        ther_files = [f for f in faers_files if 'ther' in f.lower()]

        # Define column mappings
        name_key = {
            "ISR": "primaryid",
            "dsg_drug_seq": "drug_seq",
            "DRUG_SEQ": "drug_seq",
            "START_DT": "start_dt",
            "END_DT": "end_dt",
            "DUR": "dur",
            "DUR_COD": "dur_cod"
        }

        # Define columns to keep
        column_subset = [
            "primaryid",
            "drug_seq",
            "start_dt",
            "end_dt",
            "dur",
            "dur_cod"
        ]

        # Process the data
        ther_data = self.unify_data(
            files_list=ther_files,
            name_key=name_key,
            column_subset=column_subset,
            duplicated_cols_x=None,  # No duplicate columns to handle
            duplicated_cols_y=None
        )

        # Save the processed data
        ther_data.to_pickle(output_path)

        return ther_data
