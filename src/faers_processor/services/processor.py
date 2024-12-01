"""Service for processing FAERS data files."""
import logging
from pathlib import Path
from typing import Dict, List

import dask.dataframe as dd
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from .standardizer import DataStandardizer


class FAERSProcessor:
    """Processes FAERS data files."""

    def __init__(
            self,
            data_dir: Path,
            external_dir: Path,
            chunk_size: int = 100000,
            use_dask: bool = False
    ):
        """Initialize processor with configuration.
        
        Args:
            data_dir: Base directory containing raw data
            external_dir: Directory containing external reference data
            chunk_size: Size of data chunks for processing
            use_dask: Whether to use Dask for out-of-core processing
        """
        self.data_dir = data_dir / 'raw'
        self.output_dir = data_dir / 'clean'
        self.external_dir = external_dir
        self.chunk_size = chunk_size
        self.use_dask = use_dask

        # Initialize standardizer with external data
        self.standardizer = DataStandardizer(external_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_files(self, quarter: str) -> Dict[str, pd.DataFrame]:
        """Process all FAERS files for a given quarter.
        
        Args:
            quarter: Quarter identifier (e.g., '23Q1')
            
        Returns:
            Dictionary mapping file types to processed DataFrames
        """
        processed_data = {}

        # Get list of files to process
        files = list(self.data_dir.glob(f'*{quarter}*.txt'))
        if not files:
            logging.warning(f"No files found for quarter {quarter}")
            return processed_data

        with tqdm(total=len(files), desc="Processing files") as pbar:
            for file in files:
                try:
                    df = pd.read_csv(file, delimiter='$', dtype=str)
                    file_type = self._get_file_type(file.name)
                    processed_data[file_type] = df
                    pbar.update(1)
                except Exception as e:
                    logging.error(f"Error processing {file}: {str(e)}")

        return processed_data

    def unify_files(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Unify processed FAERS files into a single DataFrame.
        
        Args:
            data: Dictionary of processed DataFrames
            
        Returns:
            Unified DataFrame
        """
        if not data:
            logging.error("No data to unify")
            return pd.DataFrame()

        try:
            # Start with demographics
            demo = data.get('DEMO', pd.DataFrame())
            if demo.empty:
                logging.error("No demographics data found")
                return pd.DataFrame()

            # Process each file type
            with tqdm(total=len(data), desc="Unifying files") as pbar:
                for file_type, df in data.items():
                    if file_type != 'DEMO':
                        demo = self._merge_dataframe(demo, df, file_type)
                    pbar.update(1)

            return demo

        except Exception as e:
            logging.error(f"Error unifying files: {str(e)}")
            return pd.DataFrame()

    def process_quarter(self, quarter: str) -> pd.DataFrame:
        """Process all FAERS files for a quarter.
        
        Args:
            quarter: Quarter identifier (e.g., '23Q1')
            
        Returns:
            Processed DataFrame
        """
        try:
            # Process individual files
            processed_data = self.process_files(quarter)
            if not processed_data:
                return pd.DataFrame()

            # Unify files
            unified_data = self.unify_files(processed_data)
            if unified_data.empty:
                return pd.DataFrame()

            # Generate summary (for logging only)
            self._generate_summary(unified_data)

            # Save output
            output_path = self.output_dir / f"faers_{quarter}_processed.parquet"
            unified_data.to_parquet(output_path)

            return unified_data

        except Exception as e:
            logging.error(f"Error processing quarter {quarter}: {str(e)}")
            return pd.DataFrame()

    def _generate_summary(self, data: pd.DataFrame) -> None:
        """Generate and log summary statistics.
        
        Args:
            data: Input DataFrame
        """
        if data.empty:
            logging.warning("No data for summary generation")
            return

        try:
            # Calculate summary statistics
            summary = {
                'Total Records': len(data),
                'Unique Cases': data['caseid'].nunique(),
                'Date Range': f"{data['fda_dt'].min()} to {data['fda_dt'].max()}",
                'Missing Values (%)': (data.isna().sum() / len(data) * 100).round(2).to_dict()
            }

            # Log summary statistics
            for key, value in summary.items():
                if key != 'Missing Values (%)':
                    logging.info(f"{key}: {value}")
                else:
                    logging.info("Missing Values (%):")
                    for col, pct in value.items():
                        logging.info(f"  {col}: {pct}%")

            # Create and save summary plot
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                data['fda_dt'].value_counts().sort_index().plot(ax=ax)
                ax.set_title('Reports by FDA Receipt Date')
                ax.set_xlabel('Date')
                ax.set_ylabel('Number of Reports')

                plot_path = self.output_dir / "reports_by_date.png"
                fig.savefig(plot_path)
                plt.close(fig)
                logging.info(f"Summary plot saved to {plot_path}")
            except Exception as plot_error:
                logging.warning(f"Could not generate summary plot: {str(plot_error)}")

        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")

    @staticmethod
    def validate_data(data: pd.DataFrame) -> bool:
        """Validate processed data.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Check required columns
            required_cols = ['primaryid', 'caseid', 'fda_dt']
            if not all(col in data.columns for col in required_cols):
                logging.error("Missing required columns")
                return False

            # Check for empty DataFrame
            if data.empty:
                logging.error("Empty DataFrame")
                return False

            # Check for missing values in key columns
            missing_key_vals = data[required_cols].isna().sum()
            if missing_key_vals.any():
                logging.warning(f"Missing values in key columns:\n{missing_key_vals}")

            # Check date format
            try:
                pd.to_datetime(data['fda_dt'])
            except Exception as e:
                logging.error(f"Invalid date format in fda_dt: {str(e)}")
                return False

            return True

        except Exception as e:
            logging.error(f"Error validating data: {str(e)}")
            return False

    @staticmethod
    def _get_file_type(filename: str) -> str:
        """Extract file type from filename."""
        file_types = {
            'DEMO': ['demo', 'demographic'],
            'DRUG': ['drug'],
            'REAC': ['reac', 'reaction'],
            'OUTC': ['outc', 'outcome'],
            'RPSR': ['rpsr', 'source'],
            'THER': ['ther', 'therapy']
        }

        filename = filename.lower()
        for file_type, patterns in file_types.items():
            if any(pattern in filename for pattern in patterns):
                return file_type
        return 'UNKNOWN'

    def _merge_dataframe(self, base_df: pd.DataFrame, merge_df: pd.DataFrame, file_type: str) -> pd.DataFrame:
        """Merge a DataFrame with the base DataFrame."""
        try:
            # Get merge columns based on file type
            merge_cols = self._get_merge_columns(file_type)
            if not merge_cols:
                logging.warning(f"No merge columns defined for {file_type}")
                return base_df

            # Perform merge
            merged = pd.merge(
                base_df,
                merge_df,
                on=merge_cols,
                how='left',
                suffixes=('', f'_{file_type.lower()}')
            )

            return merged

        except Exception as e:
            logging.error(f"Error merging {file_type}: {str(e)}")
            return base_df

    @staticmethod
    def _get_merge_columns(file_type: str) -> List[str]:
        """Get merge columns for a file type."""
        merge_columns = {
            'DRUG': ['primaryid', 'caseid'],
            'REAC': ['primaryid', 'caseid'],
            'OUTC': ['primaryid', 'caseid'],
            'RPSR': ['primaryid', 'caseid'],
            'THER': ['primaryid', 'caseid', 'drug_seq']
        }
        return merge_columns.get(file_type, [])

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

    @staticmethod
    def unify_data(files_list: List[str], name_key: Dict[str, str],
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

    def process_all(self) -> None:
        """Process all FAERS data files."""
        try:
            # Get list of all quarters
            quarters = [d.name for d in self.data_dir.iterdir() if d.is_dir()]

            if not quarters:
                logging.warning("No quarters found to process")
                return

            logging.info(f"Found {len(quarters)} quarters to process")

            # Process each quarter
            with tqdm(total=len(quarters), desc="Processing quarters") as pbar:
                for quarter in quarters:
                    try:
                        # Get files for this quarter
                        demo_file = next(self.data_dir.glob(f"{quarter}/*DEMO*.txt"), None)
                        drug_file = next(self.data_dir.glob(f"{quarter}/*DRUG*.txt"), None)
                        reac_file = next(self.data_dir.glob(f"{quarter}/*REAC*.txt"), None)

                        if not all([demo_file, drug_file, reac_file]):
                            logging.warning(f"Missing files for quarter {quarter}")
                            continue

                        # Process demographics
                        demo_df = self.process_file(demo_file, 'demographics')
                        if not demo_df.empty:
                            save_path = self.output_dir / f"{quarter}_demographics.parquet"
                            demo_df.to_parquet(save_path, engine='pyarrow', index=False)

                        # Process drugs
                        drug_df = self.process_file(drug_file, 'drugs')
                        if not drug_df.empty:
                            save_path = self.output_dir / f"{quarter}_drugs.parquet"
                            drug_df.to_parquet(save_path, engine='pyarrow', index=False)

                        # Process reactions
                        reac_df = self.process_file(reac_file, 'reactions')
                        if not reac_df.empty:
                            save_path = self.output_dir / f"{quarter}_reactions.parquet"
                            reac_df.to_parquet(save_path, engine='pyarrow', index=False)

                        logging.info(f"Successfully processed quarter {quarter}")

                    except Exception as e:
                        logging.error(f"Error processing quarter {quarter}: {str(e)}")
                    finally:
                        pbar.update(1)

        except Exception as e:
            logging.error(f"Error in process_all: {str(e)}")
            raise

    def process_file(self, file_path: Path, data_type: str) -> pd.DataFrame:
        """Process a single FAERS file.
        
        Args:
            file_path: Path to the file
            data_type: Type of data ('demographics', 'drugs', 'reactions')
        
        Returns:
            Processed DataFrame
        """
        try:
            # Read data with optimized settings
            if self.use_dask:
                df = dd.read_csv(
                    file_path,
                    sep='$',
                    dtype=str,
                    na_values=['', 'NA', 'NULL'],
                    keep_default_na=True,
                    blocksize=self.chunk_size * 1024
                )
            else:
                df = pd.read_csv(
                    file_path,
                    sep='$',
                    dtype=str,
                    na_values=['', 'NA', 'NULL'],
                    keep_default_na=True,
                    chunksize=self.chunk_size
                )

            # Process based on data type
            if data_type == 'demographics':
                result = self.standardizer.process_demographics(df)
            elif data_type == 'drugs':
                result = self.standardizer.process_drugs(df)
            else:  # reactions
                result = self.standardizer.process_reactions(df)

            # Compute if using Dask
            if self.use_dask:
                result = result.compute()

            return result

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            return pd.DataFrame()
