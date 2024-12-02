"""Main entry point for FAERS data processing pipeline."""

import argparse
import concurrent.futures
import logging
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Dict, Any

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from .services.deduplicator import Deduplicator
from .services.downloader import FAERSDownloader
from .services.processor import FAERSProcessor
from .services.standardizer import DataStandardizer

# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable Apple Silicon optimizations
logging.info("Configuring optimizations for Apple Silicon")
num_cpus = multiprocessing.cpu_count()
os.environ['OPENBLAS_NUM_THREADS'] = str(num_cpus)
os.environ['MKL_NUM_THREADS'] = str(num_cpus)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_cpus)
logging.info(f"Set thread count to {num_cpus} for optimized libraries")

# Configure pandas for better performance
pd.set_option('compute.use_numexpr', True)
pd.set_option('mode.chained_assignment', None)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging with specified level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage by choosing appropriate dtypes."""
    if df.empty:
        return df
        
    df = df.copy()
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        col_min = df[col].min()
        col_max = df[col].max()
        
        if pd.api.types.is_integer_dtype(df[col]):
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype('int8')
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype('int16')
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype('int32')
    
    # Optimize string columns
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # If column has low cardinality
            df[col] = df[col].astype('category')
    
    return df


def process_chunk(args: Dict[str, Any]) -> pd.DataFrame:
    """Process a chunk of FAERS data.
    
    Args:
        args: Dictionary containing:
            - chunk: DataFrame chunk to process
            - data_type: Type of data ('demographics', 'drugs', 'reactions')
            - standardizer: DataStandardizer instance
            - use_dask: Whether to use Dask for processing
    
    Returns:
        Processed DataFrame chunk
    """
    chunk = args['chunk']
    data_type = args['data_type']
    standardizer = args['standardizer']
    use_dask = args.get('use_dask', False)

    try:
        if use_dask:
            # Convert to Dask DataFrame for parallel processing
            chunk = dd.from_pandas(chunk, npartitions=multiprocessing.cpu_count())
            if data_type == 'demographics':
                result = standardizer.process_demographics(chunk)
            elif data_type == 'drugs':
                result = standardizer.process_drugs(chunk)
            else:  # reactions
                result = standardizer.process_reactions(chunk)
            return optimize_dtypes(result.compute())
        else:
            # Process with standard pandas
            if data_type == 'demographics':
                result = standardizer.process_demographics(chunk)
            elif data_type == 'drugs':
                result = standardizer.process_drugs(chunk)
            else:  # reactions
                result = standardizer.process_reactions(chunk)
            return optimize_dtypes(result)

    except Exception as e:
        logging.error(f"Error processing {data_type} chunk: {str(e)}")
        return pd.DataFrame()


def process_file_optimized(args: Dict[str, Any]) -> pd.DataFrame:
    """Process a single FAERS file with optimized methods.
    
    Args:
        args: Dictionary containing processing parameters
    
    Returns:
        Processed DataFrame
    """
    file_path = args['file_path']
    data_type = args['data_type']
    standardizer = args['standardizer']
    use_dask = args.get('use_dask', False)
    chunk_size = args.get('chunk_size', 100000)

    # Common read options
    read_opts = {
        'filepath_or_buffer': str(file_path),
        'sep': '$',
        'dtype': str,
        'na_values': ['', 'NA', 'NULL'],
        'keep_default_na': True
    }

    try:
        # Read data with optimized settings
        if use_dask:
            df = dd.read_csv(
                **read_opts,
                blocksize=chunk_size * 1024
            )
        else:
            chunks = pd.read_csv(
                **read_opts,
                chunksize=chunk_size
            )
            # Concatenate chunks into single DataFrame
            df = pd.concat(chunks, ignore_index=True)

        # Process based on data type
        if data_type == 'demographics':
            result = standardizer.process_demographics(df)
        elif data_type == 'drugs':
            result = standardizer.process_drugs(df)
        else:  # reactions
            result = standardizer.process_reactions(df)

        # Compute if using Dask
        if use_dask:
            result = result.compute()

        return result

    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        raise


def save_optimized_parquet(df: pd.DataFrame, output_file: Path) -> None:
    """Save DataFrame to parquet with optimized settings.
    
    Args:
        df: DataFrame to save
        output_file: Path to save file to
    """
    try:
        # Create parent directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with optimized settings
        df.to_parquet(
            output_file,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
    except Exception as e:
        logging.error(f"Error saving parquet file: {str(e)}")


def download_data(data_dir: Path, max_workers: int) -> None:
    """Download FAERS quarterly data files."""
    try:
        # Create data directories if they don't exist
        raw_dir = data_dir
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        downloader = FAERSDownloader(raw_dir)
        downloader.download_all_quarters(max_workers=max_workers)
    except Exception as e:
        logging.error(f"Error in download process: {str(e)}")
        raise


def process_data(
    data_dir: Path,
    external_dir: Path,
    chunk_size: int,
    use_dask: bool = False
) -> None:
    """Process downloaded FAERS data with optimized parallel processing.
    
    Args:
        data_dir: Path to raw data directory
        external_dir: Path to external data directory
        chunk_size: Number of rows to process at once
        use_dask: Whether to use Dask for parallel processing
    """
    try:
        # Create clean data directory
        clean_dir = data_dir.parent / 'clean'
        clean_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Processing data from: {data_dir}")
        logging.info(f"Using external data from: {external_dir}")
        logging.info(f"Saving processed data to: {clean_dir}")
        
        # Initialize processor with standardizer
        standardizer = DataStandardizer(external_dir)
        processor = FAERSProcessor(standardizer)
        
        # Process all quarters
        processor.process_all(
            input_dir=data_dir / 'Raw_FAERS_QD',  # Raw data is in Raw_FAERS_QD subdirectory
            output_dir=clean_dir / 'Clean Data'    # Clean data goes in Clean Data subdirectory
        )
        
        logging.info("Data processing completed successfully")
        
    except Exception as e:
        logging.error(f"Error in data processing: {str(e)}")
        raise


def deduplicate_data(data_dir: Path) -> None:
    """Perform deduplication on processed data."""
    try:
        clean_dir = data_dir / 'clean'
        deduplicator = Deduplicator(clean_dir)
        deduplicator.deduplicate_all()
    except Exception as e:
        logging.error(f"Error during deduplication: {str(e)}")
        raise


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process FAERS data with optimizations for Apple Silicon'
    )

    parser.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help='Base directory for data storage'
    )

    parser.add_argument(
        '--external-dir',
        type=Path,
        required=True,
        help='Directory for external reference data'
    )

    parser.add_argument(
        '--download',
        action='store_true',
        help='Download latest FAERS data'
    )

    parser.add_argument(
        '--process',
        action='store_true',
        help='Process downloaded data'
    )

    parser.add_argument(
        '--deduplicate',
        action='store_true',
        help='Remove duplicate entries'
    )

    parser.add_argument(
        '--max-workers',
        type=int,
        default=multiprocessing.cpu_count(),
        help='Maximum number of parallel workers'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100000,
        help='Size of data chunks for processing'
    )

    parser.add_argument(
        '--use-dask',
        action='store_true',
        help='Use Dask for out-of-core processing'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level'
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for FAERS data processing pipeline."""
    args = parse_args()
    setup_logging(args.log_level)

    try:
        if args.download:
            download_data(
                data_dir=args.data_dir,
                max_workers=args.max_workers
            )

        if args.process:
            process_data(
                data_dir=args.data_dir,
                external_dir=args.external_dir,
                chunk_size=args.chunk_size,
                use_dask=args.use_dask
            )

        if args.deduplicate:
            deduplicate_data(args.data_dir)

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
