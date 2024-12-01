"""Main entry point for FAERS data processing pipeline.

This module provides a command-line interface for downloading, processing, and
deduplicating FDA Adverse Event Reporting System (FAERS) data.

Usage:
    Basic download and process:
        python -m faers_processor \
            --data-dir /path/to/data \
            --external-dir /path/to/external \
            --download \
            --process

    Full pipeline with optimizations:
        python -m faers_processor \
            --data-dir /path/to/data \
            --external-dir /path/to/external \
            --download \
            --process \
            --deduplicate \
            --use-dask \
            --max-workers 8 \
            --chunk-size 200000

    Process existing data with Vaex:
        python -m faers_processor \
            --data-dir /path/to/data \
            --external-dir /path/to/external \
            --process \
            --use-vaex

Options:
    --data-dir       Base directory for data storage (required)
    --external-dir   Directory for external reference data (required)
    --download       Download latest FAERS quarterly data
    --process        Process downloaded data
    --deduplicate    Remove duplicate entries
    --max-workers    Number of parallel workers (default: CPU count)
    --chunk-size     Size of data chunks for processing (default: 100000)
    --use-dask       Enable Dask for out-of-core processing
    --use-vaex       Enable Vaex for memory-efficient processing
    --log-level      Set logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL)

Directory Structure:
    data/
        raw/            # Downloaded FAERS quarterly files
        clean/          # Processed and standardized data
            demographics.parquet
            drugs.parquet
            reactions.parquet
            demographics_dedup.parquet  # After deduplication
    external/          # External reference data (e.g., drug mappings)

Performance Tips:
    1. Use --use-dask for processing large datasets that don't fit in memory
    2. Use --use-vaex for memory-efficient processing of medium-sized datasets
    3. Adjust --chunk-size based on available RAM
    4. Set --max-workers to match your CPU core count
    5. For Apple Silicon Macs, optimizations are automatically applied
"""
import argparse
import concurrent.futures
import logging
import multiprocessing
import platform
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

# Check if running on Apple Silicon
IS_APPLE_SILICON = platform.processor() == 'arm'
if IS_APPLE_SILICON:
    # Enable Apple Silicon optimizations
    import os

    os.environ['OPENBLAS_NUM_THREADS'] = str(multiprocessing.cpu_count())
    os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(multiprocessing.cpu_count())
    os.environ['NUMEXPR_MAX_THREADS'] = str(multiprocessing.cpu_count())
    os.environ['NUMEXPR_NUM_THREADS'] = str(multiprocessing.cpu_count())

    # Enable memory-mapped temp files for better memory management
    os.environ['TMPDIR'] = '/tmp'

# Configure pandas for better performance
pd.set_option('compute.use_numexpr', True)
pd.set_option('mode.chained_assignment', None)


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('faers_processor.log')
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process FAERS data with optimizations for Apple Silicon')

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
        help='Download latest FAERS quarterly data'
    )

    parser.add_argument(
        '--process',
        action='store_true',
        help='Process downloaded data'
    )

    parser.add_argument(
        '--deduplicate',
        action='store_true',
        help='Perform deduplication on processed data'
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
        help='Chunk size for parallel processing'
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


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage by choosing appropriate dtypes."""
    for col in df.columns:
        if df[col].dtype == 'object':
            # Convert string columns to categorical if they have low cardinality
            num_unique = df[col].nunique()
            if num_unique / len(df) < 0.5:  # If less than 50% unique values
                df[col] = df[col].astype('category')
        elif df[col].dtype == 'float64':
            # Downcast float64 to float32 if possible
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            # Downcast int64 to smallest possible integer type
            df[col] = pd.to_numeric(df[col], downcast='integer')
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
    data_type = args['data_type']
    standardizer = args['standardizer']
    use_dask = args.get('use_dask', False)

    try:
        if use_dask:
            # Convert to Dask DataFrame for parallel processing
            chunk = standardizer._to_dask_df(chunk)
            if data_type == 'demographics':
                result = standardizer.process_demographics(chunk)
            elif data_type == 'drugs':
                result = standardizer.process_drugs(chunk)
            else:  # reactions
                result = standardizer.process_reactions(chunk)
            return optimize_dtypes(result)
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

    try:
        read_args = {
            'filepath_or_buffer': file_path,
            'sep': '$',
            'dtype': str,
            'na_values': ['', 'NA', 'NULL'],
            'keep_default_na': True,
            'on_bad_lines': 'skip'
        }

        if use_dask:
            df = dd.read_csv(**read_args)
        else:
            df = pd.read_csv(**read_args)

        return process_chunk({
            'chunk': df,
            'data_type': data_type,
            'standardizer': standardizer,
            'use_dask': use_dask
        })

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {str(e)}")
        return pd.DataFrame()


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
        raw_dir = data_dir / 'raw'
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
    """Process downloaded FAERS data with optimized parallel processing."""
    try:
        # Create clean data directory if it doesn't exist
        clean_dir = data_dir / 'clean'
        clean_dir.mkdir(parents=True, exist_ok=True)
        
        processor = FAERSProcessor(
            data_dir=data_dir,
            external_dir=external_dir,
            chunk_size=chunk_size,
            use_dask=use_dask
        )
        processor.process_all()
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
