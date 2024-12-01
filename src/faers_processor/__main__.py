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
from typing import Dict, Any, Tuple

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import vaex
from tqdm import tqdm

from .services.deduplicator import Deduplicator
from .services.downloader import FAERSDownloader
from .services.standardizer import DataStandardizer

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
    parser = argparse.ArgumentParser(description='FAERS Data Processing Pipeline')

    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Base directory for data storage'
    )

    parser.add_argument(
        '--external-dir',
        type=str,
        required=True,
        help='Directory containing external reference data'
    )

    parser.add_argument(
        '--download',
        action='store_true',
        help='Download latest FAERS quarterly data'
    )

    parser.add_argument(
        '--process',
        action='store_true',
        help='Process downloaded FAERS data'
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
        '--use-vaex',
        action='store_true',
        help='Use Vaex for memory-efficient processing'
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


def process_chunk_dask(args: Dict[str, Any]) -> dd.DataFrame:
    """Process a chunk of FAERS data using Dask."""
    chunk = args['chunk']
    data_type = args['data_type']
    standardizer = args['standardizer']

    try:
        if data_type == 'demographics':
            result = standardizer.process_demographics(chunk)
        elif data_type == 'drugs':
            result = standardizer.process_drugs(chunk)
        else:  # reactions
            result = standardizer.process_reactions(chunk)

        # Convert to Dask DataFrame
        return dd.from_pandas(result, npartitions=1)
    except Exception as e:
        logging.error(f"Error processing chunk: {str(e)}")
        return dd.from_pandas(pd.DataFrame(), npartitions=1)


def process_file_optimized(args: Dict[str, Any]) -> Tuple[str, pd.DataFrame]:
    """Process a single FAERS file with optimized methods."""
    file_path = args['file_path']
    data_type = args['data_type']
    standardizer = args['standardizer']
    chunk_size = args['chunk_size']
    use_dask = args.get('use_dask', False)
    use_vaex = args.get('use_vaex', False)

    try:
        if use_vaex:
            # Use Vaex for memory-efficient processing
            df = vaex.from_csv(
                file_path,
                sep='$',
                chunk_size=chunk_size,
                convert=True
            )
            # Process with Vaex
            if data_type == 'demographics':
                df = standardizer.process_demographics_vaex(df)
            elif data_type == 'drugs':
                df = standardizer.process_drugs_vaex(df)
            else:
                df = standardizer.process_reactions_vaex(df)
            return data_type, df.to_pandas_df()

        elif use_dask:
            # Use Dask for out-of-core processing
            ddf = dd.read_csv(
                file_path,
                sep='$',
                blocksize=chunk_size * 1024,  # Convert to bytes
                dtype_backend='pyarrow'
            )

            # Process with Dask
            meta = pd.DataFrame()  # Define output schema
            result = ddf.map_partitions(
                process_chunk_dask,
                args={'data_type': data_type, 'standardizer': standardizer},
                meta=meta
            )

            # Compute final result
            df = result.compute()
            return data_type, optimize_dtypes(df)

        else:
            # Use memory-mapped reading for standard processing
            chunks = []
            total_lines = sum(1 for _ in open(file_path))
            total_chunks = (total_lines - 1) // chunk_size + 1

            with pd.read_csv(
                    file_path,
                    sep='$',
                    chunksize=chunk_size,
                    memory_map=True,
                    low_memory=True,
                    dtype_backend='pyarrow'
            ) as reader:
                for chunk in tqdm(
                        reader,
                        total=total_chunks,
                        desc=f"Processing {data_type}",
                        leave=False
                ):
                    processed = process_chunk({'chunk': chunk, 'data_type': data_type, 'standardizer': standardizer})
                    chunks.append(processed)

            df = pd.concat(chunks, ignore_index=True)
            return data_type, optimize_dtypes(df)

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {str(e)}")
        return data_type, pd.DataFrame()


def save_optimized_parquet(df: pd.DataFrame, output_file: Path) -> None:
    """Save DataFrame to parquet with optimized settings."""
    # Create PyArrow table with optimized schema
    table = pa.Table.from_pandas(df, preserve_index=False)

    # Write with optimized settings
    pq.write_table(
        table,
        output_file,
        compression='snappy',
        use_dictionary=True,
        dictionary_pagesize_limit=1048576,  # 1MB
        data_page_size=1048576,  # 1MB
        write_statistics=True,
        use_byte_stream_split=True,
        use_threads=True
    )


def download_data(data_dir: Path, max_workers: int) -> None:
    """Download FAERS quarterly data files."""
    logging.info("Starting FAERS data download")
    downloader = FAERSDownloader()

    try:
        # Get list of quarters to download
        quarters = downloader.get_available_quarters()
        logging.info(f"Found {len(quarters)} quarters available for download")

        # Create download directory if it doesn't exist
        download_dir = data_dir / "raw"
        download_dir.mkdir(parents=True, exist_ok=True)

        # Download files in parallel with progress tracking
        with tqdm(total=len(quarters), desc="Downloading quarters") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit download tasks
                future_to_quarter = {
                    executor.submit(
                        downloader.download_quarter,
                        quarter,
                        download_dir
                    ): quarter for quarter in quarters
                }

                # Process completed downloads
                for future in concurrent.futures.as_completed(future_to_quarter):
                    quarter = future_to_quarter[future]
                    try:
                        future.result()
                        logging.info(f"Successfully downloaded quarter {quarter}")
                    except Exception as e:
                        logging.error(f"Failed to download quarter {quarter}: {str(e)}")
                    finally:
                        pbar.update(1)

        logging.info("FAERS data download completed")
    except Exception as e:
        logging.error(f"Error downloading FAERS data: {str(e)}")
        raise


def process_data(
        data_dir: Path,
        external_dir: Path,
        chunk_size: int,
        use_dask: bool = False,
        use_vaex: bool = False
) -> None:
    """Process downloaded FAERS data with optimized parallel processing."""
    logging.info("Starting optimized FAERS data processing")
    standardizer = DataStandardizer(external_dir)

    try:
        data_files = {
            'demographics': data_dir / "raw" / "latest" / "DEMO.txt",
            'drugs': data_dir / "raw" / "latest" / "DRUG.txt",
            'reactions': data_dir / "raw" / "latest" / "REAC.txt"
        }

        process_args = [
            {
                'file_path': file_path,
                'data_type': data_type,
                'standardizer': standardizer,
                'chunk_size': chunk_size,
                'use_dask': use_dask,
                'use_vaex': use_vaex
            }
            for data_type, file_path in data_files.items()
            if file_path.exists()
        ]

        with tqdm(total=len(process_args), desc="Processing FAERS files") as pbar:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                future_to_file = {
                    executor.submit(process_file_optimized, args): args['data_type']
                    for args in process_args
                }

                for future in concurrent.futures.as_completed(future_to_file):
                    data_type = future_to_file[future]
                    try:
                        data_type, df = future.result()
                        if not df.empty:
                            output_file = data_dir / "clean" / f"{data_type}.parquet"
                            save_optimized_parquet(df, output_file)
                            logging.info(f"Successfully processed {data_type} data")
                        pbar.update(1)
                    except Exception as e:
                        logging.error(f"Failed to process {data_type}: {str(e)}")

        logging.info("Data processing completed successfully")
    except Exception as e:
        logging.error(f"Error processing FAERS data: {str(e)}")
        raise


def deduplicate_data(data_dir: Path) -> None:
    """Perform deduplication on processed data."""
    logging.info("Starting data deduplication")
    deduplicator = Deduplicator()

    try:
        # Load processed data with progress bars and optimized settings
        with tqdm(total=3, desc="Loading data for deduplication") as pbar:
            demo_df = pd.read_parquet(
                data_dir / "clean" / "demographics.parquet",
                engine='pyarrow',
                use_threads=True,
                memory_map=True
            )
            demo_df = optimize_dtypes(demo_df)
            pbar.update(1)

            drug_df = pd.read_parquet(
                data_dir / "clean" / "drugs.parquet",
                engine='pyarrow',
                use_threads=True,
                memory_map=True
            )
            drug_df = optimize_dtypes(drug_df)
            pbar.update(1)

            reac_df = pd.read_parquet(
                data_dir / "clean" / "reactions.parquet",
                engine='pyarrow',
                use_threads=True,
                memory_map=True
            )
            reac_df = optimize_dtypes(reac_df)
            pbar.update(1)

        # Perform rule-based deduplication with progress
        logging.info("Performing rule-based deduplication")
        total_cases = len(demo_df)

        with tqdm(total=total_cases, desc="Deduplicating cases") as pbar:
            demo_df = deduplicator.rule_based_deduplication(
                demo_df=demo_df,
                drug_df=drug_df,
                reac_df=reac_df,
                progress_callback=lambda x: pbar.update(x)
            )

        # Save deduplicated data with optimized settings
        save_optimized_parquet(
            optimize_dtypes(demo_df),
            data_dir / "clean" / "demographics_dedup.parquet"
        )
        logging.info("Deduplication completed successfully")
    except Exception as e:
        logging.error(f"Error during deduplication: {str(e)}")
        raise


def main() -> None:
    """Main entry point for FAERS data processing pipeline."""
    args = parse_args()
    setup_logging(args.log_level)

    data_dir = Path(args.data_dir)
    external_dir = Path(args.external_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "raw").mkdir(exist_ok=True)
    (data_dir / "clean").mkdir(exist_ok=True)

    try:
        with tqdm(
                total=sum([args.download, args.process, args.deduplicate]),
                desc="Overall progress"
        ) as pbar:
            if args.download:
                download_data(data_dir, args.max_workers)
                pbar.update(1)

            if args.process:
                process_data(
                    data_dir,
                    external_dir,
                    args.chunk_size,
                    args.use_dask,
                    args.use_vaex
                )
                pbar.update(1)

            if args.deduplicate:
                deduplicate_data(data_dir)
                pbar.update(1)

        logging.info("Pipeline completed successfully")
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
