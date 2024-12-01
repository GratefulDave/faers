"""Main entry point for FAERS data processing pipeline."""
import argparse
import concurrent.futures
import logging
import multiprocessing
import platform
from pathlib import Path
import sys
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from .services.downloader import FAERSDownloader
from .services.standardizer import DataStandardizer
from .services.deduplicator import Deduplicator

# Check if running on Apple Silicon
IS_APPLE_SILICON = platform.processor() == 'arm'
if IS_APPLE_SILICON:
    # Enable Apple Silicon optimizations
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = str(multiprocessing.cpu_count())
    os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(multiprocessing.cpu_count())

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
        default=50000,
        help='Chunk size for parallel processing'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level'
    )
    
    return parser.parse_args()

def download_quarter(args: Dict[str, Any]) -> List[Path]:
    """Download a single FAERS quarter."""
    quarter = args['quarter']
    downloader = args['downloader']
    try:
        files = downloader.download_quarter(quarter)
        return files
    except Exception as e:
        logging.error(f"Error downloading quarter {quarter}: {str(e)}")
        return []

def download_data(data_dir: Path, max_workers: int) -> None:
    """Download latest FAERS quarterly data in parallel."""
    logging.info("Starting parallel FAERS data download")
    downloader = FAERSDownloader(
        raw_dir=data_dir / "raw",
        clean_dir=data_dir / "clean"
    )
    
    try:
        quarters = downloader.get_available_quarters()
        all_files = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_quarter = {
                executor.submit(
                    download_quarter,
                    {'quarter': quarter, 'downloader': downloader}
                ): quarter for quarter in quarters
            }
            
            with tqdm(total=len(quarters), desc="Downloading FAERS quarters") as pbar:
                for future in concurrent.futures.as_completed(future_to_quarter):
                    quarter = future_to_quarter[future]
                    try:
                        files = future.result()
                        all_files.extend(files)
                        pbar.update(1)
                        logging.info(f"Successfully downloaded quarter {quarter}")
                    except Exception as e:
                        logging.error(f"Failed to download quarter {quarter}: {str(e)}")
        
        logging.info(f"Successfully downloaded {len(all_files)} files")
    except Exception as e:
        logging.error(f"Error downloading FAERS data: {str(e)}")
        raise

def process_chunk(args: Dict[str, Any]) -> pd.DataFrame:
    """Process a chunk of FAERS data."""
    chunk = args['chunk']
    data_type = args['data_type']
    standardizer = args['standardizer']
    
    try:
        if data_type == 'demographics':
            return standardizer.process_demographics(chunk)
        elif data_type == 'drugs':
            return standardizer.process_drugs(chunk)
        else:  # reactions
            return standardizer.process_reactions(chunk)
    except Exception as e:
        logging.error(f"Error processing chunk: {str(e)}")
        return pd.DataFrame()

def process_file(args: Dict[str, Any]) -> Tuple[str, pd.DataFrame]:
    """Process a single FAERS file in parallel."""
    file_path = args['file_path']
    data_type = args['data_type']
    standardizer = args['standardizer']
    chunk_size = args['chunk_size']
    
    try:
        # Count total lines for progress bar
        total_lines = sum(1 for _ in open(file_path))
        total_chunks = (total_lines - 1) // chunk_size + 1
        
        # Read and process chunks in parallel
        chunks = []
        reader = pd.read_csv(file_path, chunksize=chunk_size, sep='$')
        
        chunk_args = [
            {
                'chunk': chunk,
                'data_type': data_type,
                'standardizer': standardizer
            }
            for chunk in reader
        ]
        
        # Process chunks with progress bar
        processed_chunks = process_map(
            process_chunk,
            chunk_args,
            max_workers=multiprocessing.cpu_count(),
            desc=f"Processing {data_type}",
            total=total_chunks,
            leave=False
        )
        
        # Combine processed chunks
        df = pd.concat(processed_chunks, ignore_index=True)
        return data_type, df
    
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {str(e)}")
        return data_type, pd.DataFrame()

def process_data(data_dir: Path, external_dir: Path, chunk_size: int) -> None:
    """Process downloaded FAERS data with parallel processing."""
    logging.info("Starting parallel FAERS data processing")
    standardizer = DataStandardizer(external_dir)
    
    try:
        # Get list of all data files to process
        data_files = {
            'demographics': data_dir / "raw" / "latest" / "DEMO.txt",
            'drugs': data_dir / "raw" / "latest" / "DRUG.txt",
            'reactions': data_dir / "raw" / "latest" / "REAC.txt"
        }
        
        # Prepare processing arguments
        process_args = [
            {
                'file_path': file_path,
                'data_type': data_type,
                'standardizer': standardizer,
                'chunk_size': chunk_size
            }
            for data_type, file_path in data_files.items()
            if file_path.exists()
        ]
        
        # Process files in parallel with progress bar
        with tqdm(total=len(process_args), desc="Processing FAERS files") as pbar:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                future_to_file = {
                    executor.submit(process_file, args): args['data_type']
                    for args in process_args
                }
                
                for future in concurrent.futures.as_completed(future_to_file):
                    data_type = future_to_file[future]
                    try:
                        data_type, df = future.result()
                        if not df.empty:
                            output_file = data_dir / "clean" / f"{data_type}.parquet"
                            df.to_parquet(
                                output_file,
                                engine='pyarrow',
                                compression='snappy',
                                use_threads=True
                            )
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
        # Load processed data with progress bars
        with tqdm(total=3, desc="Loading data for deduplication") as pbar:
            demo_df = pd.read_parquet(
                data_dir / "clean" / "demographics.parquet",
                engine='pyarrow',
                use_threads=True
            )
            pbar.update(1)
            
            drug_df = pd.read_parquet(
                data_dir / "clean" / "drugs.parquet",
                engine='pyarrow',
                use_threads=True
            )
            pbar.update(1)
            
            reac_df = pd.read_parquet(
                data_dir / "clean" / "reactions.parquet",
                engine='pyarrow',
                use_threads=True
            )
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
        demo_df.to_parquet(
            data_dir / "clean" / "demographics_dedup.parquet",
            engine='pyarrow',
            compression='snappy',
            use_threads=True
        )
        logging.info("Deduplication completed successfully")
    except Exception as e:
        logging.error(f"Error during deduplication: {str(e)}")
        raise

def main() -> None:
    """Main entry point for FAERS data processing pipeline."""
    args = parse_args()
    setup_logging(args.log_level)
    
    # Convert string paths to Path objects
    data_dir = Path(args.data_dir)
    external_dir = Path(args.external_dir)
    
    # Create necessary directories
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
                process_data(data_dir, external_dir, args.chunk_size)
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
