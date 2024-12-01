"""Main entry point for FAERS data processing pipeline."""
import argparse
import logging
from pathlib import Path
import sys
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from .services.downloader import FAERSDownloader
from .services.standardizer import DataStandardizer
from .services.deduplicator import Deduplicator

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
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level'
    )
    
    return parser.parse_args()

def download_data(data_dir: Path) -> None:
    """Download latest FAERS quarterly data."""
    logging.info("Starting FAERS data download")
    downloader = FAERSDownloader(
        raw_dir=data_dir / "raw",
        clean_dir=data_dir / "clean"
    )
    
    try:
        quarters = downloader.get_available_quarters()
        with tqdm(total=len(quarters), desc="Downloading FAERS quarters") as pbar:
            files = []
            for quarter in quarters:
                quarter_files = downloader.download_quarter(quarter)
                files.extend(quarter_files)
                pbar.update(1)
                
        logging.info(f"Successfully downloaded {len(files)} files")
    except Exception as e:
        logging.error(f"Error downloading FAERS data: {str(e)}")
        raise

def process_data(data_dir: Path, external_dir: Path) -> None:
    """Process downloaded FAERS data."""
    logging.info("Starting FAERS data processing")
    standardizer = DataStandardizer(external_dir)
    
    try:
        # Get list of all data files to process
        data_files = {
            'demographics': data_dir / "raw" / "latest" / "DEMO.txt",
            'drugs': data_dir / "raw" / "latest" / "DRUG.txt",
            'reactions': data_dir / "raw" / "latest" / "REAC.txt"
        }
        
        # Process each file type with progress bar
        with tqdm(total=len(data_files), desc="Processing FAERS files") as pbar:
            for data_type, file_path in data_files.items():
                if file_path.exists():
                    logging.info(f"Processing {data_type} data")
                    
                    # Read data with progress bar
                    total_lines = sum(1 for _ in open(file_path))
                    df = pd.DataFrame()
                    
                    for chunk in tqdm(
                        pd.read_csv(file_path, chunksize=10000, sep='$'),
                        total=total_lines // 10000 + 1,
                        desc=f"Reading {data_type}",
                        leave=False
                    ):
                        if data_type == 'demographics':
                            chunk = standardizer.process_demographics(chunk)
                        elif data_type == 'drugs':
                            chunk = standardizer.process_drugs(chunk)
                        else:  # reactions
                            chunk = standardizer.process_reactions(chunk)
                        
                        df = pd.concat([df, chunk], ignore_index=True)
                    
                    # Save processed data
                    output_file = data_dir / "clean" / f"{data_type}.parquet"
                    df.to_parquet(output_file)
                    pbar.update(1)
        
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
            demo_df = pd.read_parquet(data_dir / "clean" / "demographics.parquet")
            pbar.update(1)
            
            drug_df = pd.read_parquet(data_dir / "clean" / "drugs.parquet")
            pbar.update(1)
            
            reac_df = pd.read_parquet(data_dir / "clean" / "reactions.parquet")
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
        
        # Save deduplicated data
        demo_df.to_parquet(data_dir / "clean" / "demographics_dedup.parquet")
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
                download_data(data_dir)
                pbar.update(1)
            
            if args.process:
                process_data(data_dir, external_dir)
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
