#!/usr/bin/env python3
"""
CLI interface for FAERS data processing.

This script provides a command-line interface for downloading, processing, and deduplicating 
FDA Adverse Event Reporting System (FAERS) data. It handles all quarters from 2004Q1 up to 
either a specified maximum date or the current quarter.

Directory Structure:
    project_root/
    ├── data/
    │   ├── clean/     # Processed and deduplicated files
    │   └── raw/       # Raw FAERS quarterly data
    └── external_data/
        ├── DiAna_dictionary/
        ├── manual_fixes/
        └── meddra/

Usage:
    # Process all quarters up to current date
    python process_faers.py --project-root .

    # Process all quarters up to specific date
    python process_faers.py --project-root . --max-date 20240930

    # Run specific steps with parallel processing
    python process_faers.py --project-root . --steps process --max-workers 8 --chunk-size 50000
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import re
from datetime import datetime

from faers_processor.services.standardizer import DataStandardizer
from faers_processor.services.downloader import FAERSDownloader
from faers_processor.services.deduplicator import FAERSDeduplicator
from faers_processor.services.processor import FAERSProcessor

class ProjectPaths:
    """
    Manages project directory structure and file paths.
    """
    def __init__(self, project_root: Path):
        self.root = project_root
        self.raw = project_root / 'data' / 'raw'
        self.clean = project_root / 'data' / 'clean'
        self.external = project_root / 'external_data'
        self.diana_dict = self.external / 'DiAna_dictionary'
        self.manual_fixes = self.external / 'manual_fixes'
        self.meddra = self.external / 'meddra' / 'MedAscii'
        
        # Ensure directories exist
        self.clean.mkdir(parents=True, exist_ok=True)

def setup_logging(level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def download_data(paths: ProjectPaths, max_date: Optional[str] = None) -> None:
    """Download all FAERS quarterly data."""
    downloader = FAERSDownloader(paths.raw)
    
    # Calculate quarters to download based on max_date
    if max_date:
        year = int(max_date[:4])
        month = int(max_date[4:6])
        quarter = (month - 1) // 3 + 1
        current_year = datetime.now().year
        
        quarters = []
        for y in range(2004, year + 1):
            if y == year:
                max_q = quarter
            else:
                max_q = 4
            for q in range(1, max_q + 1):
                quarters.append(f"{y}q{q}")
    else:
        # Download up to current quarter
        current_year = datetime.now().year
        current_quarter = (datetime.now().month - 1) // 3 + 1
        quarters = []
        for y in range(2004, current_year + 1):
            if y == current_year:
                max_q = current_quarter
            else:
                max_q = 4
            for q in range(1, max_q + 1):
                quarters.append(f"{y}q{q}")
    
    for quarter in quarters:
        try:
            logging.info(f"Downloading quarter {quarter}...")
            downloader.download_quarter(quarter)
        except Exception as e:
            logging.error(f"Error downloading quarter {quarter}: {str(e)}")

def process_data(paths: ProjectPaths, parallel: bool = True, 
                max_workers: Optional[int] = None,
                chunk_size: Optional[int] = None) -> None:
    """Process all FAERS data in parallel."""
    # Initialize standardizer and processor
    standardizer = DataStandardizer(paths.external, paths.clean)
    processor = FAERSProcessor(standardizer)
    
    # Process all available quarters
    quarters = []
    for quarter_dir in paths.raw.iterdir():
        if quarter_dir.is_dir() and re.match(r'\d{4}q[1-4]', quarter_dir.name.lower()):
            quarters.append(quarter_dir)
    
    logging.info(f"Found {len(quarters)} quarters to process: {[q.name for q in sorted(quarters)]}")
    
    for quarter_dir in sorted(quarters):
        try:
            logging.info(f"\n{'='*50}")
            logging.info(f"Processing quarter: {quarter_dir.name}")
            logging.info(f"Quarter directory: {quarter_dir}")
            # Let the processor find the ASCII directory case-insensitively
            processor.process_quarter(
                quarter_dir,
                parallel=parallel,
                max_workers=max_workers,
                chunk_size=chunk_size or 50000
            )
            logging.info(f"Finished processing quarter: {quarter_dir.name}")
        except Exception as e:
            logging.error(f"Failed processing quarter {quarter_dir.name}: {str(e)}")
            logging.error(f"Quarter directory that failed: {quarter_dir}")

def deduplicate_data(paths: ProjectPaths, max_date: Optional[str] = None,
                    parallel: bool = True, max_workers: Optional[int] = None,
                    chunk_size: Optional[int] = None) -> None:
    """Deduplicate all processed FAERS data in parallel."""
    try:
        logging.info("Starting deduplication of all processed data...")
        deduplicator = FAERSDeduplicator(paths.clean)
        deduplicator.deduplicate_all(
            max_date=max_date,
            parallel=parallel,
            max_workers=max_workers,
            chunk_size=chunk_size or 50000
        )
    except Exception as e:
        logging.error(f"Error during deduplication: {str(e)}")

def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description='Process FAERS quarterly data files')
    
    parser.add_argument(
        '--project-root',
        type=Path,
        required=True,
        help='Root directory of the project'
    )
    
    parser.add_argument(
        '--max-date',
        help='Maximum date to process (YYYYMMDD format)'
    )
    
    parser.add_argument(
        '--steps',
        nargs='+',
        choices=['download', 'process', 'deduplicate', 'all'],
        default=['all'],
        help='Processing steps to execute'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        default=True,
        help='Enable parallel processing (default: True)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        help='Number of worker processes for parallel processing (default: CPU count)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=50000,
        help='Size of data chunks for parallel processing (default: 50000)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    # Initialize project paths
    paths = ProjectPaths(args.project_root)
    
    # Execute requested steps
    steps = args.steps if 'all' not in args.steps else ['download', 'process', 'deduplicate']
    
    if 'download' in steps:
        logging.info("Starting download of all quarters...")
        download_data(paths, args.max_date)
    
    if 'process' in steps:
        logging.info("Starting processing of all quarters...")
        process_data(
            paths,
            parallel=args.parallel,
            max_workers=args.max_workers,
            chunk_size=args.chunk_size
        )
    
    if 'deduplicate' in steps:
        logging.info("Starting deduplication of all data...")
        deduplicate_data(
            paths,
            max_date=args.max_date,
            parallel=args.parallel,
            max_workers=args.max_workers,
            chunk_size=args.chunk_size
        )
    
    logging.info("All requested steps completed successfully")

if __name__ == '__main__':
    main()
