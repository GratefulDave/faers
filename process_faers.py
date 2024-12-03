#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path

from src.faers_processor.services.standardizer import DataStandardizer
from src.faers_processor.services.processor import FAERSProcessor
from src.faers_processor.cli import ProjectPaths, setup_logging, process_data, download_data, deduplicate_data

def main():
    parser = argparse.ArgumentParser(description='Process FAERS quarterly data files')
    
    parser.add_argument(
        '--project-root',
        type=Path,
        default=Path(__file__).parent,
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
            max_date=args.max_date,
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

if __name__ == "__main__":
    main()
