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

    # Run specific steps
    python process_faers.py --project-root . --steps download
    python process_faers.py --project-root . --steps process
    python process_faers.py --project-root . --steps deduplicate

    # Enable parallel processing
    python process_faers.py --project-root . --parallel --workers 4

Steps:
    1. download: Downloads all FAERS quarterly data files
    2. process: Standardizes and processes all downloaded quarters
    3. deduplicate: Removes duplicates across all processed data

All steps process quarters from 2004Q1 up to either:
    - The quarter specified by max-date (YYYYMMDD format)
    - The current quarter if no max-date is specified
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

class ProjectPaths:
    """
    Manages project directory structure and file paths.
    
    This class handles all path-related operations including:
    - Directory creation and validation
    - Quarter-specific path generation
    - Case-insensitive file finding across all quarters
    
    Attributes:
        root (Path): Project root directory
        raw (Path): Directory for raw FAERS data
        clean (Path): Directory for processed data
        external (Path): Directory for external reference data
        diana_dict (Path): Directory for DiAna dictionary
        manual_fixes (Path): Directory for manual data fixes
        meddra (Path): Directory for MedDRA vocabulary files
    """
    
    def __init__(self, project_root: Path):
        """
        Initialize project directory structure.
        
        Args:
            project_root (Path): Root directory of the project
        """
        self.root = project_root
        self.raw = project_root / 'data' / 'raw'
        self.clean = project_root / 'data' / 'clean'
        self.external = project_root / 'external_data'
        self.diana_dict = self.external / 'DiAna_dictionary'
        self.manual_fixes = self.external / 'manual_fixes'
        self.meddra = self.external / 'meddra' / 'MedAscii'
        
        # Ensure directories exist
        self.clean.mkdir(parents=True, exist_ok=True)
        
    def get_quarter_dir(self, quarter: str) -> Path:
        """
        Get path to quarter directory in raw data.
        
        Args:
            quarter (str): Quarter identifier (e.g., '2004q1')
            
        Returns:
            Path: Path to the quarter's ASCII data directory
        """
        return self.raw / quarter / 'ascii'
        
    def find_text_files(self, prefix: str) -> list[Path]:
        """
        Find text files case-insensitively across all quarters.
        
        Args:
            prefix (str): File prefix to search for
            
        Returns:
            list[Path]: List of matching file paths
        """
        pattern = f"**/{prefix}*.[Tt][Xx][Tt]"
        return list(self.raw.glob(pattern))
        
    def get_all_quarters(self) -> list[str]:
        """
        Get all available quarters in raw data directory.
        
        Returns:
            list[str]: Sorted list of quarter identifiers (e.g., ['2004q1', '2004q2'])
        """
        quarters = []
        for quarter_dir in self.raw.iterdir():
            if quarter_dir.is_dir() and re.match(r'\d{4}q[1-4]', quarter_dir.name.lower()):
                quarters.append(quarter_dir.name.lower())
        return sorted(quarters)

def setup_logging(level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def download_data(paths: ProjectPaths, max_date: Optional[str] = None) -> None:
    """
    Download all FAERS quarterly data.
    
    Downloads data for all quarters from 2004Q1 up to either:
    - The quarter specified by max_date
    - The current quarter if no max_date is specified
    
    Args:
        paths (ProjectPaths): Project paths manager
        max_date (Optional[str]): Maximum date in YYYYMMDD format
    """
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

def process_data(paths: ProjectPaths, max_date: Optional[str] = None,
                parallel: bool = True, max_workers: Optional[int] = None,
                chunk_size: Optional[int] = None) -> None:
    """
    Process all FAERS data in parallel.
    
    Args:
        paths (ProjectPaths): Project paths manager
        max_date (Optional[str]): Maximum date in YYYYMMDD format
        parallel (bool): Whether to use parallel processing (default: True)
        max_workers (Optional[int]): Number of worker processes (default: CPU count)
        chunk_size (Optional[int]): Size of data chunks for parallel processing (default: 50000)
    """
    standardizer = DataStandardizer(paths.external, paths.clean)
    
    # Process all available quarters
    quarters = paths.get_all_quarters()
    for quarter in quarters:
        try:
            quarter_dir = paths.get_quarter_dir(quarter)
            if quarter_dir.exists():
                logging.info(f"Processing quarter {quarter}...")
                standardizer.process_quarter(
                    quarter_dir,
                    max_date=max_date,
                    parallel=parallel,
                    max_workers=max_workers,
                    chunk_size=chunk_size or 50000
                )
        except Exception as e:
            logging.error(f"Error processing quarter {quarter}: {str(e)}")

def deduplicate_data(paths: ProjectPaths, max_date: Optional[str] = None,
                    parallel: bool = True, max_workers: Optional[int] = None,
                    chunk_size: Optional[int] = None) -> None:
    """
    Deduplicate all processed FAERS data in parallel.
    
    Args:
        paths (ProjectPaths): Project paths manager
        max_date (Optional[str]): Maximum date in YYYYMMDD format
        parallel (bool): Whether to use parallel processing (default: True)
        max_workers (Optional[int]): Number of worker processes (default: CPU count)
        chunk_size (Optional[int]): Size of data chunks for parallel processing (default: 50000)
    """
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

if __name__ == '__main__':
    main()
