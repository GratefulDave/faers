#!/usr/bin/env python3
"""
CLI interface for FAERS data processing.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

from faers_processor.services.standardizer import DataStandardizer
from faers_processor.services.downloader import FAERSDownloader
from faers_processor.services.deduplicator import FAERSDeduplicator

class ProjectPaths:
    """Manages project directory paths."""
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
        
    def get_quarter_dir(self, quarter: str) -> Path:
        """Get path to quarter directory in raw data."""
        return self.raw / quarter / 'ascii'
        
    def find_text_files(self, quarter: str, prefix: str) -> list[Path]:
        """Find text files case-insensitively."""
        quarter_dir = self.get_quarter_dir(quarter)
        pattern = f"{prefix}*.[Tt][Xx][Tt]"
        return list(quarter_dir.glob(pattern))

def setup_logging(level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def download_data(paths: ProjectPaths, quarters: list[str], max_date: Optional[str] = None) -> None:
    """Download FAERS quarterly data."""
    downloader = FAERSDownloader(paths.raw)
    for quarter in quarters:
        downloader.download_quarter(quarter, max_date)

def process_data(paths: ProjectPaths, quarters: list[str], max_date: Optional[str] = None,
                parallel: bool = False, workers: Optional[int] = None) -> None:
    """Process FAERS data."""
    standardizer = DataStandardizer(paths.external, paths.clean)
    for quarter in quarters:
        standardizer.process_quarter(
            paths.get_quarter_dir(quarter),
            max_date=max_date,
            parallel=parallel,
            n_workers=workers
        )

def deduplicate_data(paths: ProjectPaths, max_date: Optional[str] = None) -> None:
    """Deduplicate processed FAERS data."""
    deduplicator = FAERSDeduplicator(paths.clean)
    deduplicator.deduplicate_all(max_date)

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
        '--quarters',
        nargs='+',
        help='List of quarters to process (e.g., 2004q1 2004q2)'
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
        help='Enable parallel processing'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        help='Number of worker processes for parallel processing'
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
    steps = args.steps[0] if args.steps[0] != 'all' else ['download', 'process', 'deduplicate']
    
    if 'download' in steps:
        logging.info("Starting download step...")
        download_data(paths, args.quarters, args.max_date)
    
    if 'process' in steps:
        logging.info("Starting processing step...")
        process_data(paths, args.quarters, args.max_date, args.parallel, args.workers)
    
    if 'deduplicate' in steps:
        logging.info("Starting deduplication step...")
        deduplicate_data(paths, args.max_date)
    
    logging.info("All requested steps completed successfully")

if __name__ == '__main__':
    main()
