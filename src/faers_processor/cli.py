#!/usr/bin/env python3
"""
CLI interface for FAERS data processing.
"""

import argparse
import logging
from pathlib import Path

from faers_processor.services.standardizer import DataStandardizer

def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description='Process FAERS quarterly data files')
    
    parser.add_argument(
        '--quarters-dir',
        type=Path,
        required=True,
        help='Directory containing FAERS quarterly data'
    )
    
    parser.add_argument(
        '--external-dir',
        type=Path,
        required=True,
        help='Directory containing external reference data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Directory for processed output'
    )
    
    parser.add_argument(
        '--report-file',
        type=Path,
        required=True,
        help='Path to save the processing report'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel processing using dask'
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
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize standardizer
    standardizer = DataStandardizer(args.external_dir, args.output_dir)
    
    # Process quarters and generate report
    report = standardizer.process_quarters(
        args.quarters_dir,
        parallel=args.parallel,
        n_workers=args.workers
    )
    
    # Save report
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text(report)
    logging.info(f"Report saved to {args.report_file}")

if __name__ == '__main__':
    main()
