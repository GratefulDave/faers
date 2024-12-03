#!/usr/bin/env python3

from pathlib import Path
from src.faers_processor.services.standardizer import DataStandardizer
from src.faers_processor.services.processor import FAERSProcessor

def main():
    # Get absolute paths
    root_dir = Path(__file__).parent
    input_dir = root_dir / 'data' / 'raw'
    output_dir = root_dir / 'data' / 'clean'
    external_dir = root_dir / 'external_data'
    
    # Initialize components
    standardizer = DataStandardizer(external_dir)
    processor = FAERSProcessor(standardizer)
    
    # Process all quarters
    processor.process_all(
        input_dir=input_dir,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()
