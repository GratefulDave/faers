"""Main entry point for FAERS data processing."""
import asyncio
import logging
from pathlib import Path
import argparse
from typing import Optional
from .services.downloader import FAERSDownloader
from .services.processor import FAERSProcessor
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main(raw_data_dir: Optional[Path] = None,
              clean_data_dir: Optional[Path] = None):
    """Main function to process FAERS data."""
    # Set default directories if not provided
    raw_data_dir = raw_data_dir or Path('Raw_FAERS_QD')
    clean_data_dir = clean_data_dir or Path('Clean Data')
    
    # Ensure directories exist
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    clean_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download FAERS data
    logging.info("Downloading FAERS data...")
    async with FAERSDownloader(raw_data_dir) as downloader:
        await downloader.download_all()
    
    # Process downloaded data
    logging.info("Processing FAERS data...")
    processor = FAERSProcessor(raw_data_dir)
    
    # Initialize dataframes for each type
    demo_data = []
    drug_data = []
    reac_data = []
    outc_data = []
    
    # Process each type of data file
    for file_type in ['DEMO', 'DRUG', 'REAC', 'OUTC']:
        logging.info(f"Processing {file_type} files...")
        for file_path in raw_data_dir.rglob(f"{file_type}*.txt"):
            try:
                if file_type == 'DEMO':
                    data = processor.process_demographics(file_path)
                    demo_data.extend(data)
                elif file_type == 'DRUG':
                    data = processor.process_drugs(file_path)
                    drug_data.extend(data)
                elif file_type == 'REAC':
                    data = processor.process_reactions(file_path)
                    reac_data.extend(data)
                elif file_type == 'OUTC':
                    data = processor.process_outcomes(file_path)
                    outc_data.extend(data)
                
                logging.info(f"Processed {file_path.name}")
                
            except Exception as e:
                logging.error(f"Error processing {file_path}: {str(e)}")
                continue
    
    # Convert lists to dataframes
    demo_df = pd.DataFrame([vars(d) for d in demo_data])
    drug_df = pd.DataFrame([vars(d) for d in drug_data])
    reac_df = pd.DataFrame([vars(d) for d in reac_data])
    outc_df = pd.DataFrame([vars(d) for d in outc_data])
    
    # Process multi-substance drugs
    logging.info("Processing multi-substance drugs...")
    drug_df = processor.process_multi_substance_drugs(drug_df)
    
    # Validate and deduplicate records
    logging.info("Validating and deduplicating records...")
    demo_df = processor.validate_record(demo_df, drug_df, reac_df)
    demo_df = processor.deduplicate_records(demo_df, drug_df, reac_df)
    
    # Save processed data
    logging.info("Saving processed data...")
    for name, df in [('DEMO', demo_df), ('DRUG', drug_df), 
                    ('REAC', reac_df), ('OUTC', outc_df)]:
        output_file = clean_data_dir / f"{name}.rds"
        df.to_pickle(output_file)
    
    logging.info("FAERS data processing completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process FAERS data")
    parser.add_argument("--raw-dir", type=Path, help="Directory for raw FAERS data")
    parser.add_argument("--clean-dir", type=Path, help="Directory for cleaned data")
    
    args = parser.parse_args()
    asyncio.run(main(args.raw_dir, args.clean_dir))
