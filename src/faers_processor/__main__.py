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

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize paths
    raw_data_dir = Path('Raw_FAERS_QD')
    clean_data_dir = Path('Clean_Data')
    clean_data_dir.mkdir(exist_ok=True)
    
    # Initialize processor
    processor = FAERSProcessor(raw_data_dir)
    
    # Step 1: Correct problematic files
    logging.info("Correcting problematic files...")
    problematic_files = {
        "Raw_FAERS_QD/aers_ascii_2011q2/ascii/DRUG11Q2.txt": "$$$$$$7475791",
        "Raw_FAERS_QD/aers_ascii_2011q3/ascii/DRUG11Q3.txt": "$$$$$$7652730",
        "Raw_FAERS_QD/aers_ascii_2011q4/ascii/DRUG11Q4.txt": "021487$7941354"
    }
    
    for file_path, old_line in problematic_files.items():
        full_path = raw_data_dir / file_path
        if full_path.exists():
            processor.correct_problematic_file(full_path, old_line)
    
    # Step 2: Process Demographics
    logging.info("Processing demographics data...")
    demo_data = []
    for file_path in raw_data_dir.rglob("DEMO*.txt"):
        data = processor.process_demographics(file_path)
        demo_data.extend(data)
    demo_df = pd.DataFrame([vars(d) for d in demo_data])
    
    # Step 3: Standardize demographics data
    logging.info("Standardizing demographics data...")
    demo_df = processor.standardize_sex(demo_df)
    demo_df = processor.standardize_weight(demo_df)
    demo_df = processor.standardize_occupation(demo_df)
    demo_df = processor.standardize_country(demo_df)
    demo_df = processor.standardize_dates(demo_df, ['event_dt', 'mfr_dt', 'fda_dt', 'rept_dt'])
    
    # Step 4: Process Drug data
    logging.info("Processing drug data...")
    drug_data = []
    for file_path in raw_data_dir.rglob("DRUG*.txt"):
        data = processor.process_drugs(file_path)
        drug_data.extend(data)
    drug_df = pd.DataFrame([vars(d) for d in drug_data])
    
    # Step 5: Process Drug Info data
    logging.info("Processing drug info data...")
    drug_info_data = []
    for file_path in raw_data_dir.rglob("DRUG*.txt"):
        data = processor.process_drug_info(file_path)
        drug_info_data.extend(data)
    drug_info_df = pd.DataFrame([vars(d) for d in drug_info_data])
    
    # Step 6: Process Indications data
    logging.info("Processing indications data...")
    indi_data = []
    for file_path in raw_data_dir.rglob("INDI*.txt"):
        data = processor.process_indications(file_path)
        indi_data.extend(data)
    indi_df = pd.DataFrame([vars(d) for d in indi_data])
    
    # Step 7: Process Reactions data
    logging.info("Processing reactions data...")
    reac_data = []
    for file_path in raw_data_dir.rglob("REAC*.txt"):
        data = processor.process_reactions(file_path)
        reac_data.extend(data)
    reac_df = pd.DataFrame([vars(d) for d in reac_data])
    
    # Step 8: Process Outcomes data
    logging.info("Processing outcomes data...")
    outc_data = []
    for file_path in raw_data_dir.rglob("OUTC*.txt"):
        data = processor.process_outcomes(file_path)
        outc_data.extend(data)
    outc_df = pd.DataFrame([vars(d) for d in outc_data])
    
    # Step 9: Process Report Sources data
    logging.info("Processing report sources data...")
    rpsr_data = []
    for file_path in raw_data_dir.rglob("RPSR*.txt"):
        data = processor.process_report_sources(file_path)
        rpsr_data.extend(data)
    rpsr_df = pd.DataFrame([vars(d) for d in rpsr_data])
    
    # Step 10: Process Therapy data
    logging.info("Processing therapy data...")
    ther_data = []
    for file_path in raw_data_dir.rglob("THER*.txt"):
        data = processor.process_therapy(file_path)
        ther_data.extend(data)
    ther_df = pd.DataFrame([vars(d) for d in ther_data])
    
    # Step 11: Process multi-substance drugs
    logging.info("Processing multi-substance drugs...")
    drug_df = processor.process_multi_substance_drugs(drug_df)
    
    # Step 12: Remove nullified reports
    logging.info("Removing nullified reports...")
    deleted_cases = []
    for file_path in raw_data_dir.rglob("*DELETE*.txt"):
        with open(file_path) as f:
            next(f)  # Skip header
            deleted_cases.extend([line.strip() for line in f])
    demo_df = demo_df[~demo_df['caseid'].isin(deleted_cases)]
    
    # Step 13: Deduplicate records
    logging.info("Deduplicating records...")
    demo_df = processor.deduplicate_records(demo_df, drug_df, reac_df)
    
    # Save processed data
    logging.info("Saving processed data...")
    dataframes = {
        'DEMO': demo_df,
        'DRUG': drug_df,
        'DRUG_INFO': drug_info_df,
        'INDI': indi_df,
        'REAC': reac_df,
        'OUTC': outc_df,
        'RPSR': rpsr_df,
        'THER': ther_df
    }
    
    for name, df in dataframes.items():
        if not df.empty:
            output_file = clean_data_dir / f"{name}.rds"
            df.to_pickle(output_file)
            logging.info(f"Saved {name} data to {output_file}")
    
    logging.info("FAERS data processing completed!")

if __name__ == "__main__":
    main()
