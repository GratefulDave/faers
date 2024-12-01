"""Main entry point for FAERS data processing."""
import asyncio
import logging
from pathlib import Path
import argparse
from typing import Optional
import sys
from .services.downloader import FAERSDownloader
from .services.processor import FAERSProcessor
import pandas as pd

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('faers_processing.log')
        ]
    )

def main():
    try:
        # Set up logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Initialize paths
        raw_data_dir = Path('Raw_FAERS_QD')
        clean_data_dir = Path('Clean_Data')
        clean_data_dir.mkdir(exist_ok=True)
        
        if not raw_data_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")
        
        # Initialize processor
        processor = FAERSProcessor(raw_data_dir)
        
        # Step 1: Correct problematic files
        logger.info("Correcting problematic files...")
        problematic_files = {
            "Raw_FAERS_QD/aers_ascii_2011q2/ascii/DRUG11Q2.txt": "$$$$$$7475791",
            "Raw_FAERS_QD/aers_ascii_2011q3/ascii/DRUG11Q3.txt": "$$$$$$7652730",
            "Raw_FAERS_QD/aers_ascii_2011q4/ascii/DRUG11Q4.txt": "021487$7941354"
        }
        
        for file_path, old_line in problematic_files.items():
            full_path = raw_data_dir / file_path
            if full_path.exists():
                processor.correct_problematic_file(full_path, old_line)
                logger.info(f"Corrected problematic file: {file_path}")
        
        # Step 2: Process Demographics
        logger.info("Processing demographics data...")
        demo_data = []
        demo_files = list(raw_data_dir.rglob("DEMO*.txt"))
        if not demo_files:
            raise FileNotFoundError("No demographics files found")
            
        for file_path in demo_files:
            data = processor.process_demographics(file_path)
            demo_data.extend(data)
            logger.debug(f"Processed demographics file: {file_path}")
        demo_df = pd.DataFrame([vars(d) for d in demo_data])
        
        # Step 3: Standardize demographics data
        logger.info("Standardizing demographics data...")
        demo_df = processor.standardize_sex(demo_df)
        demo_df = processor.standardize_weight(demo_df)
        demo_df = processor.standardize_occupation(demo_df)
        demo_df = processor.standardize_country(demo_df)
        demo_df = processor.standardize_dates(demo_df, ['event_dt', 'mfr_dt', 'fda_dt', 'rept_dt'])
        
        # Step 4: Process Drug data
        logger.info("Processing drug data...")
        drug_data = []
        drug_files = list(raw_data_dir.rglob("DRUG*.txt"))
        if not drug_files:
            raise FileNotFoundError("No drug files found")
            
        for file_path in drug_files:
            data = processor.process_drugs(file_path)
            drug_data.extend(data)
            logger.debug(f"Processed drug file: {file_path}")
        drug_df = pd.DataFrame([vars(d) for d in drug_data])
        
        # Step 5: Process Drug Info data
        logger.info("Processing drug info data...")
        drug_info_data = []
        drug_info_files = list(raw_data_dir.rglob("DRUG*.txt"))
        if not drug_info_files:
            raise FileNotFoundError("No drug info files found")
            
        for file_path in drug_info_files:
            data = processor.process_drug_info(file_path)
            drug_info_data.extend(data)
            logger.debug(f"Processed drug info file: {file_path}")
        drug_info_df = pd.DataFrame([vars(d) for d in drug_info_data])
        
        # Step 6: Process Indications data
        logger.info("Processing indications data...")
        indi_data = []
        indi_files = list(raw_data_dir.rglob("INDI*.txt"))
        if not indi_files:
            raise FileNotFoundError("No indications files found")
            
        for file_path in indi_files:
            data = processor.process_indications(file_path)
            indi_data.extend(data)
            logger.debug(f"Processed indications file: {file_path}")
        indi_df = pd.DataFrame([vars(d) for d in indi_data])
        
        # Step 7: Process Reactions data
        logger.info("Processing reactions data...")
        reac_data = []
        reac_files = list(raw_data_dir.rglob("REAC*.txt"))
        if not reac_files:
            raise FileNotFoundError("No reactions files found")
            
        for file_path in reac_files:
            data = processor.process_reactions(file_path)
            reac_data.extend(data)
            logger.debug(f"Processed reactions file: {file_path}")
        reac_df = pd.DataFrame([vars(d) for d in reac_data])
        
        # Step 8: Process Outcomes data
        logger.info("Processing outcomes data...")
        outc_data = []
        outc_files = list(raw_data_dir.rglob("OUTC*.txt"))
        if not outc_files:
            raise FileNotFoundError("No outcomes files found")
            
        for file_path in outc_files:
            data = processor.process_outcomes(file_path)
            outc_data.extend(data)
            logger.debug(f"Processed outcomes file: {file_path}")
        outc_df = pd.DataFrame([vars(d) for d in outc_data])
        
        # Step 9: Process Report Sources data
        logger.info("Processing report sources data...")
        rpsr_data = []
        rpsr_files = list(raw_data_dir.rglob("RPSR*.txt"))
        if not rpsr_files:
            raise FileNotFoundError("No report sources files found")
            
        for file_path in rpsr_files:
            data = processor.process_report_sources(file_path)
            rpsr_data.extend(data)
            logger.debug(f"Processed report sources file: {file_path}")
        rpsr_df = pd.DataFrame([vars(d) for d in rpsr_data])
        
        # Step 10: Process Therapy data
        logger.info("Processing therapy data...")
        ther_data = []
        ther_files = list(raw_data_dir.rglob("THER*.txt"))
        if not ther_files:
            raise FileNotFoundError("No therapy files found")
            
        for file_path in ther_files:
            data = processor.process_therapy(file_path)
            ther_data.extend(data)
            logger.debug(f"Processed therapy file: {file_path}")
        ther_df = pd.DataFrame([vars(d) for d in ther_data])
        
        # Step 11: Process multi-substance drugs
        logger.info("Processing multi-substance drugs...")
        drug_df = processor.process_multi_substance_drugs(drug_df)
        
        # Step 12: Remove nullified reports
        logger.info("Removing nullified reports...")
        deleted_cases = []
        delete_files = list(raw_data_dir.rglob("*DELETE*.txt"))
        if not delete_files:
            raise FileNotFoundError("No delete files found")
            
        for file_path in delete_files:
            with open(file_path) as f:
                next(f)  # Skip header
                deleted_cases.extend([line.strip() for line in f])
        demo_df = demo_df[~demo_df['caseid'].isin(deleted_cases)]
        
        # Step 13: Deduplicate records
        logger.info("Deduplicating records...")
        demo_df = processor.deduplicate_records(demo_df, drug_df, reac_df)
        
        # Save processed data
        logger.info("Saving processed data...")
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
                try:
                    df.to_pickle(output_file)
                    logger.info(f"Saved {name} data to {output_file}")
                except Exception as e:
                    logger.error(f"Error saving {name} data: {str(e)}")
        
        logger.info("FAERS data processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error processing FAERS data: {str(e)}")
        raise

if __name__ == "__main__":
    main()
