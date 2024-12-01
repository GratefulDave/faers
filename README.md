# FAERS Data Processor

A Python package for downloading, processing, and analyzing FDA Adverse Event Reporting System (FAERS) data.

## Features

- Automated download of FAERS quarterly data files
- Data cleaning and standardization
- Multi-substance drug processing
- Probabilistic record linkage for deduplication
- Standardization of:
  - Preferred Terms (PT)
  - Occupation codes
  - Country codes
  - Age and weight units
- Comprehensive data validation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python -m src.faers_processor
```

With custom directories:
```bash
python -m src.faers_processor --raw-dir /path/to/raw/data --clean-dir /path/to/clean/data
```

## Project Structure

- `src/faers_processor/`
  - `__main__.py`: Entry point and main processing logic
  - `services/`
    - `downloader.py`: FAERS data download functionality
    - `processor.py`: Data processing and standardization
  - `models/`
    - `faers_data.py`: Data models for FAERS entities
  - `utils/`
    - `helpers.py`: Helper functions and utilities

## Data Processing Steps

1. **Download**: Automatically downloads FAERS quarterly data files
2. **Extract**: Unzips downloaded files
3. **Clean**: 
   - Standardizes field names
   - Converts units (age to days, weight to kg)
   - Normalizes country and occupation codes
4. **Process**:
   - Handles multi-substance drugs
   - Validates record completeness
   - Deduplicates records using probabilistic matching
5. **Save**: Stores processed data in pickle format

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
