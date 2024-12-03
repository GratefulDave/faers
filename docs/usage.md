# DiAna FAERS Data Processing Usage Guide

## Quick Start

Process FAERS data with default settings:
```bash
python -m src.faers_processor --process
```

## Common Use Cases

### 1. Download and Process New Data
Download latest FAERS data and process it:
```bash
python -m src.faers_processor --download --process
```

### 2. Parallel Processing with Dask
Process data using parallel processing for better performance:
```bash
python -m src.faers_processor --process --use-dask --max-workers 4
```

### 3. Memory-Optimized Processing
Process large datasets with smaller chunks:
```bash
python -m src.faers_processor --process --chunk-size 50000
```

### 4. Complete Pipeline
Download, process, and deduplicate data:
```bash
python -m src.faers_processor --download --process --deduplicate
```

### 5. Debug Mode
Run with detailed logging for troubleshooting:
```bash
python -m src.faers_processor --process --log-level DEBUG
```

## Command Line Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--process` | Process raw FAERS data files | False | `--process` |
| `--download` | Download latest FAERS data | False | `--download` |
| `--deduplicate` | Deduplicate processed data | False | `--deduplicate` |
| `--chunk-size` | Number of rows to process at once | 100000 | `--chunk-size 50000` |
| `--use-dask` | Use Dask for parallel processing | False | `--use-dask` |
| `--max-workers` | Maximum parallel workers | CPU count | `--max-workers 4` |
| `--log-level` | Logging verbosity | INFO | `--log-level DEBUG` |

## Processing Stages

### 1. Data Download
Downloads quarterly FAERS data files:
```bash
python -m src.faers_processor --download
```

Output structure:
```
data/
├── raw/
│   ├── 2023Q1/
│   ├── 2023Q2/
│   └── ...
```

### 2. Data Processing
Standardizes and cleans FAERS data:
```bash
python -m src.faers_processor --process
```

Processing includes:
- Date standardization
- Drug name normalization
- Demographics standardization
- Error handling and logging

### 3. Deduplication
Removes duplicate records:
```bash
python -m src.faers_processor --deduplicate
```

## Performance Optimization

### Memory Usage
For large datasets, adjust chunk size:
```bash
# Process with smaller chunks for less memory usage
python -m src.faers_processor --process --chunk-size 50000

# Process with larger chunks for better performance
python -m src.faers_processor --process --chunk-size 200000
```

### Parallel Processing
Enable parallel processing for faster execution:
```bash
# Use all available CPU cores
python -m src.faers_processor --process --use-dask

# Specify number of workers
python -m src.faers_processor --process --use-dask --max-workers 4
```

## Troubleshooting

### Common Issues

1. Memory Errors
```bash
# Reduce chunk size
python -m src.faers_processor --process --chunk-size 25000
```

2. Processing Errors
```bash
# Enable debug logging
python -m src.faers_processor --process --log-level DEBUG
```

3. Performance Issues
```bash
# Enable parallel processing with optimal workers
python -m src.faers_processor --process --use-dask --max-workers $(( $(nproc) - 1 ))
```

## Output and Reporting

### Processing Report
Each processing run generates a detailed report including:
- Per-quarter statistics
- Invalid date counts
- Processing times
- Error summaries

Example report sections:
```markdown
# FAERS Processing Summary Report

## Quarter-by-Quarter Summary

### Quarter 2023Q1
#### Demographics
| Metric | Value |
|--------|--------|
| Total Rows | 100000 |
| Invalid event_dt | 150 |
| Invalid fda_dt | 200 |

#### Drug Data
| Metric | Value |
|--------|--------|
| Total Rows | 150000 |
| Parsing Errors | 10 |
```

## Best Practices

1. Start with a test run:
```bash
python -m src.faers_processor --process --chunk-size 10000 --log-level DEBUG
```

2. For production:
```bash
python -m src.faers_processor --download --process --deduplicate --use-dask
```

3. For large datasets:
```bash
python -m src.faers_processor --process --chunk-size 50000 --use-dask --max-workers 4
```

## Environment Setup

Required Python version: 3.10+

Required packages:
```bash
pip install pandas numpy dask distributed tqdm tabulate
```

## Data Directory Structure

```
project_root/
├── data/
│   ├── raw/           # Downloaded FAERS data
│   ├── processed/     # Standardized data
│   └── final/         # Deduplicated data
├── logs/              # Processing logs
└── reports/           # Generated reports
```
