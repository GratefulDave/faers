# DiAna FAERS Data Processing Framework

A comprehensive Python framework for processing FDA Adverse Event Reporting System (FAERS) data, optimized for Apple Silicon and large-scale data processing.

## Features

- Parallel data processing with Dask
- Memory-efficient operations with Vaex
- Apple Silicon (M1/M2) optimizations
- Comprehensive data standardization
- Intelligent deduplication
- Robust error handling

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DiAna.git
cd DiAna

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Processing Large-Scale Data with Dask

For datasets that don't fit in memory, use Dask's distributed computing capabilities:

### 1. Command Line Usage

```bash
# Basic usage with Dask
python -m faers_processor \
    --data-dir /path/to/data \
    --external-dir /path/to/external \
    --use-dask \
    --chunk-size 100000 \
    --max-workers 8

# For very large datasets (>100GB)
python -m faers_processor \
    --data-dir /path/to/data \
    --external-dir /path/to/external \
    --use-dask \
    --chunk-size 50000 \
    --max-workers 4 \
    --memory-limit "32GB"
```

### 2. Configuration Options

- `--use-dask`: Enable Dask for distributed processing
- `--chunk-size`: Number of rows per chunk (default: 100000)
- `--max-workers`: Number of worker processes (default: CPU count - 1)
- `--memory-limit`: Memory limit per worker (default: "4GB")

### 3. Memory Management

Dask provides several ways to manage memory:

```python
# In your processing script
import dask
from dask.distributed import Client, LocalCluster

# Configure Dask for your system
dask.config.set({
    'distributed.worker.memory.target': 0.6,  # Target 60% memory use
    'distributed.worker.memory.spill': 0.7,   # Spill to disk at 70%
    'distributed.worker.memory.pause': 0.8,   # Pause work at 80%
    'distributed.worker.memory.terminate': 0.95  # Restart worker at 95%
})

# Create a local cluster with memory limits
cluster = LocalCluster(
    n_workers=4,  # Number of worker processes
    threads_per_worker=2,  # Threads per worker
    memory_limit='32GB'  # Memory limit per worker
)

# Create a client
client = Client(cluster)
```

### 4. Processing Strategies

#### a. Chunked Processing
```python
# Process data in chunks
ddf = dd.read_csv('large_file.csv', blocksize='64MB')
ddf = ddf.map_partitions(process_chunk)
result = ddf.compute()
```

#### b. Out-of-Core Processing
```python
# Process data that doesn't fit in memory
ddf = dd.read_parquet('large_data/*.parquet')
result = (ddf
    .map_partitions(standardize_data)
    .repartition(npartitions=100)
    .map_partitions(deduplicate_data)
    .compute())
```

#### c. Incremental Processing
```python
# Process and save incrementally
for chunk in ddf.partitions:
    processed = chunk.compute()
    save_to_disk(processed)
```

### 5. Best Practices

1. **Memory Management**
   - Monitor memory usage with `dask.distributed.diagnostics`
   - Use `blocksize` for CSV files and `partition_size` for parquet
   - Implement spill-to-disk for large intermediate results

2. **Performance Optimization**
   - Use appropriate chunk sizes (typically 50-100MB)
   - Monitor task progress with `progress()`
   - Use profiling tools: `dask.distributed.diagnostics`

3. **Error Handling**
   - Implement retry logic for failed tasks
   - Use checkpointing for long-running computations
   - Monitor worker health and restart as needed

### 6. Example Pipeline

```python
from faers_processor import DataStandardizer
import dask.dataframe as dd

# Initialize with Dask configuration
standardizer = DataStandardizer(
    external_dir='path/to/external',
    use_dask=True,
    chunk_size=100000,
    memory_limit='32GB'
)

# Process demographics data
demo_ddf = dd.read_csv('DEMO*.txt', blocksize='64MB')
demo_processed = demo_ddf.map_partitions(
    standardizer.process_demographics_dask
)

# Process drug data
drug_ddf = dd.read_csv('DRUG*.txt', blocksize='64MB')
drug_processed = drug_ddf.map_partitions(
    standardizer.process_drugs_dask
)

# Save results incrementally
demo_processed.to_parquet(
    'processed/demo',
    partition_on=['quarter'],
    engine='pyarrow',
    compression='snappy'
)
```

### 7. Monitoring and Debugging

1. **Dask Dashboard**
   ```bash
   # Start Jupyter with Dask dashboard
   jupyter lab --DaskDashboard.link="http://localhost:8787/status"
   ```

2. **Memory Monitoring**
   ```python
   from dask.distributed import performance_report

   with performance_report(filename="dask-report.html"):
       result = ddf.compute()
   ```

3. **Progress Tracking**
   ```python
   from dask.diagnostics import ProgressBar

   with ProgressBar():
       result = ddf.compute()
   ```

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce chunk size
   - Increase disk spill threshold
   - Use incremental processing

2. **Performance Issues**
   - Adjust number of workers
   - Monitor CPU/memory usage
   - Check network bottlenecks

3. **Data Integrity**
   - Validate input data
   - Implement checksums
   - Use data quality checks

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
