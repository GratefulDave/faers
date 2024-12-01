# FAERS Data Processing Documentation

## Overview

This documentation describes the Python implementation of the FAERS (FDA Adverse Event Reporting System) data processing pipeline. The implementation closely follows the original R script while adding Python-specific improvements for robustness and error handling.

## Data Structure

The FAERS data consists of several interconnected datasets:

1. **DEMO (Demographics)**: Contains patient demographic information
   - Primary key: `primary_id`
   - Key fields: `case_id`, `case_version`, `sex`, `age`, `weight`, `reporter_country`

2. **DRUG**: Contains drug information
   - Primary key: (`primary_id`, `drug_seq`)
   - Key fields: `role_code`, `drugname`, `prod_ai`

3. **DRUG_INFO**: Additional drug information
   - Foreign key: (`primary_id`, `drug_seq`) references DRUG
   - Key fields: `val_vbm`, `route`, `dose_form`, `dose_freq`

4. **INDI (Indications)**: Drug indications
   - Foreign key: (`primary_id`, `drug_seq`) references DRUG
   - Key fields: `indi_pt` (Preferred Term)

5. **REAC (Reactions)**: Adverse reactions
   - Primary key: `primary_id`
   - Key fields: `pt` (Preferred Term)

6. **OUTC (Outcomes)**: Patient outcomes
   - Primary key: `primary_id`
   - Key field: `outcome_code`

7. **RPSR (Report Sources)**: Report source information
   - Primary key: `primary_id`
   - Key field: `rpsr_cod`

8. **THER (Therapy)**: Drug therapy information
   - Foreign key: (`primary_id`, `drug_seq`) references DRUG
   - Key fields: `start_dt`, `end_dt`, `dur`, `dur_cod`

## System Architecture

```mermaid
graph TB
    subgraph Input
        A[Raw FAERS Files] --> B[File Processor]
        E[External Data] --> B
    end
    
    subgraph Core Processing
        B --> C[Data Standardization]
        C --> D[Record Deduplication]
        D --> F[Data Validation]
    end
    
    subgraph Output
        F --> G[Clean Data Files]
        F --> H[Processing Logs]
    end
    
    subgraph External Services
        I[MedDRA Dictionary] -.-> C
    end
```

## Component Interaction

```mermaid
sequenceDiagram
    participant M as Main
    participant P as Processor
    participant F as File System
    participant E as External Data
    
    M->>P: Initialize
    P->>E: Load Reference Data
    
    loop For each file type
        M->>P: Process Files
        P->>F: Read Raw Data
        F-->>P: Raw Data
        P->>P: Standardize Data
        P->>P: Validate Data
        P->>F: Save Clean Data
    end
    
    M->>P: Process Multi-substance Drugs
    M->>P: Remove Nullified Reports
    M->>P: Deduplicate Records
    P->>F: Save Final Data
```

## Data Flow

```mermaid
flowchart LR
    subgraph Input Files
        DEMO[DEMO.txt]
        DRUG[DRUG.txt]
        REAC[REAC.txt]
        INDI[INDI.txt]
        THER[THER.txt]
        OUTC[OUTC.txt]
    end
    
    subgraph Processing
        direction TB
        STAND[Standardization]
        DEDUP[Deduplication]
        VAL[Validation]
    end
    
    subgraph Output Files
        DEMO_C[DEMO.rds]
        DRUG_C[DRUG.rds]
        REAC_C[REAC.rds]
        INDI_C[INDI.rds]
        THER_C[THER.rds]
        OUTC_C[OUTC.rds]
    end
    
    DEMO --> STAND
    DRUG --> STAND
    REAC --> STAND
    INDI --> STAND
    THER --> STAND
    OUTC --> STAND
    
    STAND --> DEDUP
    DEDUP --> VAL
    
    VAL --> DEMO_C
    VAL --> DRUG_C
    VAL --> REAC_C
    VAL --> INDI_C
    VAL --> THER_C
    VAL --> OUTC_C
```

## Processing States

```mermaid
stateDiagram-v2
    [*] --> Initialization
    
    Initialization --> LoadingExternalData
    LoadingExternalData --> ProcessingFiles
    
    state ProcessingFiles {
        [*] --> Demographics
        Demographics --> Drugs
        Drugs --> DrugInfo
        DrugInfo --> Indications
        Indications --> Reactions
        Reactions --> Outcomes
        Outcomes --> ReportSources
        ReportSources --> Therapy
        Therapy --> [*]
    }
    
    ProcessingFiles --> MultiSubstanceDrugs
    MultiSubstanceDrugs --> RemovingNullified
    RemovingNullified --> Deduplication
    Deduplication --> DataValidation
    DataValidation --> SavingData
    SavingData --> [*]
```

## Error Handling Flow

```mermaid
flowchart TD
    A[Start Processing] --> B{File Exists?}
    B -->|No| C[Log Error]
    B -->|Yes| D{Valid Format?}
    
    D -->|No| E[Log Error]
    D -->|Yes| F{Process Data}
    
    F -->|Success| G[Save Data]
    F -->|Failure| H[Log Error]
    
    C --> I[Continue to Next File]
    E --> I
    H --> I
    
    G --> I
    I --> J{More Files?}
    J -->|Yes| B
    J -->|No| K[End Processing]
```

## Data Relationships

```mermaid
erDiagram
    DEMO ||--o{ DRUG : contains
    DEMO ||--o{ REAC : reports
    DEMO ||--o{ OUTC : has
    DRUG ||--o{ INDI : indicates
    DRUG ||--o{ THER : prescribes
    
    DEMO {
        string primary_id PK
        string case_id
        string sex
        float age
        float weight
        string country
    }
    
    DRUG {
        string primary_id FK
        int drug_seq
        string role_code
        string drug_name
        string prod_ai
    }
    
    REAC {
        string primary_id FK
        string pt
    }
    
    INDI {
        string primary_id FK
        int drug_seq FK
        string indi_pt
    }
    
    THER {
        string primary_id FK
        int drug_seq FK
        date start_dt
        date end_dt
    }
    
    OUTC {
        string primary_id FK
        string outc_code
    }
```

## Processing Pipeline

### 1. File Preparation
- Correct problematic files with missing newlines
- Handle specific cases from 2011Q2-Q4

### 2. Data Processing Steps

#### a. Demographics Processing
- Standardize sex values (M/F)
- Convert weights to kilograms
- Standardize country codes
- Convert age to days
- Validate dates

#### b. Drug Processing
- Process drug names
- Handle multi-substance drugs
- Link with drug information and therapy data
- Standardize routes and dose forms

#### c. Reaction Processing
- Standardize Preferred Terms (PT)
- Remove invalid reactions
- Link with outcomes

#### d. Data Cleaning
- Remove nullified reports
- Deduplicate records using:
  - Rule-based deduplication
  - Probabilistic record linkage
  - Suspect drug consideration

### 3. Standardization

#### a. External Reference Data
Located in `external_data/`:
- Country codes (`country_codes.csv`)
- Occupation codes (`occupation_codes.csv`)
- Drug name mappings (placeholder)

#### b. MedDRA Integration
- **Note**: MedDRA dictionary access requires separate subscription
- Use `pt_fixed` for standardizing unstandardized PTs
- Handles multiple levels of MedDRA hierarchy

### 4. Output

Processed data is saved in the `Clean_Data` directory:
- All files saved as `.rds` format
- Maintains relationships between datasets
- Includes standardization status

## Error Handling

The Python implementation adds robust error handling:
- File operation validation
- Data format checking
- Logging of processing steps
- Recovery from processing errors

## Logging

Comprehensive logging is implemented:
- Console output for monitoring
- File logging for debugging
- Progress tracking
- Error reporting

## Usage

```python
from faers_processor import FAERSProcessor

# Initialize processor
processor = FAERSProcessor("Raw_FAERS_QD")

# Process all data
processor.process_all()
```

## Dependencies

- Python 3.8+
- pandas
- numpy
- pathlib
- logging

## Development History

```mermaid
gitGraph
    commit id: "init" tag: "v0.1.0"
    commit id: "setup-project-structure"
    branch feature/data-processor
    checkout feature/data-processor
    commit id: "add-basic-processor"
    commit id: "implement-standardization"
    checkout main
    merge feature/data-processor
    
    branch feature/file-handling
    checkout feature/file-handling
    commit id: "add-file-validation"
    commit id: "improve-error-handling"
    checkout main
    merge feature/file-handling
    
    branch feature/data-cleaning
    checkout feature/data-cleaning
    commit id: "add-weight-standardization"
    commit id: "add-date-validation"
    commit id: "add-sex-validation"
    checkout main
    merge feature/data-cleaning tag: "v0.2.0"
    
    branch feature/deduplication
    checkout feature/deduplication
    commit id: "implement-basic-dedup"
    commit id: "add-probabilistic-linkage"
    commit id: "optimize-dedup-performance"
    checkout main
    merge feature/deduplication
    
    branch feature/external-data
    checkout feature/external-data
    commit id: "add-country-codes"
    commit id: "add-occupation-codes"
    commit id: "add-drug-names"
    checkout main
    merge feature/external-data tag: "v0.3.0"
    
    branch feature/documentation
    checkout feature/documentation
    commit id: "add-basic-docs"
    commit id: "add-system-diagrams"
    commit id: "add-api-docs"
    checkout main
    merge feature/documentation
    
    branch feature/testing
    checkout feature/testing
    commit id: "add-unit-tests"
    commit id: "add-integration-tests"
    commit id: "add-test-data"
    checkout main
    merge feature/testing tag: "v0.4.0"
```

## Feature Branch Details

### feature/data-processor
- Basic processor implementation
- Data standardization framework
- Core processing pipeline

### feature/file-handling
- File validation mechanisms
- Improved error handling
- Robust file operations

### feature/data-cleaning
- Weight standardization across units
- Date validation and normalization
- Sex field standardization
- Data quality checks

### feature/deduplication
- Basic deduplication logic
- Probabilistic record linkage
- Performance optimizations
- Duplicate marking system

### feature/external-data
- Country code mapping
- Occupation code standardization
- Drug name reference data
- External data management

### feature/documentation
- Basic documentation structure
- System architecture diagrams
- API documentation
- Usage guides

### feature/testing
- Unit test framework
- Integration tests
- Test data generation
- CI/CD setup

## Notes

1. The implementation maintains exact compatibility with the R script's output
2. Additional validation and error checking has been added
3. Processing is more memory-efficient
4. Better handling of edge cases
