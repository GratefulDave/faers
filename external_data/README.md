# External Reference Data

This directory contains external reference files used for data standardization and mapping:

## Files

1. `country_codes.csv`: Standard country codes and mappings
   - ISO country codes
   - Common country name variations
   - Region information

2. `occupation_codes.csv`: Occupation code standardization
   - Standard occupation codes
   - Code descriptions
   - Mapping of variations

3. `drug_names.csv`: Drug name standardization
   - Generic names
   - Brand names
   - Common variations

4. `pt_standardization.csv`: Preferred Terms standardization
   - MedDRA PT codes
   - Common variations
   - Hierarchical relationships

## Usage

These files are used by the FAERS processor for:
- Standardizing country codes
- Mapping occupation codes
- Normalizing drug names
- Standardizing preferred terms (PT)

## Updating

To update these files:
1. Maintain the same column structure
2. Add new mappings as needed
3. Document any major changes
4. Keep a backup of previous versions
