# Data Preparation Scripts

Scripts to extract and prepare reference data from raw source files.

## Prerequisites

Download these files before running the scripts:

### 1. ICD-10-CM Code Table (April 2026 or later)
- **File**: `icd10cm-codes-April-1-2026.txt` (or similar)
- **Source**: https://www.cdc.gov/nchs/icd/icd-10-cm/files.html
- **Size**: ~10-15 MB
- **Format**: Tab-delimited flat file
  - Format: `code<tab>description`
  - Codes WITHOUT dots: `E1165` not `E11.65`
  - Example: `E1165    Type 2 diabetes mellitus with hyperglycemia`
  - Script automatically inserts dots at position 3
- **Description**: Official ICD-10-CM code descriptions from CMS

### 2. SNOMED→ICD-10 Crosswalk
- **File**: Tab-delimited .txt or .csv format
  - From SNOMED CT US Edition: `.txt` file in "SNOMED CT to ICD-10-CM Map" folder
  - From UMLS: `snomed_icd_10_map.csv_0_0_0.csv`
- **Source**:
  - https://www.nlm.nih.gov/research/umls/mapping_projects/snomedct_to_icd10cm.html
  - Or SNOMED CT United States Edition download
- **Size**: ~40-100 MB
- **Format**: Tab-delimited with columns:
  - `referencedComponentId` (SNOMED CT code)
  - `mapTarget` (ICD-10-CM code)
  - `mapGroup`, `mapPriority`, `mapRule`, `mapAdvice`, `active`
- **Description**: Maps SNOMED CT concepts to ICD-10-CM codes
- **Requires**: Free UMLS account

### 3. UMLS Metathesaurus 2025AB
- **Files**: MRCONSO.RRF, MRHIER.RRF, MRREL.RRF, MRSTY.RRF
- **Source**: https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html
- **Size**: ~1-2 GB (full download)
- **Description**: UMLS knowledge sources including SNOMED CT
- **Requires**: Free UMLS account (approval takes 1-3 business days)
- **Location**: Extract to `data/raw_umls/META/`

## Setup

```bash
# Create directory for raw files
mkdir -p data/raw_umls/META

# Place downloaded files:
# - ICD-10 table → data/raw_umls/ (or any location)
# - SNOMED crosswalk CSV → data/raw_umls/
# - UMLS META files → data/raw_umls/META/
```

## Script Execution Order

Run these scripts in order:

### Step 1: Parse ICD-10 Codes

```bash
python scripts/build_icd10_reference.py data/raw_umls/icd10cm-codes-April-1-2026.txt \
    --output data/reference/icd10_diabetes_codes.json
```

**Output**: `data/reference/icd10_diabetes_codes.json` (~50-60 diabetes-related codes)

**What it does**:
- Reads CMS flat file (tab-delimited, codes without dots)
- Automatically formats codes: E1165 → E11.65, Z794 → Z79.4
- Filters to diabetes codes (E08-E13)
- Adds related codes: Z79.4/Z79.84 (medication use), G57/G63 (neuropathy), N18 (CKD), I25/I10 (cardiovascular)
- Marks HCC (high-severity) codes
- Identifies billable vs non-billable codes (4+ character codes are billable)

### Step 2: Parse SNOMED→ICD-10 Crosswalk

```bash
# If using .txt format (from SNOMED CT US Edition)
python scripts/build_crosswalk.py data/raw_umls/snomed_to_icd10_map.txt \\
    --output data/reference/snomed_icd10_crosswalk.json \\
    --filter-diabetes \\
    --icd10-reference data/reference/icd10_diabetes_codes.json

# Or if using .csv format (from UMLS)
python scripts/build_crosswalk.py data/raw_umls/snomed_icd_10_map.csv_0_0_0.csv \\
    --output data/reference/snomed_icd10_crosswalk.json \\
    --filter-diabetes \\
    --icd10-reference data/reference/icd10_diabetes_codes.json
```

**Output**: `data/reference/snomed_icd10_crosswalk.json` (diabetes-filtered crosswalk)

**What it does**:
- Parses SNOMED→ICD-10 mapping CSV
- Filters to diabetes-related mappings (if --filter-diabetes)
- Ranks mappings by priority when multiple ICD-10 codes map to one SNOMED concept
- Handles one-to-many and many-to-one mappings

### Step 3: Extract SNOMED Diabetes Concepts

**Option A: SNOMED CT US Edition (native format) - Recommended**

```bash
python scripts/extract_snomed_diabetes.py \\
    data/raw_snomed/Snapshot/Terminology \\
    --format snomed \\
    --output data/reference/snomed_diabetes_subset.json \\
    --max-depth 4
```

**Option B: UMLS Metathesaurus**

```bash
python scripts/extract_snomed_diabetes.py \\
    data/raw_umls/META \\
    --format umls \\
    --output data/reference/snomed_diabetes_subset.json \\
    --max-depth 4
```

**Option C: Auto-detect format**

```bash
python scripts/extract_snomed_diabetes.py \\
    data/raw_snomed/Snapshot/Terminology \\
    --format auto \\
    --output data/reference/snomed_diabetes_subset.json
```

**Output**: `data/reference/snomed_diabetes_subset.json` (~200-300 concepts)

**What it does**:
- **SNOMED native format**:
  - Processes sct2_Description_Snapshot*.txt **line-by-line**
  - Optionally processes sct2_Relationship_Snapshot*.txt for hierarchy
  - Uses SNOMED concept IDs directly (73211009, 44054006, 46635009)
  - Extracts FSN (Fully Specified Names) and synonyms
- **UMLS format**:
  - Processes MRCONSO.RRF (481 MB) **line-by-line**
  - Optionally processes MRHIER.RRF for hierarchy
  - Uses CUIs (C0011847, C0011860, C0011854)
  - Adds semantic types from MRSTY.RRF if available
- Follows is_a hierarchy to find all descendant concepts
- **Never loads entire files into memory** (safe for large files)

**Options**:
- `--format snomed|umls|auto`: Input format (default: auto-detect)
- `--max-depth N`: Limit hierarchy traversal depth (default: 4)
- `--no-hierarchy`: Skip hierarchy expansion, use seed concepts only (faster, fewer concepts)

### Step 4: Pre-compute SNOMED Embeddings (Phase 1)

This script will be run after implementing the embedding factory:

```bash
python scripts/precompute_embeddings.py \\
    --snomed-data data/reference/snomed_diabetes_subset.json \\
    --output-dir data/reference/snomed_embeddings \\
    --models sapbert biobert pubmedbert
```

**Output**: `data/reference/snomed_embeddings/{model}.npz` for each embedding model

**What it does**:
- Embeds all SNOMED concept display names using each embedding model
- Stores embeddings as .npz files (numpy compressed format)
- Used by SemanticResolver for fast cosine similarity search

## Verification

After running all scripts, verify the output:

```bash
ls -lh data/reference/
```

Expected files:
- `icd10_diabetes_codes.json` (~50 KB)
- `snomed_icd10_crosswalk.json` (~500 KB - 2 MB depending on filtering)
- `snomed_diabetes_subset.json` (~500 KB - 1 MB)
- `snomed_embeddings/` (created in Phase 1)

## Troubleshooting

### UMLS account not approved yet
- Use only Step 1 (ICD-10 parsing) for now
- Steps 2-3 require SNOMED data from UMLS
- Apply for UMLS account at https://uts.nlm.nih.gov → "Sign Up"
- Approval typically takes 1-3 business days

### File not found errors
- Check file paths match your downloaded files
- UMLS files must be in `META/` subdirectory
- Use `--help` on any script to see options

### Memory issues
- Scripts process large files line-by-line, not loading full files
- If still encountering issues, increase max-depth limit or use --no-hierarchy

### ICD-10 table format not recognized
- Try `--format format2` for structured text format
- Try `--format format1` for tab-delimited format
- Use `--format auto` (default) to try both

### Crosswalk file format issues
- Script auto-detects tab or comma delimiter
- Ensure file has headers: `referencedComponentId`, `mapTarget`, etc.
- Tab-delimited .txt format (from SNOMED CT US Edition) is fully supported
- Comma-delimited .csv format (from UMLS) is also supported

## Next Steps

After running these scripts:
1. Implement embedding factory and models (Phase 1)
2. Run `precompute_embeddings.py` to generate embedding files
3. Begin building Approach 3 pipeline (Phase 2)

See `clinical-icd10-project-plan.md` for full execution plan.
