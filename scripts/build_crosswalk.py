"""
Parse SNOMED CT → ICD-10-CM crosswalk CSV to JSON.

Input: snomed_icd_10_map.csv_0_0_0.csv (SNOMED→ICD-10 mapping file)
Output: data/reference/snomed_icd10_crosswalk.json

The crosswalk file maps SNOMED CT concept IDs to ICD-10-CM codes.
Multiple ICD-10 codes may map to a single SNOMED concept.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import argparse


def parse_crosswalk_csv(file_path: Path) -> Dict[str, List[Dict]]:
    """
    Parse SNOMED→ICD-10 crosswalk file (CSV or tab-delimited TXT).

    Expected columns:
    - referencedComponentId: SNOMED CT concept ID
    - mapTarget: ICD-10-CM code
    - mapGroup: Grouping for multiple mappings
    - mapPriority: Priority within group (1 = highest)
    - mapRule: Optional mapping rule/condition
    - mapAdvice: Additional guidance
    - active: Whether mapping is active (1 = active)

    Standard format (SNOMED CT to ICD-10-CM Map):
    id    effectiveTime    active    moduleId    refsetId    referencedComponentId
    mapGroup    mapPriority    mapRule    mapAdvice    mapTarget    correlationId    mapCategoryId

    Returns:
        Dictionary mapping SNOMED IDs to list of ICD-10 mappings
    """
    crosswalk = defaultdict(list)
    row_count = 0
    error_count = 0
    skipped_inactive = 0

    # Detect delimiter (tab or comma)
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        delimiter = '\t' if '\t' in first_line else ','

    print(f"Detected delimiter: {'TAB' if delimiter == '\t' else 'COMMA'}\n")

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        # Get column names
        columns = reader.fieldnames
        print(f"File columns: {columns}\n")

        # Map column names (handle variations)
        snomed_col = None
        icd10_col = None
        active_col = None

        for col in columns:
            col_lower = col.lower()
            if 'referencedcomponent' in col_lower or col_lower == 'snomed_code':
                snomed_col = col
            elif 'maptarget' in col_lower or col_lower == 'icd10_code':
                icd10_col = col
            elif col_lower == 'active':
                active_col = col

        if not snomed_col or not icd10_col:
            raise ValueError(
                f"Could not identify SNOMED and ICD-10 columns. "
                f"Found columns: {columns}"
            )

        print(f"Using columns:")
        print(f"  SNOMED: {snomed_col}")
        print(f"  ICD-10: {icd10_col}")
        if active_col:
            print(f"  Active: {active_col}")
        print()

        for row in reader:
            row_count += 1

            if row_count % 100000 == 0:
                print(f"  Processed {row_count} rows, found {len(crosswalk)} SNOMED concepts...")

            try:
                # Skip inactive mappings
                if active_col and row.get(active_col, '1').strip() == '0':
                    skipped_inactive += 1
                    continue

                snomed_id = row[snomed_col].strip()
                icd10_code = row[icd10_col].strip()

                if not snomed_id or not icd10_code:
                    continue

                # Skip invalid ICD-10 codes
                if icd10_code in ('NULL', 'NO MAP', '') or len(icd10_code) < 3:
                    continue

                # Extract additional fields if available
                map_group = row.get('mapGroup', '1').strip()
                map_priority = row.get('mapPriority', '1').strip()
                map_rule = row.get('mapRule', '').strip()
                map_advice = row.get('mapAdvice', '').strip()

                # Convert to integers, default to 1
                try:
                    map_group = int(map_group) if map_group else 1
                except ValueError:
                    map_group = 1

                try:
                    map_priority = int(map_priority) if map_priority else 1
                except ValueError:
                    map_priority = 1

                mapping = {
                    'icd10_code': icd10_code,
                    'map_group': map_group,
                    'map_priority': map_priority
                }

                if map_rule and map_rule != 'TRUE':
                    mapping['map_rule'] = map_rule
                if map_advice:
                    mapping['map_advice'] = map_advice

                crosswalk[snomed_id].append(mapping)

            except Exception as e:
                error_count += 1
                if error_count <= 5:  # Show first few errors
                    print(f"Warning: Error parsing row {row_count}: {e}")

        print(f"\nProcessed {row_count} rows")
        if skipped_inactive:
            print(f"Skipped {skipped_inactive} inactive mappings")
        if error_count:
            print(f"Errors: {error_count}")

    return dict(crosswalk)


def filter_diabetes_crosswalk(crosswalk: Dict, diabetes_codes: set) -> Dict:
    """
    Filter crosswalk to only diabetes-related ICD-10 codes.

    Args:
        crosswalk: Full crosswalk dictionary
        diabetes_codes: Set of diabetes-related ICD-10 codes

    Returns:
        Filtered crosswalk
    """
    filtered = {}

    for snomed_id, mappings in crosswalk.items():
        diabetes_mappings = [
            m for m in mappings
            if any(m['icd10_code'].startswith(code[:3]) for code in diabetes_codes)
        ]

        if diabetes_mappings:
            filtered[snomed_id] = diabetes_mappings

    return filtered


def sort_mappings(crosswalk: Dict) -> Dict:
    """Sort mappings by priority within each SNOMED concept."""
    for snomed_id in crosswalk:
        crosswalk[snomed_id].sort(
            key=lambda m: (m['map_group'], m['map_priority'])
        )
    return crosswalk


def main():
    parser = argparse.ArgumentParser(
        description='Parse SNOMED CT to ICD-10-CM crosswalk file to JSON',
        epilog="""
Example usage:
  python build_crosswalk.py snomed_icd10_map.txt --filter-diabetes
  python build_crosswalk.py snomed_icd_10_map.csv_0_0_0.csv --output data/reference/crosswalk.json

Supported formats:
  - Tab-delimited .txt (standard SNOMED CT to ICD-10-CM Map format)
  - Comma-delimited .csv
        """
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to SNOMED→ICD-10 crosswalk file (.txt or .csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/reference/snomed_icd10_crosswalk.json',
        help='Output JSON file path'
    )
    parser.add_argument(
        '--filter-diabetes',
        action='store_true',
        help='Filter to diabetes-related codes only'
    )
    parser.add_argument(
        '--icd10-reference',
        type=str,
        default='data/reference/icd10_diabetes_codes.json',
        help='ICD-10 reference file for filtering'
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print(f"\nExpected file: SNOMED CT to ICD-10-CM Map (.txt or .csv)")
        print(f"Typical filename: snomed_icd_10_map.csv_0_0_0.csv or similar")
        print(f"Download from: https://www.nlm.nih.gov/research/umls/mapping_projects/snomedct_to_icd10cm.html")
        print(f"Or from SNOMED CT United States Edition download")
        return 1

    print(f"{'='*60}")
    print(f"SNOMED→ICD-10 Crosswalk Parser")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}\n")

    # Parse crosswalk
    print("Parsing crosswalk CSV...")
    crosswalk = parse_crosswalk_csv(input_path)

    # Filter to diabetes codes if requested
    if args.filter_diabetes:
        print("\nFiltering to diabetes-related codes...")

        icd10_ref_path = Path(args.icd10_reference)
        if icd10_ref_path.exists():
            with open(icd10_ref_path, 'r') as f:
                icd10_data = json.load(f)
                diabetes_codes = {c['code'] for c in icd10_data['codes']}

            before_count = len(crosswalk)
            crosswalk = filter_diabetes_crosswalk(crosswalk, diabetes_codes)
            after_count = len(crosswalk)

            print(f"  Before filtering: {before_count} SNOMED concepts")
            print(f"  After filtering: {after_count} SNOMED concepts")
        else:
            print(f"  Warning: ICD-10 reference not found at {icd10_ref_path}")
            print(f"  Skipping filtering")

    # Sort mappings by priority
    crosswalk = sort_mappings(crosswalk)

    # Statistics
    print(f"\n{'='*60}")
    print(f"Crosswalk Statistics")
    print(f"{'='*60}")
    print(f"Total SNOMED concepts: {len(crosswalk)}")

    total_mappings = sum(len(mappings) for mappings in crosswalk.values())
    print(f"Total mappings: {total_mappings}")

    avg_mappings = total_mappings / len(crosswalk) if crosswalk else 0
    print(f"Average mappings per concept: {avg_mappings:.2f}")

    # Count many-to-many patterns
    one_to_one = sum(1 for m in crosswalk.values() if len(m) == 1)
    one_to_many = sum(1 for m in crosswalk.values() if len(m) > 1)

    print(f"\nMapping patterns:")
    print(f"  One-to-one: {one_to_one}")
    print(f"  One-to-many: {one_to_many}")

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'description': 'SNOMED CT to ICD-10-CM crosswalk mappings',
            'source_file': input_path.name,
            'total_snomed_concepts': len(crosswalk),
            'total_mappings': total_mappings,
            'filtered_to_diabetes': args.filter_diabetes
        },
        'mappings': crosswalk
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to {output_path}")

    # Show sample mappings
    print(f"\nSample mappings (first 5 SNOMED concepts):")
    for snomed_id in list(crosswalk.keys())[:5]:
        mappings = crosswalk[snomed_id]
        print(f"\n  SNOMED {snomed_id}:")
        for m in mappings[:3]:  # Show up to 3 ICD-10 codes per SNOMED
            print(f"    → {m['icd10_code']} (priority: {m['map_priority']})")
        if len(mappings) > 3:
            print(f"    ... +{len(mappings) - 3} more")

    return 0


if __name__ == '__main__':
    exit(main())
