"""
Extract diabetes-related ICD-10 codes from the official ICD-10-CM table.

Filters to codes relevant for the diabetes POC:
- E08-E13: All diabetes codes
- Z79.4, Z79.84: Long-term medication use
- G57.x, G63.x: Neuropathy codes
- N18.x: Chronic kidney disease
- I25.x, I10: Cardiovascular codes
- Z94.0: Transplant status
- E78.x: Lipid disorders
- M67.x: Ganglion cysts
- R73.x, R20.x: Symptoms

Input: ICD-10-CM table file from CMS/CDC
  Format: Tab-delimited flat file, one code per line
  Example: E1165<tab>Type 2 diabetes mellitus with hyperglycemia
  Note: Codes have NO dots (E1165 not E11.65) - script adds them automatically

Output: data/reference/icd10_diabetes_codes.json
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Set
import argparse


# Diabetes-relevant code prefixes
RELEVANT_PREFIXES = {
    'E08', 'E09', 'E10', 'E11', 'E12', 'E13',  # Diabetes
    'Z79.4', 'Z79.84',  # Medication use
    'G57', 'G63',  # Neuropathy
    'N18',  # CKD
    'I25', 'I10',  # Cardiovascular
    'Z94.0',  # Transplant
    'E78',  # Lipid disorders
    'M67',  # Ganglion
    'R73', 'R20',  # Symptoms
}

# HCC (Hierarchical Condition Category) patterns
# These are high-severity codes used for risk adjustment
HCC_PATTERNS = [
    r'^E10\.',  # Type 1 DM with complications
    r'^E11\.2',  # Type 2 DM with kidney complications
    r'^E11\.3',  # Type 2 DM with eye complications
    r'^E11\.4',  # Type 2 DM with neurological complications
    r'^E11\.5',  # Type 2 DM with circulatory complications
    r'^E11\.6',  # Type 2 DM with other specified complications
    r'^E13\.',  # Other specified DM with complications
    r'^N18\.',  # CKD
    r'^I25\.',  # Chronic ischemic heart disease
    r'^Z94\.0',  # Kidney transplant status
]


def is_relevant_code(code: str) -> bool:
    """Check if code matches our diabetes-relevant prefixes."""
    for prefix in RELEVANT_PREFIXES:
        if code.startswith(prefix):
            return True
    return False


def is_hcc_code(code: str) -> bool:
    """Check if code is an HCC (high-severity) code."""
    for pattern in HCC_PATTERNS:
        if re.match(pattern, code):
            return True
    return False


def get_code_category(code: str) -> str:
    """Extract the category (first 3 characters or specific Z code)."""
    if code.startswith('Z'):
        # Z codes: keep full code pattern
        return code[:6] if '.' in code else code[:3]
    return code[:3]


def parse_icd10_table_format1(file_path: Path) -> List[Dict]:
    """
    Parse ICD-10-CM table - Format 1 (tab-delimited with billable flag).

    Expected format: CODE\tDESCRIPTION\tBILLABLE

    Handles codes with or without dots.
    """
    codes = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            if len(parts) < 2:
                continue

            raw_code = parts[0].strip()
            description = parts[1].strip()
            billable = parts[2].strip().lower() == 'true' if len(parts) > 2 else True

            # Format code (add dot if needed)
            code = format_icd10_code(raw_code)

            if is_relevant_code(code):
                codes.append({
                    'code': code,
                    'display': description,
                    'billable': billable,
                    'category': get_code_category(code),
                    'hcc': is_hcc_code(code)
                })

    return codes


def format_icd10_code(raw_code: str) -> str:
    """
    Format ICD-10 code by inserting dot after position 3.

    CMS flat files have codes without dots (e.g., E1165, Z794).

    Rules:
    - Codes with 3 characters: no dot needed (category level, e.g., E11)
    - Codes with 4+ characters: insert dot after position 3 (e.g., E1165 → E11.65)

    Args:
        raw_code: Code without dots (e.g., E1165)

    Returns:
        Formatted code with dot (e.g., E11.65)
    """
    raw_code = raw_code.strip()
    if len(raw_code) <= 3:
        return raw_code
    return raw_code[:3] + "." + raw_code[3:]


def parse_icd10_table_format2(file_path: Path) -> List[Dict]:
    """
    Parse ICD-10-CM table - Format 2 (CMS flat file, tab-delimited).

    Expected format from CMS/CDC files:
    code<tab>description

    Example:
    A000    Cholera due to Vibrio cholerae 01, biovar cholerae
    E1165   Type 2 diabetes mellitus with hyperglycemia

    Codes have NO dots and must be formatted.
    """
    codes = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Skip header lines (if any)
            if line.startswith('#') or 'CODE' in line.upper() and line_num == 1:
                continue

            # Try tab-delimited format first
            if '\t' in line:
                parts = line.split('\t', 1)
                if len(parts) >= 2:
                    raw_code = parts[0].strip()
                    description = parts[1].strip()

                    # Format code (add dot)
                    code = format_icd10_code(raw_code)

                    if is_relevant_code(code):
                        # Billable codes are typically 4+ characters
                        billable = len(raw_code) >= 4

                        codes.append({
                            'code': code,
                            'display': description,
                            'billable': billable,
                            'category': get_code_category(code),
                            'hcc': is_hcc_code(code)
                        })
                    continue

            # Fallback: try to parse as space-separated
            match = re.match(r'^([A-Z]\d{2,7})\s+(.+)$', line)
            if match:
                raw_code = match.group(1).strip()
                description = match.group(2).strip()

                # Format code (add dot)
                code = format_icd10_code(raw_code)

                if is_relevant_code(code):
                    billable = len(raw_code) >= 4

                    codes.append({
                        'code': code,
                        'display': description,
                        'billable': billable,
                        'category': get_code_category(code),
                        'hcc': is_hcc_code(code)
                    })

    return codes


def parse_icd10_table(file_path: Path, format_type: str = 'auto') -> List[Dict]:
    """
    Parse ICD-10-CM table file, auto-detecting format.

    Args:
        file_path: Path to ICD-10 table file
        format_type: 'auto', 'format1' (tab-delimited), or 'format2' (structured text)

    Returns:
        List of code dictionaries
    """
    if format_type == 'auto':
        # Try format1 first (tab-delimited)
        try:
            codes = parse_icd10_table_format1(file_path)
            if codes:
                print(f"Parsed using Format 1 (tab-delimited)")
                return codes
        except Exception as e:
            print(f"Format 1 parsing failed: {e}")

        # Try format2 (structured text)
        try:
            codes = parse_icd10_table_format2(file_path)
            if codes:
                print(f"Parsed using Format 2 (structured text)")
                return codes
        except Exception as e:
            print(f"Format 2 parsing failed: {e}")

        raise ValueError("Could not parse ICD-10 table with any known format")

    elif format_type == 'format1':
        return parse_icd10_table_format1(file_path)
    elif format_type == 'format2':
        return parse_icd10_table_format2(file_path)
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def deduplicate_codes(codes: List[Dict]) -> List[Dict]:
    """Remove duplicate codes, keeping the most complete entry."""
    seen = {}
    for code_data in codes:
        code = code_data['code']
        if code not in seen:
            seen[code] = code_data
        else:
            # Keep entry with longer description
            if len(code_data['display']) > len(seen[code]['display']):
                seen[code] = code_data

    return list(seen.values())


def main():
    parser = argparse.ArgumentParser(
        description='Extract diabetes-related ICD-10-CM codes from CMS flat file to JSON',
        epilog="""
Example usage:
  python build_icd10_reference.py data/raw_umls/icd10cm-codes-April-1-2026.txt

Expected input format (CMS flat file):
  Tab-delimited: code<tab>description
  Codes without dots: E1165 (not E11.65)
  Example line: E1165    Type 2 diabetes mellitus with hyperglycemia

  Script automatically inserts dots: E1165 → E11.65

Download from: https://www.cdc.gov/nchs/icd/icd-10-cm/files.html
File typically named: icd10cm-codes-{date}.txt
        """
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to ICD-10-CM flat file (.txt, tab-delimited)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/reference/icd10_diabetes_codes.json',
        help='Output JSON file path (default: data/reference/icd10_diabetes_codes.json)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['auto', 'format1', 'format2'],
        default='auto',
        help='Input file format (default: auto-detect). format2=CMS flat file'
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"✗ Input file not found: {input_path}")
        print(f"\nExpected file: ICD-10-CM flat file (tab-delimited)")
        print(f"Typical name: icd10cm-codes-April-1-2026.txt")
        print(f"Format: code<tab>description (codes without dots)")
        print(f"Download from: https://www.cdc.gov/nchs/icd/icd-10-cm/files.html")
        return 1

    print(f"Processing: {input_path}")
    print(f"Output: {output_path}\n")

    # Parse codes
    codes = parse_icd10_table(input_path, args.format)

    # Deduplicate
    codes = deduplicate_codes(codes)

    # Sort by code
    codes.sort(key=lambda x: x['code'])

    # Statistics
    print(f"\n{'='*60}")
    print(f"Extraction Results")
    print(f"{'='*60}")
    print(f"Total codes extracted: {len(codes)}")

    # Count by category
    categories = {}
    for code in codes:
        cat = code['category']
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\nCodes by category:")
    for cat in sorted(categories.keys()):
        print(f"  {cat}: {categories[cat]}")

    # Count HCC codes
    hcc_count = sum(1 for c in codes if c['hcc'])
    print(f"\nHCC (high-severity) codes: {hcc_count}")

    # Count billable codes
    billable_count = sum(1 for c in codes if c['billable'])
    print(f"Billable codes: {billable_count}")

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'description': 'ICD-10-CM codes relevant to diabetes POC',
            'source_file': str(input_path.name),
            'focus': 'Diabetes (E08-E13) and related conditions',
            'total_codes': len(codes),
            'hcc_codes': hcc_count,
            'billable_codes': billable_count
        },
        'codes': codes
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_path}")

    # Show some sample codes
    print(f"\nSample codes:")
    for code in codes[:10]:
        hcc_mark = " [HCC]" if code['hcc'] else ""
        print(f"  {code['code']:<10} {code['display'][:60]}{hcc_mark}")

    return 0


def test_code_formatting():
    """Test the ICD-10 code formatting function."""
    test_cases = [
        ("E11", "E11"),           # 3 chars, no dot
        ("E1165", "E11.65"),      # 5 chars, add dot
        ("Z794", "Z79.4"),        # 4 chars, add dot
        ("I10", "I10"),           # 3 chars, no dot
        ("N186", "N18.6"),        # 4 chars, add dot
        ("E1165789", "E11.65789"), # 8 chars, add dot
    ]

    print("Testing ICD-10 code formatting:")
    all_pass = True
    for raw, expected in test_cases:
        result = format_icd10_code(raw)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_pass = False
        print(f"  {status} {raw} → {result} (expected: {expected})")

    return all_pass


if __name__ == '__main__':
    # Quick test if run with --test argument
    import sys
    if '--test' in sys.argv:
        if test_code_formatting():
            print("\nAll tests passed")
            exit(0)
        else:
            print("\n✗ Some tests failed")
            exit(1)

    exit(main())
