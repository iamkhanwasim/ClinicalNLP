"""
Extract diabetes-related SNOMED CT concepts from SNOMED CT files.

This script processes SNOMED CT terminology files line-by-line to extract
diabetes concepts. NEVER loads entire files into memory.

Supports two input formats:

1. **SNOMED CT US Edition (native format)** - Recommended
   - Directory: .../Snapshot/Terminology/
   - Required file: sct2_Description_Snapshot-en_US1000124_YYYYMMDD.txt
   - Optional file: sct2_Relationship_Snapshot-en_US1000124_YYYYMMDD.txt
   - Tab-delimited format
   - Uses SNOMED concept IDs directly (e.g., 73211009 for Diabetes mellitus)
   - Extracts display names, synonyms, and FSN (Fully Specified Names)
   - Follows is_a hierarchy if Relationship file provided

2. **UMLS Metathesaurus (alternative)**
   - Directory: .../META/
   - Required file: MRCONSO.RRF
   - Optional files: MRHIER.RRF, MRSTY.RRF
   - Uses CUIs mapped to SNOMED codes
   - Larger files (481 MB+)

Output: data/reference/snomed_diabetes_subset.json

Process:
1. Start with root diabetes concepts (SNOMED IDs or UMLS CUIs)
2. Optional: follow is_a hierarchy to find descendants (max depth: 4)
3. Read terminology file line-by-line
4. Extract ~200-300 diabetes concepts with synonyms

Usage:
  # SNOMED CT native format
  python extract_snomed_diabetes.py /path/to/Snapshot/Terminology --format snomed

  # UMLS format
  python extract_snomed_diabetes.py /path/to/META --format umls

  # Auto-detect format
  python extract_snomed_diabetes.py /path/to/data --format auto
"""

import json
from pathlib import Path
from typing import Dict, Set, List
from collections import defaultdict
import argparse


# Root diabetes SNOMED CT concept IDs
ROOT_DIABETES_CONCEPTS = {
    '73211009',   # Diabetes mellitus
    '44054006',   # Type 2 diabetes mellitus
    '46635009',   # Type 1 diabetes mellitus
}

# Additional seed concepts for diabetes complications and related conditions
SEED_CONCEPTS = {
    # Complications
    '420662003',  # Diabetic ketoacidosis
    '127013003',  # Diabetic nephropathy
    '230572002',  # Diabetic neuropathy
    '4855003',    # Diabetic retinopathy
    '609566009',  # Diabetic foot ulcer
    '302821008',  # Diabetic peripheral angiopathy
    '421326000',  # Diabetic polyneuropathy
    '420279001',  # Diabetic chronic kidney disease

    # Related conditions
    '80394007',   # Hyperglycemia
    '302866003',  # Hypoglycemia
    '359642000',  # Uncontrolled diabetes mellitus

    # Medications (for inference rules)
    '325072002',  # Insulin
    '109081006',  # Metformin
    '386964003',  # Glucophage (brand name)
}

# UMLS CUI equivalents (for UMLS format compatibility)
UMLS_ROOT_CUIS = {
    'C0011847',   # Diabetes Mellitus
    'C0011860',   # Type 2 Diabetes Mellitus (NIDDM)
    'C0011854',   # Type 1 Diabetes Mellitus (IDDM)
}


def parse_rrf_line(line: str) -> List[str]:
    """Parse pipe-delimited RRF line."""
    return [field.strip() for field in line.strip().split('|')]


def parse_snomed_description_file(
    description_path: Path,
    target_concept_ids: Set[str]
) -> Dict[str, Dict]:
    """
    Extract SNOMED concepts from Description file (SNOMED CT native format).

    File format (tab-delimited):
    id    effectiveTime    active    moduleId    conceptId    languageCode    typeId    term    caseSignificanceId

    Key columns:
    - conceptId: SNOMED CT concept ID (e.g., 73211009)
    - term: Display name or synonym
    - active: 1 = active, 0 = inactive
    - languageCode: en for English
    - typeId: 900000000000003001 = FSN (Fully Specified Name)
              900000000000013009 = Synonym

    Args:
        description_path: Path to Description file
        target_concept_ids: Set of SNOMED concept IDs to extract

    Returns:
        Dictionary mapping concept IDs to concept data
    """
    concepts = {}
    synonyms = defaultdict(list)
    fsn_map = {}  # Fully Specified Names
    line_count = 0
    match_count = 0

    print(f"Processing SNOMED CT Description file...")
    print(f"Target concepts: {len(target_concept_ids)}")

    # Type IDs
    FSN_TYPE = '900000000000003001'
    SYNONYM_TYPE = '900000000000013009'

    with open(description_path, 'r', encoding='utf-8', errors='ignore') as f:
        # Read header
        header_line = f.readline()
        if 'conceptId' not in header_line:
            print("Warning: Header not found, assuming first line is data")
            f.seek(0)

        for line in f:
            line_count += 1

            if line_count % 1000000 == 0:
                print(f"  Processed {line_count//1000000}M lines, found {match_count} descriptions")

            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue

            # Parse fields
            active = fields[2]
            concept_id = fields[4]
            language_code = fields[5]
            type_id = fields[6]
            term = fields[7]

            # Filter: active English terms for target concepts
            if active != '1' or language_code != 'en':
                continue

            if concept_id in target_concept_ids:
                match_count += 1

                # Store FSN as primary display
                if type_id == FSN_TYPE:
                    fsn_map[concept_id] = term

                # Collect all terms as synonyms
                if term not in synonyms[concept_id]:
                    synonyms[concept_id].append(term)

                # Create concept entry if doesn't exist
                if concept_id not in concepts:
                    concepts[concept_id] = {
                        'cui': concept_id,  # Use concept_id as CUI for compatibility
                        'snomed_code': concept_id,
                        'display': term,  # Will be updated with FSN if available
                        'source': 'SNOMEDCT_US'
                    }

    print(f"  Total lines processed: {line_count}")
    print(f"  Descriptions found: {match_count}")
    print(f"  Unique concepts: {len(concepts)}")

    # Update display names with FSN
    for concept_id, concept in concepts.items():
        if concept_id in fsn_map:
            concept['display'] = fsn_map[concept_id]
        concept['synonyms'] = synonyms[concept_id]

    return concepts


def extract_from_mrconso(
    mrconso_path: Path,
    target_cuis: Set[str],
    sab_filter: str = "SNOMEDCT_US"
) -> Dict[str, Dict]:
    """
    Extract SNOMED concepts from MRCONSO.RRF line-by-line.

    MRCONSO.RRF format (pipe-delimited):
    CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS|CVF

    Key columns:
    - 0: CUI (Concept Unique Identifier)
    - 11: SAB (Source Abbreviation) - filter to "SNOMEDCT_US"
    - 13: CODE (SNOMED CT code)
    - 14: STR (String/display name)
    - 3: LUI (Lexical Unique Identifier)
    - 12: TTY (Term Type) - PT = preferred term

    Args:
        mrconso_path: Path to MRCONSO.RRF
        target_cuis: Set of CUIs to extract
        sab_filter: Source to filter (default: SNOMEDCT_US)

    Returns:
        Dictionary mapping CUI to concept data
    """
    concepts = {}
    synonyms = defaultdict(list)
    line_count = 0
    match_count = 0

    print(f"Processing MRCONSO.RRF...")
    print(f"Target CUIs: {len(target_cuis)}")

    with open(mrconso_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_count += 1

            if line_count % 1000000 == 0:
                print(f"  Processed {line_count//1000000}M lines, found {match_count} matches")

            fields = parse_rrf_line(line)
            if len(fields) < 15:
                continue

            cui = fields[0]
            sab = fields[11]
            snomed_code = fields[13]
            display_name = fields[14]
            tty = fields[12]  # Term type

            # Filter to target CUIs and SNOMED source
            if cui in target_cuis and sab == sab_filter:
                match_count += 1

                # Store preferred term as main concept
                if tty == 'PT' or cui not in concepts:
                    concepts[cui] = {
                        'cui': cui,
                        'snomed_code': snomed_code,
                        'display': display_name,
                        'source': sab
                    }

                # Collect synonyms
                if display_name not in synonyms[cui]:
                    synonyms[cui].append(display_name)

    print(f"  Total lines processed: {line_count}")
    print(f"  Concepts found: {len(concepts)}")

    # Add synonyms to concepts
    for cui, concept in concepts.items():
        concept['synonyms'] = synonyms[cui]

    return concepts


def extract_children_from_mrhier(
    mrhier_path: Path,
    root_cuis: Set[str],
    sab_filter: str = "SNOMEDCT_US",
    max_depth: int = 4
) -> Set[str]:
    """
    Follow is_a hierarchy in MRHIER.RRF to find all descendant concepts.

    MRHIER.RRF format (pipe-delimited):
    CUI|AUI|CXN|PAUI|SAB|RELA|PTR|HCD|CVF

    Key columns:
    - 0: CUI
    - 4: SAB (Source) - filter to "SNOMEDCT_US"
    - 6: PTR (Path to root) - hierarchical path

    Args:
        mrhier_path: Path to MRHIER.RRF
        root_cuis: Starting CUIs
        sab_filter: Source filter
        max_depth: Maximum depth to traverse (prevent explosion)

    Returns:
        Set of all descendant CUIs
    """
    all_cuis = set(root_cuis)
    line_count = 0

    print(f"\nProcessing MRHIER.RRF...")
    print(f"Starting with {len(root_cuis)} root CUIs")

    with open(mrhier_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_count += 1

            if line_count % 1000000 == 0:
                print(f"  Processed {line_count//1000000}M lines, found {len(all_cuis)} concepts")

            fields = parse_rrf_line(line)
            if len(fields) < 7:
                continue

            cui = fields[0]
            sab = fields[4]
            ptr = fields[6]  # Path to root

            if sab != sab_filter:
                continue

            # Check if any root CUI appears in the path
            ptr_cuis = ptr.split('.')
            if any(root_cui in ptr_cuis for root_cui in root_cuis):
                # Check depth
                depth = len(ptr_cuis)
                if depth <= max_depth:
                    all_cuis.add(cui)

    print(f"  Total lines processed: {line_count}")
    print(f"  Total descendant concepts: {len(all_cuis)}")

    return all_cuis


def add_semantic_types(concepts: Dict, mrsty_path: Path = None) -> Dict:
    """
    Add semantic types from MRSTY.RRF if available.

    MRSTY.RRF format:
    CUI|TUI|STN|STY|ATUI|CVF

    Key columns:
    - 0: CUI
    - 3: STY (Semantic Type)

    Args:
        concepts: Concept dictionary
        mrsty_path: Path to MRSTY.RRF (optional)

    Returns:
        Updated concepts with semantic types
    """
    if not mrsty_path or not mrsty_path.exists():
        print("\n  MRSTY.RRF not provided, skipping semantic types")
        return concepts

    print(f"\nAdding semantic types from MRSTY.RRF...")
    target_cuis = set(concepts.keys())
    match_count = 0

    with open(mrsty_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            fields = parse_rrf_line(line)
            if len(fields) < 4:
                continue

            cui = fields[0]
            semantic_type = fields[3]

            if cui in target_cuis:
                if 'semantic_types' not in concepts[cui]:
                    concepts[cui]['semantic_types'] = []
                concepts[cui]['semantic_types'].append(semantic_type)
                match_count += 1

    print(f"  Added semantic types to {match_count} concepts")
    return concepts


def parse_snomed_relationship_file(
    relationship_path: Path,
    root_concept_ids: Set[str],
    max_depth: int = 4
) -> Set[str]:
    """
    Follow is_a hierarchy in SNOMED CT Relationship file.

    File format (tab-delimited):
    id    effectiveTime    active    moduleId    sourceId    destinationId
    relationshipGroup    typeId    characteristicTypeId    modifierId

    Key columns:
    - sourceId: Child concept
    - destinationId: Parent concept
    - typeId: 116680003 = "Is a" relationship
    - active: 1 = active

    Args:
        relationship_path: Path to Relationship file
        root_concept_ids: Starting concept IDs
        max_depth: Maximum depth to traverse

    Returns:
        Set of all descendant concept IDs
    """
    all_concepts = set(root_concept_ids)
    parent_to_children = defaultdict(set)
    line_count = 0

    print(f"\nProcessing SNOMED CT Relationship file...")
    print(f"Starting with {len(root_concept_ids)} root concepts")

    IS_A_TYPE = '116680003'

    # Build parent->child map for is_a relationships
    with open(relationship_path, 'r', encoding='utf-8', errors='ignore') as f:
        # Read header
        header_line = f.readline()
        if 'sourceId' not in header_line:
            print("Warning: Header not found, assuming first line is data")
            f.seek(0)

        for line in f:
            line_count += 1

            if line_count % 1000000 == 0:
                print(f"  Processed {line_count//1000000}M lines...")

            fields = line.strip().split('\t')
            if len(fields) < 10:
                continue

            active = fields[2]
            source_id = fields[4]  # Child
            destination_id = fields[5]  # Parent
            type_id = fields[7]

            # Filter: active is_a relationships
            if active != '1' or type_id != IS_A_TYPE:
                continue

            parent_to_children[destination_id].add(source_id)

    print(f"  Built hierarchy map from {line_count} lines")
    print(f"  Found {len(parent_to_children)} parent concepts with children")

    # Traverse hierarchy breadth-first
    queue = [(concept_id, 0) for concept_id in root_concept_ids]
    visited = set(root_concept_ids)

    while queue:
        current_id, depth = queue.pop(0)

        if depth >= max_depth:
            continue

        # Get children of current concept
        children = parent_to_children.get(current_id, set())
        for child_id in children:
            if child_id not in visited:
                visited.add(child_id)
                all_concepts.add(child_id)
                queue.append((child_id, depth + 1))

    print(f"  Total descendant concepts: {len(all_concepts)}")
    return all_concepts


def main():
    parser = argparse.ArgumentParser(
        description='Extract diabetes SNOMED concepts',
        epilog="""
Supports two input formats:

1. SNOMED CT US Edition (native format) - Recommended:
   python extract_snomed_diabetes.py /path/to/Snapshot/Terminology --format snomed

2. UMLS Metathesaurus:
   python extract_snomed_diabetes.py /path/to/META --format umls
        """
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Path to SNOMED Terminology directory or UMLS META directory'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['auto', 'snomed', 'umls'],
        default='auto',
        help='Input format: snomed (native), umls (MRCONSO), or auto-detect'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/reference/snomed_diabetes_subset.json',
        help='Output JSON file'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=4,
        help='Maximum hierarchy depth to traverse (default: 4)'
    )
    parser.add_argument(
        '--no-hierarchy',
        action='store_true',
        help='Skip hierarchy expansion, use only seed concepts'
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not input_dir.exists():
        print(f"✗ Input directory not found: {input_dir}")
        return 1

    # Auto-detect format if needed
    format_type = args.format
    if format_type == 'auto':
        # Check for UMLS files
        if (input_dir / "MRCONSO.RRF").exists():
            format_type = 'umls'
            print(f"Auto-detected format: UMLS Metathesaurus")
        # Check for SNOMED CT files
        elif list(input_dir.glob("sct2_Description_Snapshot*.txt")):
            format_type = 'snomed'
            print(f"Auto-detected format: SNOMED CT US Edition")
        else:
            print(f"✗ Could not auto-detect format")
            print(f"\nLooking for either:")
            print(f"  UMLS: MRCONSO.RRF in {input_dir}")
            print(f"  SNOMED: sct2_Description_Snapshot*.txt in {input_dir}")
            return 1

    print(f"{'='*60}")
    print(f"SNOMED Diabetes Concept Extraction")
    print(f"{'='*60}")
    print(f"Input Directory: {input_dir}")
    print(f"Format: {format_type.upper()}")
    print(f"Output: {output_path}\n")

    # Process based on format
    if format_type == 'snomed':
        # Native SNOMED CT format
        print("Processing SNOMED CT US Edition files...\n")

        # Find Description file
        description_files = list(input_dir.glob("sct2_Description_Snapshot*.txt"))
        if not description_files:
            print(f"✗ Description file not found: sct2_Description_Snapshot*.txt")
            return 1
        description_path = description_files[0]

        # Find Relationship file (optional, for hierarchy)
        relationship_files = list(input_dir.glob("sct2_Relationship_Snapshot*.txt"))
        relationship_path = relationship_files[0] if relationship_files else None

        print(f"Found Description file: {description_path.name}")
        if relationship_path:
            print(f"Found Relationship file: {relationship_path.name}\n")
        else:
            print(f"Relationship file not found (hierarchy expansion disabled)\n")

        # Step 1: Expand concept set using hierarchy (if enabled)
        all_diabetes_concepts = ROOT_DIABETES_CONCEPTS | SEED_CONCEPTS

        if not args.no_hierarchy and relationship_path:
            all_diabetes_concepts = parse_snomed_relationship_file(
                relationship_path,
                all_diabetes_concepts,
                max_depth=args.max_depth
            )
        else:
            if args.no_hierarchy:
                print(f"\nSkipping hierarchy expansion (--no-hierarchy)")
            else:
                print(f"\nSkipping hierarchy expansion (Relationship file not found)")
            print(f"Using {len(all_diabetes_concepts)} seed concepts only")

        # Step 2: Extract concepts from Description file
        concepts = parse_snomed_description_file(description_path, all_diabetes_concepts)

    elif format_type == 'umls':
        # UMLS Metathesaurus format
        print("Processing UMLS Metathesaurus files...\n")

        mrconso_path = input_dir / "MRCONSO.RRF"
        mrhier_path = input_dir / "MRHIER.RRF"
        mrsty_path = input_dir / "MRSTY.RRF"

        if not mrconso_path.exists():
            print(f"✗ MRCONSO.RRF not found at {mrconso_path}")
            print(f"\nExpected UMLS directory structure:")
            print(f"  {input_dir}/")
            print(f"    MRCONSO.RRF  (481 MB)")
            print(f"    MRHIER.RRF   (367 MB)")
            print(f"    MRSTY.RRF    (optional)")
            return 1

        # Step 1: Expand CUI set using hierarchy (if enabled)
        all_diabetes_cuis = set(UMLS_ROOT_CUIS)

        if not args.no_hierarchy and mrhier_path.exists():
            all_diabetes_cuis = extract_children_from_mrhier(
                mrhier_path,
                all_diabetes_cuis,
                max_depth=args.max_depth
            )
        else:
            print(f"\nSkipping hierarchy expansion")
            print(f"Using {len(all_diabetes_cuis)} seed CUIs only")

        # Step 2: Extract concepts from MRCONSO
        concepts = extract_from_mrconso(mrconso_path, all_diabetes_cuis)

        # Step 3: Add semantic types (optional)
        if mrsty_path.exists():
            concepts = add_semantic_types(concepts, mrsty_path)
    else:
        print(f"✗ Unknown format: {format_type}")
        return 1

    # Convert to list
    concepts_list = list(concepts.values())

    # Sort by display name
    concepts_list.sort(key=lambda x: x['display'])

    # Statistics
    print(f"\n{'='*60}")
    print(f"Extraction Results")
    print(f"{'='*60}")
    print(f"Total concepts extracted: {len(concepts_list)}")

    # Count by semantic type (if available)
    if concepts_list and 'semantic_types' in concepts_list[0]:
        all_types = set()
        for c in concepts_list:
            all_types.update(c.get('semantic_types', []))
        print(f"Unique semantic types: {len(all_types)}")

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare metadata based on format
    if format_type == 'snomed':
        metadata = {
            'description': 'SNOMED CT diabetes concepts extracted from SNOMED CT US Edition',
            'source': 'SNOMEDCT_US',
            'format': 'native',
            'root_concepts': list(ROOT_DIABETES_CONCEPTS),
            'total_concepts': len(concepts_list),
            'extraction_method': 'hierarchy' if not args.no_hierarchy else 'seeds_only',
            'max_depth': args.max_depth if not args.no_hierarchy else None
        }
    else:  # umls
        metadata = {
            'description': 'SNOMED CT diabetes concepts extracted from UMLS',
            'umls_version': '2025AB',
            'source': 'SNOMEDCT_US',
            'format': 'umls',
            'root_cuis': list(UMLS_ROOT_CUIS),
            'total_concepts': len(concepts_list),
            'extraction_method': 'hierarchy' if not args.no_hierarchy else 'seeds_only',
            'max_depth': args.max_depth if not args.no_hierarchy else None
        }

    output_data = {
        'metadata': metadata,
        'concepts': concepts_list
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_path}")

    # Show sample concepts
    print(f"\nSample concepts (first 10):")
    for concept in concepts_list[:10]:
        synonyms_count = len(concept.get('synonyms', []))
        print(f"  {concept['cui']} | {concept['snomed_code']:<12} | {concept['display'][:50]}")
        print(f"    ({synonyms_count} synonyms)")

    return 0


if __name__ == '__main__':
    exit(main())
