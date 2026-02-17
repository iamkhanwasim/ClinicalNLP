"""
ICD-10 mapper for converting resolved concepts to ICD-10 codes.

Handles both direct ICD-10 codes and SNOMED→ICD-10 crosswalk mapping.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict
from src.shared.models import ResolvedConcept, ICDCode
from config.settings import ICD10_CODES_PATH, SNOMED_CROSSWALK_PATH


class ICD10Mapper:
    """
    Maps resolved medical concepts to ICD-10 billing codes.

    Supports:
    - Direct ICD-10 codes (from DirectICD10Resolver)
    - SNOMED CT crosswalk mapping (when available)
    - Code validation and enrichment with metadata
    """

    def __init__(
        self,
        icd10_codes_path: str = ICD10_CODES_PATH,
        crosswalk_path: Optional[str] = None
    ):
        """
        Initialize the ICD-10 mapper.

        Args:
            icd10_codes_path: Path to ICD-10 codes JSON file
            crosswalk_path: Optional path to SNOMED→ICD-10 crosswalk
        """
        self.icd10_codes_path = Path(icd10_codes_path)
        self.crosswalk_path = Path(crosswalk_path) if crosswalk_path else None

        # Load ICD-10 codes
        self.codes = self._load_icd10_codes()
        self.code_by_id = {code['code']: code for code in self.codes}

        # Load crosswalk if available
        self.crosswalk = None
        if self.crosswalk_path and self.crosswalk_path.exists():
            self.crosswalk = self._load_crosswalk()
            print(f"Loaded SNOMED→ICD-10 crosswalk: {len(self.crosswalk)} mappings")

        print(f"ICD-10 Mapper initialized with {len(self.codes)} codes")

    def _load_icd10_codes(self) -> List[Dict]:
        """Load ICD-10 codes from JSON file."""
        if not self.icd10_codes_path.exists():
            raise FileNotFoundError(
                f"ICD-10 codes file not found: {self.icd10_codes_path}"
            )

        with open(self.icd10_codes_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data.get('codes', [])

    def _load_crosswalk(self) -> Dict[str, List[Dict]]:
        """
        Load SNOMED→ICD-10 crosswalk mappings.

        Returns:
            Dictionary mapping SNOMED concept IDs to list of ICD-10 codes
        """
        with open(self.crosswalk_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert to efficient lookup structure
        crosswalk = {}
        for mapping in data.get('mappings', []):
            snomed_id = mapping['snomed_code']
            if snomed_id not in crosswalk:
                crosswalk[snomed_id] = []
            crosswalk[snomed_id].append({
                'icd10_code': mapping['icd10_code'],
                'map_group': mapping.get('map_group', 1),
                'map_priority': mapping.get('map_priority', 1),
                'map_rule': mapping.get('map_rule'),
                'map_advice': mapping.get('map_advice')
            })

        return crosswalk

    def map(self, concept: ResolvedConcept) -> List[ICDCode]:
        """
        Map a resolved concept to ICD-10 codes.

        Args:
            concept: ResolvedConcept object

        Returns:
            List of ICDCode objects (may be empty if mapping fails)
        """
        if concept.source == "icd10_direct":
            # Direct ICD-10 code - validate and enrich
            return [self._validate_code(concept.code)]
        elif concept.source == "snomed":
            # SNOMED concept - use crosswalk
            return self._crosswalk_lookup(concept.code)
        else:
            # Unknown source
            return []

    def _validate_code(self, code: str) -> ICDCode:
        """
        Validate and enrich an ICD-10 code with metadata.

        Args:
            code: ICD-10 code string

        Returns:
            ICDCode object with full metadata
        """
        code = code.strip().upper()
        code_data = self.code_by_id.get(code)

        if code_data:
            return ICDCode(
                code=code_data['code'],
                display=code_data['display'],
                billable=code_data.get('billable', True),
                category=code_data.get('category'),
                hcc=self._is_hcc_code(code)
            )
        else:
            # Code not in our dataset - return with minimal info
            return ICDCode(
                code=code,
                display=f"Unknown ICD-10 code: {code}",
                billable=False,
                category=code[:3] if len(code) >= 3 else None,
                hcc=False
            )

    def _crosswalk_lookup(self, snomed_code: str) -> List[ICDCode]:
        """
        Look up ICD-10 codes via SNOMED crosswalk.

        Args:
            snomed_code: SNOMED CT concept ID

        Returns:
            List of mapped ICD-10 codes (sorted by map priority)
        """
        if not self.crosswalk:
            raise RuntimeError(
                "SNOMED crosswalk not loaded. "
                "Use DirectICD10Resolver or load crosswalk data."
            )

        mappings = self.crosswalk.get(snomed_code, [])

        if not mappings:
            return []

        # Sort by map priority (lower is higher priority)
        mappings = sorted(mappings, key=lambda x: x['map_priority'])

        # Convert to ICDCode objects
        icd_codes = []
        for mapping in mappings:
            icd_code = self._validate_code(mapping['icd10_code'])
            icd_codes.append(icd_code)

        return icd_codes

    def _is_hcc_code(self, code: str) -> bool:
        """
        Determine if an ICD-10 code is an HCC (Hierarchical Condition Category).

        HCC codes are used for risk adjustment in Medicare Advantage.
        Key diabetes-related HCC codes include complications.

        Args:
            code: ICD-10 code

        Returns:
            True if code is an HCC code
        """
        # Common diabetes HCC code patterns
        hcc_patterns = [
            'E10.',  # Type 1 diabetes with complications
            'E11.2',  # Type 2 diabetes with kidney complications
            'E11.3',  # Type 2 diabetes with ophthalmic complications
            'E11.4',  # Type 2 diabetes with neurological complications
            'E11.5',  # Type 2 diabetes with circulatory complications
            'E11.6',  # Type 2 diabetes with other specified complications
            'E13.',  # Other specified diabetes with complications
            'N18.',  # Chronic kidney disease
            'I25.',  # Chronic ischemic heart disease
            'Z94.0', # Kidney transplant status
        ]

        for pattern in hcc_patterns:
            if code.startswith(pattern):
                return True

        return False

    def map_batch(self, concepts: List[ResolvedConcept]) -> List[List[ICDCode]]:
        """
        Map multiple concepts to ICD-10 codes in batch.

        Args:
            concepts: List of ResolvedConcept objects

        Returns:
            List of lists of ICDCode objects
        """
        return [self.map(concept) for concept in concepts]

    def get_code_hierarchy(self, code: str) -> List[ICDCode]:
        """
        Get the code hierarchy for an ICD-10 code.

        Returns codes from most general (3-char) to most specific.

        Args:
            code: ICD-10 code

        Returns:
            List of ICDCode objects in hierarchy
        """
        code = code.strip().upper()
        hierarchy = []

        # 3-character code (category)
        if len(code) >= 3:
            category_code = code[:3]
            category = self._validate_code(category_code)
            if category:
                hierarchy.append(category)

        # 4-character code (subcategory)
        if len(code) >= 4:
            subcat_code = code[:4]
            subcat = self._validate_code(subcat_code)
            if subcat:
                hierarchy.append(subcat)

        # 5-character code (specific)
        if len(code) >= 5:
            specific_code = code[:5]
            specific = self._validate_code(specific_code)
            if specific:
                hierarchy.append(specific)

        # Full code
        full = self._validate_code(code)
        if full and full not in hierarchy:
            hierarchy.append(full)

        return hierarchy

    def suggest_more_specific_codes(self, code: str) -> List[ICDCode]:
        """
        Suggest more specific codes for a given ICD-10 code.

        Useful for identifying when a more specific code might be available
        with additional context.

        Args:
            code: ICD-10 code

        Returns:
            List of more specific ICDCode objects
        """
        code = code.strip().upper()
        more_specific = []

        # Find all codes that start with this code
        for candidate_code in self.code_by_id.keys():
            if candidate_code.startswith(code) and len(candidate_code) > len(code):
                icd_code = self._validate_code(candidate_code)
                more_specific.append(icd_code)

        return more_specific

    def get_diabetes_related_codes(self) -> List[ICDCode]:
        """
        Get all diabetes-related ICD-10 codes from the dataset.

        Returns:
            List of diabetes-related ICDCode objects
        """
        diabetes_codes = []

        for code_data in self.codes:
            code = code_data['code']
            # Diabetes codes are E08-E13
            if code.startswith('E08') or code.startswith('E09') or \
               code.startswith('E10') or code.startswith('E11') or \
               code.startswith('E13'):
                diabetes_codes.append(self._validate_code(code))

        return diabetes_codes


def map_concept_to_icd10(
    concept: ResolvedConcept,
    icd10_codes_path: str = ICD10_CODES_PATH
) -> List[ICDCode]:
    """
    Convenience function to map a single concept to ICD-10 codes.

    Args:
        concept: ResolvedConcept object
        icd10_codes_path: Path to ICD-10 codes file

    Returns:
        List of ICDCode objects
    """
    mapper = ICD10Mapper(icd10_codes_path=icd10_codes_path)
    return mapper.map(concept)


if __name__ == "__main__":
    # Test the mapper
    from src.shared.concept_resolver import DirectICD10Resolver

    print("Testing ICD10Mapper\n")

    # Initialize
    mapper = ICD10Mapper()
    resolver = DirectICD10Resolver()

    # Test entities
    test_entities = [
        "Type 2 diabetes mellitus with hyperglycemia",
        "diabetic neuropathy",
        "chronic kidney disease",
        "hypertension",
    ]

    print(f"{'Entity':<45} {'ICD-10':<10} {'HCC':<5} {'Display'}")
    print("=" * 100)

    for entity_text in test_entities:
        # Resolve to concept
        concept = resolver.resolve(entity_text, "condition")

        if concept:
            # Map to ICD-10
            icd_codes = mapper.map(concept)

            for icd_code in icd_codes:
                hcc_mark = "✓" if icd_code.hcc else ""
                print(f"{entity_text:<45} {icd_code.code:<10} {hcc_mark:<5} {icd_code.display[:40]}")

                # Show more specific options if available
                more_specific = mapper.suggest_more_specific_codes(icd_code.code)
                if more_specific and len(icd_code.code) < 6:
                    print(f"{'  → More specific options:':<45} {len(more_specific)} available")

        else:
            print(f"{entity_text:<45} {'NOT RESOLVED'}")

        print()

    # Show diabetes code statistics
    print("\n" + "=" * 100)
    diabetes_codes = mapper.get_diabetes_related_codes()
    print(f"Total diabetes-related codes in dataset: {len(diabetes_codes)}")

    hcc_count = sum(1 for code in diabetes_codes if code.hcc)
    print(f"HCC codes: {hcc_count}")
