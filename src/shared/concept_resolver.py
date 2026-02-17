"""
Concept resolver for mapping clinical entities to standardized codes.

Provides a pluggable architecture supporting:
- Direct ICD-10 matching (Phase A - no SNOMED dependency)
- SNOMED CT intermediate resolution (Phase B - when UMLS approved)
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from rapidfuzz import fuzz, process

from src.shared.models import ResolvedConcept, EntitySpan, ICDCode
from config.settings import (
    ICD10_CODES_PATH,
    SNOMED_SUBSET_PATH,
    SNOMED_CROSSWALK_PATH,
    FUZZY_MATCH_THRESHOLD
)


class ConceptResolver(ABC):
    """
    Abstract base class for concept resolution.

    This pluggable interface allows switching between direct ICD-10 matching
    and SNOMED CT-based resolution without changing pipeline code.
    """

    @abstractmethod
    def resolve(
        self,
        entity_text: str,
        entity_type: str,
        context: Optional[Dict] = None
    ) -> Optional[ResolvedConcept]:
        """
        Resolve an entity to a medical concept.

        Args:
            entity_text: The text of the entity to resolve
            entity_type: Type of entity (condition, medication, etc.)
            context: Optional additional context for resolution

        Returns:
            ResolvedConcept if successful, None otherwise
        """
        pass

    @abstractmethod
    def resolve_batch(
        self,
        entities: List[EntitySpan],
        context: Optional[Dict] = None
    ) -> List[Optional[ResolvedConcept]]:
        """
        Resolve multiple entities in batch.

        Args:
            entities: List of EntitySpan objects
            context: Optional additional context

        Returns:
            List of ResolvedConcept objects (None for unresolved entities)
        """
        pass


class DirectICD10Resolver(ConceptResolver):
    """
    Phase A: Direct fuzzy matching against ICD-10 code descriptions.

    No SNOMED dependency - maps entity text directly to ICD-10 codes using
    multiple matching strategies.
    """

    def __init__(self, icd10_codes_path: str = ICD10_CODES_PATH):
        """
        Initialize the resolver with ICD-10 codes.

        Args:
            icd10_codes_path: Path to ICD-10 codes JSON file
        """
        self.icd10_codes_path = Path(icd10_codes_path)
        self.codes = self._load_icd10_codes()

        # Build search indices
        self.code_by_id = {code['code']: code for code in self.codes}
        self.display_texts = [code['display'] for code in self.codes]
        self.code_list = [code['code'] for code in self.codes]

        # Build keyword index for faster lookup
        self.keyword_index = self._build_keyword_index()

        print(f"Loaded {len(self.codes)} ICD-10 codes from {icd10_codes_path}")

    def _load_icd10_codes(self) -> List[Dict]:
        """Load ICD-10 codes from JSON file."""
        if not self.icd10_codes_path.exists():
            raise FileNotFoundError(
                f"ICD-10 codes file not found: {self.icd10_codes_path}"
            )

        with open(self.icd10_codes_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data.get('codes', [])

    def _build_keyword_index(self) -> Dict[str, List[str]]:
        """
        Build keyword-to-codes index for efficient keyword matching.

        Returns:
            Dictionary mapping keywords to list of ICD-10 codes
        """
        index = {}

        for code_data in self.codes:
            code = code_data['code']
            display = code_data['display'].lower()

            # Extract keywords (significant words)
            words = display.split()
            for word in words:
                # Skip very common words
                if word in {'with', 'without', 'and', 'or', 'the', 'of', 'in', 'on'}:
                    continue

                if len(word) >= 4:  # Only index words of 4+ characters
                    if word not in index:
                        index[word] = []
                    index[word].append(code)

        return index

    def resolve(
        self,
        entity_text: str,
        entity_type: str,
        context: Optional[Dict] = None
    ) -> Optional[ResolvedConcept]:
        """
        Resolve entity to ICD-10 code using multiple strategies.

        Strategy 1: Exact match on display name
        Strategy 2: Fuzzy match using token_sort_ratio
        Strategy 3: Keyword-based matching

        Args:
            entity_text: Text to resolve
            entity_type: Type of entity
            context: Optional context (e.g., other entities, lab values)

        Returns:
            ResolvedConcept if match found, None otherwise
        """
        entity_text = entity_text.strip().lower()

        if not entity_text or len(entity_text) < 3:
            return None

        # Strategy 1: Exact match
        result = self._exact_match(entity_text)
        if result:
            return result

        # Strategy 2: Fuzzy match
        result = self._fuzzy_match(entity_text)
        if result:
            return result

        # Strategy 3: Keyword match
        result = self._keyword_match(entity_text, entity_type)
        if result:
            return result

        return None

    def _exact_match(self, entity_text: str) -> Optional[ResolvedConcept]:
        """Try exact string matching against code displays."""
        entity_lower = entity_text.lower()

        for code_data in self.codes:
            display_lower = code_data['display'].lower()

            if entity_lower == display_lower or entity_lower in display_lower:
                return ResolvedConcept(
                    code=code_data['code'],
                    display=code_data['display'],
                    source="icd10_direct",
                    confidence=1.0
                )

        return None

    def _fuzzy_match(
        self,
        entity_text: str,
        threshold: float = FUZZY_MATCH_THRESHOLD
    ) -> Optional[ResolvedConcept]:
        """
        Use fuzzy string matching to find best ICD-10 code match.

        Uses rapidfuzz's token_sort_ratio for flexible matching.
        """
        # Use process.extractOne for best match
        result = process.extractOne(
            entity_text,
            self.display_texts,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold * 100  # rapidfuzz uses 0-100 scale
        )

        if result:
            matched_display, score, index = result
            confidence = score / 100.0  # Convert to 0-1 scale
            matched_code = self.codes[index]

            # Find alternative matches
            alternatives = process.extract(
                entity_text,
                self.display_texts,
                scorer=fuzz.token_sort_ratio,
                limit=3,
                score_cutoff=(threshold - 0.1) * 100
            )

            alternative_codes = []
            for alt_display, alt_score, alt_index in alternatives[1:]:  # Skip first (best match)
                alt_code = self.codes[alt_index]
                alternative_codes.append({
                    'code': alt_code['code'],
                    'display': alt_code['display'],
                    'confidence': alt_score / 100.0
                })

            return ResolvedConcept(
                code=matched_code['code'],
                display=matched_code['display'],
                source="icd10_direct",
                confidence=confidence,
                alternative_codes=alternative_codes
            )

        return None

    def _keyword_match(
        self,
        entity_text: str,
        entity_type: str
    ) -> Optional[ResolvedConcept]:
        """
        Match based on keywords in entity text.

        Particularly useful for complex entities like "diabetic neuropathy"
        where fuzzy matching might struggle.
        """
        entity_lower = entity_text.lower()
        words = entity_lower.split()

        # Find codes that match multiple keywords
        candidate_codes = {}

        for word in words:
            if word in self.keyword_index:
                for code in self.keyword_index[word]:
                    candidate_codes[code] = candidate_codes.get(code, 0) + 1

        if not candidate_codes:
            return None

        # Sort by number of matching keywords
        best_code = max(candidate_codes, key=candidate_codes.get)
        match_count = candidate_codes[best_code]

        # Calculate confidence based on match ratio
        confidence = min(0.9, (match_count / len(words)) * 1.2)

        if confidence < 0.5:
            return None

        code_data = self.code_by_id[best_code]

        return ResolvedConcept(
            code=code_data['code'],
            display=code_data['display'],
            source="icd10_direct",
            confidence=confidence
        )

    def resolve_batch(
        self,
        entities: List[EntitySpan],
        context: Optional[Dict] = None
    ) -> List[Optional[ResolvedConcept]]:
        """
        Resolve multiple entities in batch.

        Args:
            entities: List of EntitySpan objects
            context: Optional context

        Returns:
            List of ResolvedConcept objects (None for unresolved)
        """
        return [
            self.resolve(entity.text, entity.entity_type, context)
            for entity in entities
        ]

    def get_code_info(self, code: str) -> Optional[ICDCode]:
        """
        Get full information for an ICD-10 code.

        Args:
            code: ICD-10 code

        Returns:
            ICDCode object if found
        """
        code = code.strip().upper()
        code_data = self.code_by_id.get(code)

        if code_data:
            return ICDCode(
                code=code_data['code'],
                display=code_data['display'],
                billable=code_data.get('billable', True),
                category=code_data.get('category'),
                hcc=code_data.get('hcc', False)
            )

        return None


class SNOMEDResolver(ConceptResolver):
    """
    Phase B: SNOMED CT concept resolution + crosswalk to ICD-10.

    Requires UMLS account and SNOMED CT data.
    Will be implemented when SNOMED data becomes available.
    """

    def __init__(
        self,
        snomed_data_path: str = SNOMED_SUBSET_PATH,
        crosswalk_path: str = SNOMED_CROSSWALK_PATH
    ):
        """
        Initialize SNOMED resolver.

        Args:
            snomed_data_path: Path to SNOMED concepts JSON
            crosswalk_path: Path to SNOMED→ICD-10 crosswalk JSON
        """
        self.snomed_data_path = Path(snomed_data_path)
        self.crosswalk_path = Path(crosswalk_path)

        # Check if data files exist
        if not self.snomed_data_path.exists():
            raise FileNotFoundError(
                f"SNOMED data not found: {snomed_data_path}. "
                f"This requires UMLS approval. Use DirectICD10Resolver for now."
            )

        if not self.crosswalk_path.exists():
            raise FileNotFoundError(
                f"SNOMED→ICD-10 crosswalk not found: {crosswalk_path}"
            )

        self.concepts = self._load_snomed_concepts()
        self.crosswalk = self._load_crosswalk()

        print(f"Loaded {len(self.concepts)} SNOMED concepts")
        print(f"Loaded {len(self.crosswalk)} crosswalk mappings")

    def _load_snomed_concepts(self) -> Dict:
        """Load SNOMED concepts from JSON."""
        with open(self.snomed_data_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_crosswalk(self) -> Dict:
        """Load SNOMED→ICD-10 crosswalk mappings."""
        with open(self.crosswalk_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def resolve(
        self,
        entity_text: str,
        entity_type: str,
        context: Optional[Dict] = None
    ) -> Optional[ResolvedConcept]:
        """
        Resolve entity to SNOMED concept.

        TODO: Implement when SNOMED data is available.
        Strategy 1: Exact match on SNOMED display name / synonyms
        Strategy 2: Fuzzy string matching
        Strategy 3: (Optional) SapBERT embedding similarity
        """
        # Placeholder implementation
        raise NotImplementedError(
            "SNOMED resolution will be implemented in Phase B "
            "when UMLS approval is granted and SNOMED data is available."
        )

    def resolve_batch(
        self,
        entities: List[EntitySpan],
        context: Optional[Dict] = None
    ) -> List[Optional[ResolvedConcept]]:
        """Batch resolution for SNOMED."""
        raise NotImplementedError("SNOMED resolution not yet implemented.")


# Factory function for creating resolvers
def create_resolver(mode: str = "direct_icd10") -> ConceptResolver:
    """
    Factory function to create appropriate concept resolver.

    Args:
        mode: Resolution mode ("direct_icd10" or "snomed")

    Returns:
        ConceptResolver instance
    """
    if mode == "direct_icd10":
        return DirectICD10Resolver()
    elif mode == "snomed":
        return SNOMEDResolver()
    else:
        raise ValueError(f"Unknown resolution mode: {mode}")


if __name__ == "__main__":
    # Test the resolver
    print("Testing DirectICD10Resolver\n")

    resolver = DirectICD10Resolver()

    # Test cases
    test_entities = [
        ("diabetes mellitus", "condition"),
        ("Type 2 diabetes with hyperglycemia", "condition"),
        ("diabetic neuropathy", "condition"),
        ("insulin", "medication"),
        ("hypertension", "condition"),
        ("chronic kidney disease", "condition"),
    ]

    print(f"{'Entity':<40} {'Code':<10} {'Confidence':<12} {'Display'}")
    print("=" * 100)

    for entity_text, entity_type in test_entities:
        result = resolver.resolve(entity_text, entity_type)

        if result:
            print(f"{entity_text:<40} {result.code:<10} {result.confidence:<12.2f} {result.display[:40]}")
        else:
            print(f"{entity_text:<40} {'NOT FOUND':<10}")

    print("\n" + "=" * 100)
