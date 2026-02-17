"""
Named Entity Recognition (NER) extractor for clinical text.

Uses ScispaCy and optionally Med7 for biomedical entity extraction.
Extracts entities with character offsets for evidence span tracking.
"""

import spacy
from spacy.language import Language
import time
from typing import List, Optional, Set
from pathlib import Path

from src.shared.models import EntitySpan, NERResult, ClinicalNote
from config.settings import (
    NER_MODEL,
    NER_MODEL_SECONDARY,
    USE_SECONDARY_NER,
    ENTITY_TYPE_MAPPING,
    MIN_ENTITY_LENGTH,
    CONDITION_LABELS,
    MEDICATION_LABELS
)


class NERExtractor:
    """
    Extract named entities from clinical notes using biomedical NER models.

    Supports multiple NER models and merges results for better coverage.
    Maintains character offsets and section attribution for evidence tracking.
    """

    def __init__(
        self,
        model_name: str = NER_MODEL,
        secondary_model: Optional[str] = None,
        use_secondary: bool = USE_SECONDARY_NER
    ):
        """
        Initialize the NER extractor.

        Args:
            model_name: Primary spaCy model name (default: en_ner_bc5cdr_md)
            secondary_model: Optional secondary model for ensemble extraction
            use_secondary: Whether to use secondary model
        """
        self.model_name = model_name
        self.secondary_model_name = secondary_model or NER_MODEL_SECONDARY
        self.use_secondary = use_secondary

        # Load primary model
        try:
            self.nlp = spacy.load(model_name)
            print(f"Loaded primary NER model: {model_name}")
        except OSError:
            raise RuntimeError(
                f"Failed to load NER model '{model_name}'. "
                f"Please install it with: pip install {model_name}"
            )

        # Load secondary model if requested
        self.nlp_secondary = None
        if self.use_secondary:
            try:
                self.nlp_secondary = spacy.load(self.secondary_model_name)
                print(f"Loaded secondary NER model: {self.secondary_model_name}")
            except OSError:
                print(f"⚠ Warning: Could not load secondary model '{self.secondary_model_name}'")
                self.use_secondary = False

    def extract(self, note: ClinicalNote) -> NERResult:
        """
        Extract entities from a clinical note.

        Processes each section separately to maintain section attribution.
        Merges results from primary and secondary models if enabled.

        Args:
            note: Parsed ClinicalNote object

        Returns:
            NERResult containing all extracted entities
        """
        start_time = time.time()
        all_entities = []

        # Get section offsets for character position calculation
        section_offsets = self._calculate_section_offsets(note)

        # Extract entities from each section
        for section_name, section_text in note.sections.items():
            if not section_text.strip():
                continue

            section_start_offset = section_offsets.get(section_name, 0)

            # Primary model extraction
            entities = self._extract_from_text(
                section_text,
                section_name,
                section_start_offset,
                self.nlp
            )
            all_entities.extend(entities)

            # Secondary model extraction (if enabled)
            if self.use_secondary and self.nlp_secondary:
                secondary_entities = self._extract_from_text(
                    section_text,
                    section_name,
                    section_start_offset,
                    self.nlp_secondary
                )
                # Merge, avoiding duplicates
                all_entities = self._merge_entities(all_entities, secondary_entities)

        # Post-processing: filter and deduplicate
        all_entities = self._post_process_entities(all_entities)

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        return NERResult(
            entities=all_entities,
            model_name=self.model_name,
            processing_time_ms=processing_time
        )

    def _extract_from_text(
        self,
        text: str,
        section_name: str,
        section_offset: int,
        nlp_model: Language
    ) -> List[EntitySpan]:
        """
        Extract entities from text using a specific spaCy model.

        Args:
            text: Text to process
            section_name: Name of the section
            section_offset: Character offset of section start in full note
            nlp_model: spaCy NLP model to use

        Returns:
            List of EntitySpan objects
        """
        entities = []
        doc = nlp_model(text)

        for ent in doc.ents:
            # Skip very short entities (likely noise)
            if len(ent.text.strip()) < MIN_ENTITY_LENGTH:
                continue

            # Map NER label to our entity type
            entity_type = self._classify_entity_type(ent.label_)

            # Calculate absolute character positions in the full note
            absolute_start = section_offset + ent.start_char
            absolute_end = section_offset + ent.end_char

            entity = EntitySpan(
                text=ent.text.strip(),
                start_char=absolute_start,
                end_char=absolute_end,
                section=section_name,
                entity_type=entity_type,
                label=ent.label_,
                confidence=None  # ScispaCy doesn't provide confidence scores
            )

            entities.append(entity)

        return entities

    def _classify_entity_type(self, ner_label: str) -> str:
        """
        Map NER label to our standardized entity types.

        Args:
            ner_label: Label from the NER model (e.g., DISEASE, CHEMICAL)

        Returns:
            Standardized entity type (condition, medication, lab_value, etc.)
        """
        # Check predefined mapping first
        if ner_label in ENTITY_TYPE_MAPPING:
            return ENTITY_TYPE_MAPPING[ner_label]

        # Fallback logic based on label characteristics
        if ner_label in CONDITION_LABELS:
            return "condition"
        elif ner_label in MEDICATION_LABELS:
            return "medication"
        else:
            return "other"

    def _calculate_section_offsets(self, note: ClinicalNote) -> dict[str, int]:
        """
        Calculate character offsets for each section in the raw text.

        Args:
            note: Parsed ClinicalNote object

        Returns:
            Dictionary mapping section names to start offsets
        """
        offsets = {}
        current_offset = 0

        for section_name, section_content in note.sections.items():
            # Find section in raw text
            section_start = note.raw_text.find(section_content, current_offset)
            if section_start != -1:
                offsets[section_name] = section_start
                current_offset = section_start + len(section_content)
            else:
                # Fallback: approximate offset
                offsets[section_name] = current_offset
                current_offset += len(section_content)

        return offsets

    def _merge_entities(
        self,
        entities1: List[EntitySpan],
        entities2: List[EntitySpan]
    ) -> List[EntitySpan]:
        """
        Merge entity lists, removing duplicates.

        Two entities are considered duplicates if they have significant overlap
        in their text spans.

        Args:
            entities1: First list of entities
            entities2: Second list of entities

        Returns:
            Merged list without duplicates
        """
        merged = list(entities1)

        for ent2 in entities2:
            is_duplicate = False

            for ent1 in entities1:
                if self._is_overlapping(ent1, ent2):
                    is_duplicate = True
                    break

            if not is_duplicate:
                merged.append(ent2)

        return merged

    def _is_overlapping(self, ent1: EntitySpan, ent2: EntitySpan, threshold: float = 0.5) -> bool:
        """
        Check if two entities overlap significantly.

        Args:
            ent1: First entity
            ent2: Second entity
            threshold: Minimum overlap ratio to consider entities as duplicates

        Returns:
            True if entities overlap above threshold
        """
        # Check if in same section
        if ent1.section != ent2.section:
            return False

        # Calculate overlap
        overlap_start = max(ent1.start_char, ent2.start_char)
        overlap_end = min(ent1.end_char, ent2.end_char)

        if overlap_start >= overlap_end:
            return False  # No overlap

        overlap_length = overlap_end - overlap_start
        min_length = min(ent1.end_char - ent1.start_char, ent2.end_char - ent2.start_char)

        overlap_ratio = overlap_length / min_length

        return overlap_ratio >= threshold

    def _post_process_entities(self, entities: List[EntitySpan]) -> List[EntitySpan]:
        """
        Post-process entities: filter noise, deduplicate, normalize.

        Args:
            entities: Raw list of entities

        Returns:
            Cleaned list of entities
        """
        processed = []
        seen_texts = set()

        for entity in entities:
            # Skip if exact duplicate text in same section
            entity_key = f"{entity.section}:{entity.text.lower()}"
            if entity_key in seen_texts:
                continue

            # Additional filtering
            text = entity.text.strip()

            # Skip entities that are just numbers or very common words
            if text.isdigit():
                continue

            # Skip single-character entities
            if len(text) <= 1:
                continue

            seen_texts.add(entity_key)
            processed.append(entity)

        return processed

    def extract_by_section(self, note: ClinicalNote, section_name: str) -> List[EntitySpan]:
        """
        Extract entities from a specific section only.

        Args:
            note: Parsed ClinicalNote object
            section_name: Name of section to extract from

        Returns:
            List of entities from that section
        """
        result = self.extract(note)
        return result.get_entities_by_section(section_name)

    def extract_by_type(self, note: ClinicalNote, entity_type: str) -> List[EntitySpan]:
        """
        Extract entities of a specific type only.

        Args:
            note: Parsed ClinicalNote object
            entity_type: Type of entity to extract (condition, medication, etc.)

        Returns:
            List of entities of that type
        """
        result = self.extract(note)
        return result.get_entities_by_type(entity_type)


def extract_entities(note: ClinicalNote, model_name: str = NER_MODEL) -> NERResult:
    """
    Convenience function to extract entities from a note.

    Args:
        note: Parsed ClinicalNote object
        model_name: NER model to use

    Returns:
        NERResult with extracted entities
    """
    extractor = NERExtractor(model_name=model_name)
    return extractor.extract(note)


if __name__ == "__main__":
    # Test the NER extractor
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from src.shared.note_parser import parse_clinical_note
    from config.settings import CLINICAL_NOTES_DIR

    # Test with first clinical note
    note_files = list(CLINICAL_NOTES_DIR.glob("*.txt"))
    if not note_files:
        print("No clinical notes found!")
        sys.exit(1)

    note_file = note_files[0]
    print(f"Testing NER extraction on: {note_file.name}\n")

    # Parse note
    note = parse_clinical_note(note_file)
    print(f"Parsed note: {note.note_id}")
    print(f"  Sections: {len(note.sections)}")
    print(f"  Length: {len(note.raw_text)} characters\n")

    # Extract entities
    print("Extracting entities...")
    extractor = NERExtractor()
    result = extractor.extract(note)

    print(f"\n{'='*60}")
    print(f"Extraction Results")
    print(f"{'='*60}")
    print(f"Total entities: {len(result.entities)}")
    print(f"Processing time: {result.processing_time_ms:.2f} ms")

    # Group by type
    by_type = {}
    for entity in result.entities:
        if entity.entity_type not in by_type:
            by_type[entity.entity_type] = []
        by_type[entity.entity_type].append(entity)

    print(f"\nEntities by type:")
    for entity_type, entities in by_type.items():
        print(f"  {entity_type}: {len(entities)}")

    # Show sample entities
    print(f"\nSample entities (first 10):")
    for i, entity in enumerate(result.entities[:10], 1):
        print(f"  {i}. {entity}")
