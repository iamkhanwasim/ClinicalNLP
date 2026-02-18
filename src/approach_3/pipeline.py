"""
Approach 3: Baseline NER Pipeline (Refactored for Phase 1 Architecture)

Pipeline flow:
1. Parse clinical note into sections
2. Extract entities using NER Factory (swappable models)
3. Resolve entities to SNOMED concepts using Semantic Resolver (embedding-based)
4. Map SNOMED concepts to ICD-10 codes using crosswalk
5. Auto-compute inference_strength for each code
6. Build evidence spans with needs_review flags
7. Format output with color-coded inference strength

Key improvements over old architecture:
- Factory pattern for swappable NER models
- Embedding-based semantic resolution (no fuzzy matching)
- Inference strength tracking (explicit/strong/weak)
- Pre-computed SNOMED embeddings for fast lookup
- Rich formatted output with review flags
"""

import time
from pathlib import Path
from typing import Optional, List, Dict

from src.shared.models import (
    ClinicalNote,
    PipelineOutput,
    ExtractionResult,
    EvidenceSpan,
    Entity,
    EntitySpan,
    EnrichedEntity,
    SNOMEDConcept,
    ICD10Code,
    ICDCode
)
from src.shared.note_parser import NoteParser
from src.factories.ner_factory import NERFactory
from src.factories.embedding_factory import EmbeddingFactory
from src.resolvers.semantic_resolver import SemanticResolver
from src.resolvers.icd10_mapper import ICD10Mapper
from src.shared.output_formatter import OutputFormatter


class Approach3Pipeline:
    """
    Baseline NER → SNOMED → ICD-10 pipeline with new architecture.

    This pipeline demonstrates the baseline approach without context enrichment
    or LLM validation. It operates at entity level and serves as a benchmark
    for more advanced approaches (3+KG and 4).

    Features:
    - Swappable NER models via factory pattern
    - Embedding-based SNOMED resolution
    - Crosswalk-based ICD-10 mapping
    - Automatic inference strength computation
    - needs_review flagging for uncertain codes
    """

    def __init__(
        self,
        ner_model: str = "stanza_clinical",
        embedding_model: str = "biobert",
        snomed_data_path: str = "data/reference/snomed_diabetes_subset.json",
        snomed_embeddings_dir: str = "data/reference/snomed_embeddings",
        crosswalk_path: str = "data/reference/snomed_icd10_crosswalk.json",
        icd10_codes_path: str = "data/reference/icd10_diabetes_codes.json",
        confidence_threshold: float = 0.7,
        top_k: int = 3
    ):
        """
        Initialize the pipeline components.

        Args:
            ner_model: NER model name (stanza_clinical, scispacy, biobert_ner, med7)
            embedding_model: Embedding model name (biobert, sapbert, pubmedbert)
            snomed_data_path: Path to SNOMED diabetes subset JSON
            snomed_embeddings_dir: Directory with pre-computed embeddings (.npz)
            crosswalk_path: Path to SNOMED→ICD-10 crosswalk JSON
            icd10_codes_path: Path to ICD-10 codes JSON
            confidence_threshold: Minimum confidence for SNOMED matches (0.0-1.0)
            top_k: Number of top SNOMED matches to consider
        """
        print(f"Initializing Approach 3 Pipeline...")
        print(f"  NER model: {ner_model}")
        print(f"  Embedding model: {embedding_model}")

        # Initialize components
        self.parser = NoteParser()
        self.ner = NERFactory.create_extractor(ner_model)

        # Create embedder and semantic resolver
        embedder = EmbeddingFactory.create_embedder(embedding_model)
        embeddings_path = Path(snomed_embeddings_dir) / f"{embedding_model}.npz"

        self.semantic_resolver = SemanticResolver(
            embedder=embedder,
            snomed_data_path=Path(snomed_data_path),
            embeddings_path=embeddings_path,
            top_k=top_k,
            confidence_threshold=confidence_threshold
        )
        self.icd10_mapper = ICD10Mapper(
            crosswalk_path=Path(crosswalk_path),
            icd10_reference_path=Path(icd10_codes_path)
        )
        self.formatter = OutputFormatter()

        # Store config
        self.ner_model_name = ner_model
        self.embedding_model_name = embedding_model
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k

        print(f"Pipeline ready!")

    def process_file(self, file_path: str | Path) -> PipelineOutput:
        """
        Process a clinical note from a file.

        Args:
            file_path: Path to clinical note text file

        Returns:
            PipelineOutput with extracted codes and evidence
        """
        # Parse note
        note = self.parser.parse_file(file_path)
        return self.process(note)

    def process(self, note: ClinicalNote) -> PipelineOutput:
        """
        Process a clinical note through the full pipeline.

        Args:
            note: Parsed ClinicalNote object

        Returns:
            PipelineOutput with extracted ICD-10 codes and inference strength
        """
        start_time = time.time()

        print(f"\nProcessing note: {note.note_id}")

        # Step 1: Extract entities with NER
        entities = self.ner.extract_entities(note.raw_text)
        print(f"  Extracted {len(entities)} entities")

        # Step 2: Process each entity through SNOMED → ICD-10 pipeline
        extractions = []
        skipped = 0

        for entity in entities:
            extraction = self._process_entity(entity, note)
            if extraction:
                extractions.append(extraction)
            else:
                skipped += 1

        print(f"  Successfully mapped {len(extractions)} entities to ICD-10 codes")
        print(f"  Skipped {skipped} entities (no SNOMED/ICD-10 match)")

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        # Count needs_review items
        needs_review_count = sum(1 for e in extractions if e.needs_review)

        return PipelineOutput(
            note_id=note.note_id,
            approach="approach_3",
            ner_model=self.ner_model_name,
            embedding_model=self.embedding_model_name,
            extractions=extractions,
            review_required_count=needs_review_count,
            processing_time_ms=processing_time,
            ner_result=None,  # We don't use old NERResult anymore
            metadata={
                "total_entities": len(entities),
                "mapped_codes": len(extractions),
                "skipped_entities": skipped,
                "confidence_threshold": self.confidence_threshold
            }
        )

    def _process_entity(
        self,
        entity: Entity,
        note: ClinicalNote
    ) -> Optional[ExtractionResult]:
        """
        Process a single entity: resolve to SNOMED, then map to ICD-10.

        Args:
            entity: Entity from NER
            note: Full clinical note for context

        Returns:
            ExtractionResult if successful, None otherwise
        """
        # Step 1: Resolve entity to SNOMED concept
        snomed_results = self.semantic_resolver.resolve(entity.text)

        if not snomed_results:
            return None

        # Get top match (highest confidence)
        snomed_concept, confidence = snomed_results[0]

        # Step 2: Map SNOMED concept to ICD-10 codes
        icd_codes = self.icd10_mapper.map_snomed_to_icd10(
            snomed_concept.snomed_code,
            semantic_confidence=confidence
        )

        if not icd_codes:
            return None

        # Use the first (highest priority) code
        icd_code = icd_codes[0]

        # Build evidence span from entity
        evidence = EvidenceSpan(
            text=entity.text,
            section="unknown",  # We don't have section info from simple NER
            char_start=entity.start_char,
            char_end=entity.end_char,
            reasoning=f"Extracted by {self.ner_model_name} NER as {entity.label}"
        )

        # Convert Entity to EntitySpan
        # Map NER label to valid entity type
        entity_type_map = {
            'PROBLEM': 'condition',
            'TREATMENT': 'medication',
            'TEST': 'lab_value',
            'Clinical_event': 'condition',
            'Evidential': 'other',
            'DISEASE': 'condition',
            'SYMPTOM': 'symptom',
            'MEDICATION': 'medication',
            'PROCEDURE': 'procedure'
        }
        entity_type = entity_type_map.get(entity.label, 'other')

        entity_span = EntitySpan(
            text=entity.text,
            start_char=entity.start_char,
            end_char=entity.end_char,
            section="unknown",
            entity_type=entity_type,
            label=entity.label,
            source_model=self.ner_model_name,
            confidence=getattr(entity, 'confidence', None)
        )

        # Create ICDCode from ICD10Code
        icd_code_obj = ICDCode(
            code=icd_code.code,
            display=icd_code.display,
            billable=icd_code.billable,
            category=icd_code.code.split('.')[0] if '.' in icd_code.code else icd_code.code[:3],
            hcc=icd_code.hcc
        )

        return ExtractionResult(
            condition=entity.text,
            snomed_concept=snomed_concept,
            icd10_code=icd_code_obj,
            confidence=icd_code.confidence,
            inference_strength=icd_code.inference_strength,
            needs_review=icd_code.needs_review,
            evidence_spans=[evidence],
            enrichment_reasoning="",  # No enrichment in Approach 3
            source_entity=entity_span
        )

    def process_batch(
        self,
        notes: List[ClinicalNote]
    ) -> List[PipelineOutput]:
        """
        Process multiple notes in batch.

        Args:
            notes: List of ClinicalNote objects

        Returns:
            List of PipelineOutput objects
        """
        outputs = []
        for i, note in enumerate(notes):
            print(f"\n[{i+1}/{len(notes)}] Processing {note.note_id}...")
            output = self.process(note)
            outputs.append(output)

        return outputs


def run_approach_3(
    file_path: str | Path,
    ner_model: str = "stanza_clinical",
    embedding_model: str = "biobert"
) -> PipelineOutput:
    """
    Convenience function to run Approach 3 on a clinical note.

    Args:
        file_path: Path to clinical note file
        ner_model: NER model name
        embedding_model: Embedding model name

    Returns:
        PipelineOutput with results
    """
    pipeline = Approach3Pipeline(
        ner_model=ner_model,
        embedding_model=embedding_model
    )
    return pipeline.process_file(file_path)


if __name__ == "__main__":
    # Test Approach 3 pipeline with new architecture
    import sys
    from pathlib import Path

    print("="*80)
    print("Testing Approach 3: Baseline NER Pipeline (New Architecture)")
    print("="*80)

    # Find clinical notes
    notes_dir = Path("data/clinical_notes")
    note_files = list(notes_dir.glob("*.txt"))

    if not note_files:
        print("No clinical notes found in data/clinical_notes!")
        sys.exit(1)

    # Test with first note
    note_file = note_files[0]
    print(f"\nProcessing: {note_file.name}\n")

    # Run pipeline with default settings
    try:
        pipeline = Approach3Pipeline(
            # ner_model="biobert_ner",
            ner_model="med7",
            embedding_model="pubmedbert",
            confidence_threshold=0.7,
            top_k=3
        )
        output = pipeline.process_file(note_file)

        # Display results with rich formatting
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)

        formatter = OutputFormatter()
        formatter.display_pipeline_output_rich(output)

        # Save to file
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        json_path = output_dir / f"{output.note_id}_approach3.json"
        formatter.save_json(output, json_path)
        print(f"\nSaved results to {json_path}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
