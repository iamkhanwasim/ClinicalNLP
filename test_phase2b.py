"""
Test script for Phase 2B fixes.

Runs the Approach 3 pipeline on Note 1 with new defaults:
- ScispaCy NER (cleaner extraction)
- SapBERT embeddings (best SNOMED matching)
- Entity filtering (removes garbage)
- Entity deduplication (removes duplicates)
- Section attribution (proper section tags)
- Fixed crosswalk ranking (Type 2 preferred for ambiguous DM)
"""

from pathlib import Path
from src.approach_3.pipeline import Approach3Pipeline
from src.shared.output_formatter import OutputFormatter

def main():
    print("=" * 80)
    print("Phase 2B: Testing Approach 3 Pipeline with Fixed Configuration")
    print("=" * 80)
    print("\nConfiguration:")
    print("  - NER Model: ScispaCy (default)")
    print("  - Embedding Model: SapBERT (default)")
    print("  - Entity Filtering: Enabled")
    print("  - Entity Deduplication: Enabled")
    print("  - Section Attribution: Fixed")
    print("  - Crosswalk Ranking: Fixed (Type 2 preferred)")
    print("\n")

    # Note 1 path
    note_file = Path("data/clinical_notes/note_001_amazon_109224.txt")

    # Initialize pipeline with defaults (ScispaCy + SapBERT)
    print("Initializing pipeline with default configuration...")
    pipeline = Approach3Pipeline(
        # ner_model and embedding_model will use defaults from config
        confidence_threshold=0.7,
        top_k=3
    )

    # Process note
    print(f"\nProcessing: {note_file.name}")
    output = pipeline.process_file(note_file)

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    formatter = OutputFormatter()
    formatter.display_pipeline_output_rich(output)

    # Save output
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    json_path = output_dir / f"{output.note_id}_phase2b_test.json"
    formatter.save_json(output, json_path)
    print(f"\nSaved results to {json_path}")

    # Verification checks
    print("\n" + "=" * 80)
    print("PHASE 2B VERIFICATION")
    print("=" * 80)

    print("\n✓ Key Fixes to Verify:")
    print("  1. No garbage entities (section headers, 'financial constraints', etc.)")
    print("  2. No duplicate entities")
    print("  3. All entities have section attribution (not 'unknown')")
    print("  4. 'Hypertension' resolves to hypertension (not diabetes)")
    print("  5. 'Diabetes mellitus' maps to E11.x (Type 2), not E10.x (Type 1)")
    print("  6. Expanded SNOMED subset includes co-occurring concepts")

    # Check for section attribution
    unknown_sections = sum(1 for e in output.extractions if e.source_entity and e.source_entity.section == "unknown")
    print(f"\n✓ Entities with 'unknown' section: {unknown_sections}/{len(output.extractions)}")

    # Check for specific entities
    print("\n✓ Extracted Conditions:")
    for extraction in output.extractions[:10]:  # Show first 10
        section = extraction.source_entity.section if extraction.source_entity else "N/A"
        print(f"  - {extraction.condition} -> {extraction.icd10_code.code} (section: {section})")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
