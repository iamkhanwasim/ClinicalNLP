"""
Simple test script for NER factory.

Tests the NER extractors without needing all dependencies.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_ner_imports():
    """Test if NER factory can be imported."""
    print("Testing NER factory imports...")
    try:
        from src.factories.ner_factory import NERFactory
        print("[OK] NERFactory imported successfully")

        # List available models
        models = NERFactory.list_models()
        print(f"[OK] Available NER models: {', '.join(models)}")

        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False


def test_stanza_ner():
    """Test Stanza NER extractor."""
    print("\n" + "="*60)
    print("Testing Stanza NER Extractor")
    print("="*60)

    try:
        from src.factories.ner_factory import NERFactory

        print("Creating Stanza extractor...")
        extractor = NERFactory.create_extractor("stanza")
        print(f"[OK] Created: {extractor.get_model_name()}")

        # Test text
        sample_text = (
            "Patient has type 2 diabetes mellitus with diabetic retinopathy. "
            "Started on metformin 500mg twice daily and insulin glargine 10 units at bedtime."
        )

        print(f"\nTest text:\n{sample_text}\n")
        print("Extracting entities...")

        entities = extractor.extract_entities(sample_text)

        print(f"\n[OK] Found {len(entities)} entities:")
        for i, ent in enumerate(entities, 1):
            conf_str = f" (conf: {ent.confidence:.2f})" if ent.confidence < 1.0 else ""
            print(f"  {i}. [{ent.label}] {ent.text}{conf_str}")
            print(f"     Position: {ent.start_char}-{ent.end_char}")

        return True

    except ImportError as e:
        print(f"[FAIL] Stanza not installed: {e}")
        print("  Install with: pip install stanza")
        print("  Then download model: python -c \"import stanza; stanza.download('en', package='mimic')\"")
        return False
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scispacy_ner():
    """Test ScispaCy NER extractor."""
    print("\n" + "="*60)
    print("Testing ScispaCy NER Extractor")
    print("="*60)

    try:
        from src.factories.ner_factory import NERFactory

        print("Creating ScispaCy extractor...")
        extractor = NERFactory.create_extractor("scispacy")
        print(f"[OK] Created: {extractor.get_model_name()}")

        # Test text
        sample_text = (
            "Patient has type 2 diabetes mellitus with diabetic retinopathy. "
            "Started on metformin 500mg twice daily and insulin glargine 10 units at bedtime."
        )

        print(f"\nTest text:\n{sample_text}\n")
        print("Extracting entities...")

        entities = extractor.extract_entities(sample_text)

        print(f"\n[OK] Found {len(entities)} entities:")
        for i, ent in enumerate(entities, 1):
            print(f"  {i}. [{ent.label}] {ent.text}")
            print(f"     Position: {ent.start_char}-{ent.end_char}")

        return True

    except OSError as e:
        print(f"[FAIL] ScispaCy BC5CDR model not installed")
        print("  Install with:")
        print("  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.5/en_ner_bc5cdr_md-0.5.5.tar.gz")
        return False
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_biobert_ner():
    """Test BioBERT NER extractor."""
    print("\n" + "="*60)
    print("Testing BioBERT NER Extractor")
    print("="*60)

    try:
        from src.factories.ner_factory import NERFactory

        print("Creating BioBERT NER extractor...")
        print("(This will download the model on first run, ~440MB)")
        extractor = NERFactory.create_extractor("biobert")
        print(f"[OK] Created: {extractor.get_model_name()}")

        # Test text
        sample_text = (
            "Patient has type 2 diabetes mellitus with diabetic retinopathy. "
            "Started on metformin 500mg twice daily."
        )

        print(f"\nTest text:\n{sample_text}\n")
        print("Extracting entities...")

        entities = extractor.extract_entities(sample_text)

        print(f"\n[OK] Found {len(entities)} entities:")
        for i, ent in enumerate(entities, 1):
            print(f"  {i}. [{ent.label}] {ent.text}")
            print(f"     Confidence: {ent.confidence:.3f}")
            print(f"     Position: {ent.start_char}-{ent.end_char}")

        return True

    except ImportError as e:
        print(f"[FAIL] Transformers not installed: {e}")
        print("  Install with: pip install transformers torch")
        return False
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("NER Factory Test Suite")
    print("="*60)
    print()

    # Test imports
    if not test_ner_imports():
        print("\n[FAIL] Import test failed. Cannot continue.")
        return 1

    # Track results
    results = []

    # Test each NER model
    print("\nTesting NER models...")
    results.append(("Stanza", test_stanza_ner()))
    results.append(("ScispaCy", test_scispacy_ner()))
    results.append(("BioBERT", test_biobert_ner()))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, success in results:
        status = "[OK] PASS" if success else "[FAIL] FAIL"
        print(f"{name}: {status}")

    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(main())
