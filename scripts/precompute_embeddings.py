"""
Pre-compute embeddings for SNOMED concepts.

This script embeds all SNOMED concepts using specified embedding models
and saves the results as .npz files for fast lookup by SemanticResolver.

Input: data/reference/snomed_diabetes_subset.json
Output: data/reference/snomed_embeddings/{model_name}.npz

Each .npz file contains:
- embeddings: numpy array of shape (N, embedding_dim)
- concept_ids: numpy array of SNOMED concept IDs
- display_names: numpy array of display names

Usage:
    python scripts/precompute_embeddings.py \
        --snomed-data data/reference/snomed_diabetes_subset.json \
        --output-dir data/reference/snomed_embeddings \
        --models sapbert biobert pubmedbert
"""

import json
import argparse
from pathlib import Path
from typing import List
import logging

import numpy as np
from tqdm import tqdm

# Add parent directory to path to import from src
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.factories.embedding_factory import EmbeddingFactory

logger = logging.getLogger(__name__)


def load_snomed_concepts(data_path: Path) -> List[dict]:
    """Load SNOMED concepts from JSON."""
    logger.info(f"Loading SNOMED concepts from {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    concepts = data.get('concepts', [])
    logger.info(f"Loaded {len(concepts)} SNOMED concepts")

    return concepts


def precompute_embeddings(
    concepts: List[dict],
    model_name: str,
    output_path: Path
):
    """
    Pre-compute embeddings for all concepts using specified model.

    Args:
        concepts: List of SNOMED concept dictionaries
        model_name: Embedding model name
        output_path: Path to save .npz file
    """
    logger.info(f"\nProcessing with {model_name}...")
    logger.info("=" * 60)

    # Create embedder
    embedder = EmbeddingFactory.create_embedder(model_name)

    # Extract concept IDs and display names
    concept_ids = []
    display_names = []
    for concept in concepts:
        concept_ids.append(concept['snomed_code'])
        display_names.append(concept['display'])

    # Embed all display names
    logger.info(f"Embedding {len(display_names)} concepts...")
    embeddings_list = []

    for display_name in tqdm(display_names, desc=f"Embedding ({model_name})"):
        embedding = embedder.encode(display_name)
        embeddings_list.append(embedding)

    # Stack into numpy array
    embeddings = np.vstack(embeddings_list)

    logger.info(f"Generated embeddings: shape {embeddings.shape}")

    # Save to .npz file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        concept_ids=np.array(concept_ids),
        display_names=np.array(display_names)
    )

    logger.info(f"Saved embeddings to {output_path}")

    # Report file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"  File size: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='Pre-compute embeddings for SNOMED concepts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Embed with all models
  python scripts/precompute_embeddings.py \\
      --snomed-data data/reference/snomed_diabetes_subset.json \\
      --output-dir data/reference/snomed_embeddings \\
      --models sapbert biobert pubmedbert

  # Embed with SapBERT only (recommended)
  python scripts/precompute_embeddings.py \\
      --snomed-data data/reference/snomed_diabetes_subset.json \\
      --output-dir data/reference/snomed_embeddings \\
      --models sapbert

Output files:
  data/reference/snomed_embeddings/sapbert.npz
  data/reference/snomed_embeddings/biobert.npz
  data/reference/snomed_embeddings/pubmedbert.npz

Each .npz file contains:
  - embeddings: (N, 768) array of embeddings
  - concept_ids: (N,) array of SNOMED concept IDs
  - display_names: (N,) array of display names
        """
    )
    parser.add_argument(
        '--snomed-data',
        type=str,
        required=True,
        help='Path to snomed_diabetes_subset.json'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for embedding files'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['sapbert'],
        choices=['sapbert', 'biobert', 'pubmedbert'],
        help='Embedding models to use (default: sapbert)'
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Parse paths
    snomed_data_path = Path(args.snomed_data)
    output_dir = Path(args.output_dir)

    # Validate input
    if not snomed_data_path.exists():
        logger.error(f"✗ SNOMED data not found: {snomed_data_path}")
        logger.error("Run scripts/extract_snomed_diabetes.py first")
        return 1

    print(f"{'='*60}")
    print(f"Pre-computing SNOMED Embeddings")
    print(f"{'='*60}")
    print(f"SNOMED data: {snomed_data_path}")
    print(f"Output dir: {output_dir}")
    print(f"Models: {', '.join(args.models)}\n")

    # Load concepts
    concepts = load_snomed_concepts(snomed_data_path)

    # Process each model
    for model_name in args.models:
        output_path = output_dir / f"{model_name}.npz"

        try:
            precompute_embeddings(concepts, model_name, output_path)
        except Exception as e:
            logger.error(f"✗ Failed to process {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"Pre-computation Complete")
    print(f"{'='*60}")
    print(f"Embedding files saved to: {output_dir}")
    print(f"\nYou can now use SemanticResolver with pre-computed embeddings.")

    return 0


if __name__ == "__main__":
    exit(main())
