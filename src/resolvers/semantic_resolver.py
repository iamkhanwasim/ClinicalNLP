"""
Semantic Resolver - Embedding-based SNOMED concept normalization.

Uses pre-computed embeddings for fast cosine similarity search.
NO fuzzy string matching (rapidfuzz).

Process:
1. Exact match fast path (case-insensitive)
2. Embed entity text using same model
3. Compute cosine similarity against pre-computed SNOMED embeddings
4. Return top-k matches with confidence scores

Pre-computed embeddings stored as .npz files:
- data/reference/snomed_embeddings/{model_name}.npz
- Contains: concept_ids, embeddings, display_names
"""

from typing import List, Optional, Dict, Tuple
from pathlib import Path
import json
import logging

import numpy as np
from scipy.spatial.distance import cosine

from ..shared.models import SNOMEDConcept
from ..factories.embedding_factory import BaseEmbedder, EmbeddingFactory

logger = logging.getLogger(__name__)


class SemanticResolver:
    """
    Semantic resolver for SNOMED concept normalization.

    Uses embedding-based cosine similarity for concept matching.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        snomed_data_path: Path,
        embeddings_path: Optional[Path] = None,
        top_k: int = 5,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize semantic resolver.

        Args:
            embedder: Embedding model instance
            snomed_data_path: Path to snomed_diabetes_subset.json
            embeddings_path: Path to pre-computed embeddings .npz file
                            If None, will look in data/reference/snomed_embeddings/{model}.npz
            top_k: Number of top matches to return
            confidence_threshold: Minimum cosine similarity for a match
        """
        self.embedder = embedder
        self.snomed_data_path = Path(snomed_data_path)
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold

        # Auto-detect embeddings path if not provided
        if embeddings_path is None:
            model_name = embedder.get_model_name()
            embeddings_path = (
                self.snomed_data_path.parent / "snomed_embeddings" / f"{model_name}.npz"
            )
        self.embeddings_path = Path(embeddings_path)

        # Data structures
        self.snomed_concepts: Dict[str, SNOMEDConcept] = {}
        self.concept_embeddings: Optional[np.ndarray] = None
        self.concept_ids: List[str] = []
        self.exact_match_index: Dict[str, str] = {}  # lowercase text -> concept_id

        # Load data
        self._load_snomed_concepts()
        self._load_embeddings()
        self._build_exact_match_index()

    def _load_snomed_concepts(self):
        """Load SNOMED concepts from JSON."""
        logger.info(f"Loading SNOMED concepts from {self.snomed_data_path}")

        with open(self.snomed_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        concepts = data.get('concepts', [])
        for concept_data in concepts:
            concept = SNOMEDConcept(
                cui=concept_data.get('cui'),
                snomed_code=concept_data.get('snomed_code'),
                display=concept_data.get('display'),
                synonyms=concept_data.get('synonyms', []),
                semantic_types=concept_data.get('semantic_types', [])
            )
            # Use snomed_code as primary key
            self.snomed_concepts[concept.snomed_code] = concept

        logger.info(f"Loaded {len(self.snomed_concepts)} SNOMED concepts")

    def _load_embeddings(self):
        """Load pre-computed embeddings from .npz file."""
        if not self.embeddings_path.exists():
            logger.warning(
                f"Pre-computed embeddings not found at {self.embeddings_path}. "
                f"Run precompute_embeddings.py first. "
                f"Resolver will work but will be slower (embedding on-the-fly)."
            )
            self.concept_embeddings = None
            return

        logger.info(f"Loading pre-computed embeddings from {self.embeddings_path}")

        data = np.load(self.embeddings_path, allow_pickle=True)
        self.concept_embeddings = data['embeddings']
        self.concept_ids = data['concept_ids'].tolist()

        logger.info(
            f"Loaded {len(self.concept_ids)} pre-computed embeddings "
            f"(dim: {self.concept_embeddings.shape[1]})"
        )

    def _build_exact_match_index(self):
        """Build index for exact match fast path."""
        for concept_id, concept in self.snomed_concepts.items():
            # Index display name
            self.exact_match_index[concept.display.lower()] = concept_id

            # Index all synonyms
            for synonym in concept.synonyms:
                self.exact_match_index[synonym.lower()] = concept_id

        logger.info(f"Built exact match index with {len(self.exact_match_index)} entries")

    def _exact_match(self, text: str) -> Optional[SNOMEDConcept]:
        """Fast path: exact match (case-insensitive)."""
        text_lower = text.lower().strip()
        concept_id = self.exact_match_index.get(text_lower)

        if concept_id:
            return self.snomed_concepts[concept_id]

        return None

    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        # scipy.spatial.distance.cosine returns distance (1 - similarity)
        return 1.0 - cosine(embedding1, embedding2)

    def resolve(
        self,
        entity_text: str,
        use_exact_match: bool = True
    ) -> List[Tuple[SNOMEDConcept, float]]:
        """
        Resolve entity text to SNOMED concepts.

        Args:
            entity_text: Entity mention from clinical text
            use_exact_match: Use exact match fast path if True

        Returns:
            List of (SNOMEDConcept, confidence) tuples, sorted by confidence descending
        """
        # Try exact match first
        if use_exact_match:
            exact_match = self._exact_match(entity_text)
            if exact_match:
                logger.debug(f"Exact match for '{entity_text}': {exact_match.display}")
                return [(exact_match, 1.0)]

        # Embedding-based search
        return self._semantic_search(entity_text)

    def _semantic_search(self, entity_text: str) -> List[Tuple[SNOMEDConcept, float]]:
        """
        Semantic search using embeddings.

        Args:
            entity_text: Entity mention

        Returns:
            Top-k matches with confidence scores
        """
        # Embed query text
        query_embedding = self.embedder.encode(entity_text)
        # encode() returns (1, embedding_dim) for single string, squeeze to (embedding_dim,)
        if query_embedding.ndim == 2:
            query_embedding = query_embedding.squeeze()

        # If pre-computed embeddings available, use them
        if self.concept_embeddings is not None:
            similarities = []
            for i, concept_id in enumerate(self.concept_ids):
                concept_embedding = self.concept_embeddings[i]
                similarity = self._cosine_similarity(query_embedding, concept_embedding)
                similarities.append((concept_id, similarity))

            # Sort by similarity descending
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Filter by threshold and take top-k
            results = []
            for concept_id, similarity in similarities[:self.top_k]:
                if similarity >= self.confidence_threshold:
                    concept = self.snomed_concepts[concept_id]
                    results.append((concept, similarity))

            return results

        else:
            # Fallback: compute embeddings on-the-fly (slower)
            logger.warning("Computing embeddings on-the-fly (slow). Pre-compute embeddings for better performance.")

            similarities = []
            for concept_id, concept in self.snomed_concepts.items():
                concept_embedding = self.embedder.encode(concept.display)
                if concept_embedding.ndim == 2:
                    concept_embedding = concept_embedding.squeeze()
                similarity = self._cosine_similarity(query_embedding, concept_embedding)
                similarities.append((concept, similarity))

            # Sort by similarity descending
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Filter by threshold and take top-k
            results = [
                (concept, similarity)
                for concept, similarity in similarities[:self.top_k]
                if similarity >= self.confidence_threshold
            ]

            return results

    def resolve_best(
        self,
        entity_text: str,
        use_exact_match: bool = True
    ) -> Optional[Tuple[SNOMEDConcept, float]]:
        """
        Resolve to best matching concept.

        Args:
            entity_text: Entity mention
            use_exact_match: Use exact match fast path

        Returns:
            (SNOMEDConcept, confidence) or None if no match above threshold
        """
        results = self.resolve(entity_text, use_exact_match=use_exact_match)
        if results:
            return results[0]
        return None

    def batch_resolve(
        self,
        entity_texts: List[str],
        use_exact_match: bool = True
    ) -> List[List[Tuple[SNOMEDConcept, float]]]:
        """
        Batch resolve multiple entity texts.

        Args:
            entity_texts: List of entity mentions
            use_exact_match: Use exact match fast path

        Returns:
            List of results for each entity
        """
        return [
            self.resolve(text, use_exact_match=use_exact_match)
            for text in entity_texts
        ]


def test_semantic_resolver():
    """Test semantic resolver with sample entities."""
    # Paths (adjust as needed)
    snomed_data_path = Path("data/reference/snomed_diabetes_subset.json")

    if not snomed_data_path.exists():
        print(f"SNOMED data not found at {snomed_data_path}")
        print("Run data preparation scripts first.")
        return

    print("Testing Semantic Resolver")
    print("=" * 60)

    # Test with SapBERT
    print("\nUsing SapBERT embedder")
    print("-" * 60)

    embedder = EmbeddingFactory.create_embedder("sapbert")

    resolver = SemanticResolver(
        embedder=embedder,
        snomed_data_path=snomed_data_path,
        top_k=3,
        confidence_threshold=0.6
    )

    # Sample entities
    test_entities = [
        "diabetes mellitus",
        "type 2 diabetes",
        "diabetic retinopathy",
        "hyperglycemia",
        "insulin",
        "DM",  # Common abbreviation
        "sugar disease",  # Colloquial term
    ]

    for entity_text in test_entities:
        print(f"\nQuery: '{entity_text}'")
        results = resolver.resolve(entity_text)

        if results:
            print(f"  Found {len(results)} matches:")
            for concept, confidence in results:
                print(f"    [{confidence:.3f}] {concept.snomed_code} | {concept.display}")
        else:
            print(f"  No matches found")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    test_semantic_resolver()
