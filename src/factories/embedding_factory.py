"""
Embedding factory for biomedical text embedding models.

Supports multiple biomedical embedding models for benchmarking:
- SapBERT: Optimized for medical concept normalization (default)
- BioBERT: General biomedical language understanding
- PubMedBERT: Trained on PubMed abstracts

All models produce dense embeddings for cosine similarity-based
semantic search.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union
from pathlib import Path
import torch


class BaseEmbedder(ABC):
    """
    Abstract base class for text embedding models.

    All embedders must implement the encode() method to produce
    numpy arrays of embeddings.
    """

    @abstractmethod
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Encode text(s) to dense embeddings.

        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for processing

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name for identification."""
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality."""
        pass


class SapBERTEmbedder(BaseEmbedder):
    """
    SapBERT embedder optimized for medical concept normalization.

    Model: cambridgeltl/SapBERT-from-PubMedBERT-fulltext
    Trained on: UMLS synonym pairs (self-alignment pre-training)
    Best for: Mapping clinical text to SNOMED concepts
    Embedding dim: 768
    """

    def __init__(self, model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"):
        """
        Initialize SapBERT model.

        Args:
            model_name: HuggingFace model identifier
        """
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "transformers library required for SapBERT. "
                "Install with: pip install transformers"
            )

        self._model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        # Use CUDA if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        print(f"Loaded SapBERT model: {model_name}")
        print(f"  Device: {self.device}")

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Encode texts using SapBERT."""
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )

            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Get embeddings (no gradient computation)
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Use [CLS] token embedding
                embeddings = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    @property
    def model_name(self) -> str:
        return "sapbert"

    @property
    def embedding_dim(self) -> int:
        return 768


class BioBERTEmbedder(BaseEmbedder):
    """
    BioBERT embedder for general biomedical text.

    Model: dmis-lab/biobert-base-cased-v1.2
    Trained on: PubMed abstracts + PMC full-text articles
    Best for: General biomedical language understanding
    Embedding dim: 768
    """

    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.2"):
        """Initialize BioBERT model."""
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "transformers library required. "
                "Install with: pip install transformers"
            )

        self._model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        print(f"Loaded BioBERT model: {model_name}")
        print(f"  Device: {self.device}")

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Encode texts using BioBERT."""
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )

            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)
                embeddings = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    @property
    def model_name(self) -> str:
        return "biobert"

    @property
    def embedding_dim(self) -> int:
        return 768


class PubMedBERTEmbedder(BaseEmbedder):
    """
    PubMedBERT embedder trained from scratch on PubMed.

    Model: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    Trained on: PubMed abstracts and full-text articles (from scratch, not BERT)
    Best for: PubMed-style biomedical text
    Embedding dim: 768
    """

    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"):
        """Initialize PubMedBERT model."""
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "transformers library required. "
                "Install with: pip install transformers"
            )

        self._model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        print(f"Loaded PubMedBERT model: {model_name}")
        print(f"  Device: {self.device}")

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Encode texts using PubMedBERT."""
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )

            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)
                embeddings = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    @property
    def model_name(self) -> str:
        return "pubmedbert"

    @property
    def embedding_dim(self) -> int:
        return 768


class EmbeddingFactory:
    """
    Factory for creating embedding models.

    Supports:
    - sapbert: SapBERT (default, best for concept normalization)
    - biobert: BioBERT (benchmark)
    - pubmedbert: PubMedBERT (benchmark)
    """

    _registry = {
        'sapbert': SapBERTEmbedder,
        'biobert': BioBERTEmbedder,
        'pubmedbert': PubMedBERTEmbedder,
    }

    @staticmethod
    def create(model_name: str) -> BaseEmbedder:
        """
        Create an embedding model by name.

        Args:
            model_name: One of 'sapbert', 'biobert', 'pubmedbert'

        Returns:
            BaseEmbedder instance

        Raises:
            ValueError: If model_name not recognized
        """
        model_name = model_name.lower()

        if model_name not in EmbeddingFactory._registry:
            available = ', '.join(EmbeddingFactory._registry.keys())
            raise ValueError(
                f"Unknown embedding model: {model_name}. "
                f"Available models: {available}"
            )

        model_class = EmbeddingFactory._registry[model_name]
        return model_class()

    @staticmethod
    def list_models() -> List[str]:
        """List all available embedding models."""
        return list(EmbeddingFactory._registry.keys())


    # Convenience function
    def create_embedder(model_name: str = "sapbert") -> BaseEmbedder:
        """
        Create an embedding model.

        Args:
            model_name: Model identifier (default: 'sapbert')

        Returns:
            BaseEmbedder instance
        """
        return EmbeddingFactory.create(model_name)


if __name__ == "__main__":
    # Test the embedding factory
    print("Testing Embedding Factory\n")

    # Test with SapBERT (default)
    print("=" * 60)
    embedder = create_embedder("sapbert")

    test_texts = [
        "diabetes mellitus",
        "Type 2 diabetes with hyperglycemia",
        "diabetic neuropathy"
    ]

    print(f"\nEmbedding test texts...")
    embeddings = embedder.encode(test_texts)

    print(f"Shape: {embeddings.shape}")
    print(f"Model: {embedder.model_name}")
    print(f"Embedding dim: {embedder.embedding_dim}")

    # Test cosine similarity
    from scipy.spatial.distance import cosine

    sim_01 = 1 - cosine(embeddings[0], embeddings[1])
    sim_02 = 1 - cosine(embeddings[0], embeddings[2])

    print(f"\nCosine similarities:")
    print(f"  '{test_texts[0]}' vs '{test_texts[1]}': {sim_01:.3f}")
    print(f"  '{test_texts[0]}' vs '{test_texts[2]}': {sim_02:.3f}")

    print("\nEmbedding factory test passed")
