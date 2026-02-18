"""
NER Factory - Swappable Named Entity Recognition models.

Provides factory pattern for different NER approaches:
- Stanza: Clinical NER trained on i2b2 datasets
- ScispaCy: Biomedical NER trained on BC5CDR
- Med7: Medication-focused NER
- BioBERT NER: Transformer-based NER via HuggingFace

All extractors return standardized Entity objects with:
- text: Entity mention
- label: Entity type (PROBLEM, TREATMENT, TEST, etc.)
- start_char: Character offset
- end_char: Character offset
- confidence: 0.0-1.0 (if available)
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path
import logging

from ..shared.models import Entity

logger = logging.getLogger(__name__)


class BaseNERExtractor(ABC):
    """Abstract base class for NER extractors."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    @abstractmethod
    def load_model(self):
        """Load the NER model."""
        pass

    @abstractmethod
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract named entities from text.

        Args:
            text: Input clinical text

        Returns:
            List of Entity objects
        """
        pass

    def get_model_name(self) -> str:
        """Return the model identifier."""
        return self.model_name


class StanzaNERExtractor(BaseNERExtractor):
    """
    Stanza clinical NER extractor.

    Model: i2b2 clinical NER model
    Entities: PROBLEM, TREATMENT, TEST
    Language: English
    """

    def __init__(self, model_name: str = "stanza_clinical"):
        super().__init__(model_name)
        self.nlp = None

    def load_model(self):
        """Load Stanza biomedical NER model."""
        try:
            import stanza

            # Use default English NER model with biomedical entities
            # MIMIC package only has tokenizer, not NER
            logger.info(f"Downloading Stanza English NER model...")            
            stanza.download('en', package='i2b2', processors='tokenize,ner')            

            logger.info(f"Loading Stanza biomedical NER model...")
            self.nlp = stanza.Pipeline(
                lang='en',
                processors={
                    'tokenize': 'default',
                    'ner': 'i2b2'
                },
                use_gpu=False,  # Set to True if GPU available
                logging_level='ERROR'
            )
            logger.info(f"Stanza NER model loaded")

        except ImportError:
            raise ImportError(
                "Stanza not installed. Install with: pip install stanza"
            )

    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities using Stanza."""
        if self.nlp is None:
            self.load_model()

        doc = self.nlp(text)
        entities = []

        for ent in doc.entities:
            entities.append(Entity(
                text=ent.text,
                label=ent.type,
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=1.0  # Stanza doesn't provide confidence scores
            ))

        return entities


class ScispaCyNERExtractor(BaseNERExtractor):
    """
    ScispaCy biomedical NER extractor.

    Model: en_ner_bc5cdr_md (BC5CDR corpus)
    Entities: DISEASE, CHEMICAL
    Optimized for: Disease and drug mentions
    """

    def __init__(self, model_name: str = "scispacy_bc5cdr"):
        super().__init__(model_name)
        self.nlp = None

    def load_model(self):
        """Load ScispaCy BC5CDR NER model."""
        try:
            import spacy

            # Model: en_ner_bc5cdr_md
            # Install: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

            logger.info(f"Loading ScispaCy BC5CDR NER model...")
            self.nlp = spacy.load("en_ner_bc5cdr_md")
            logger.info(f"ScispaCy model loaded")

        except ImportError:
            raise ImportError(
                "Spacy not installed. Install with: pip install spacy"
            )
        except OSError:
            raise OSError(
                "ScispaCy BC5CDR model not found. Install with:\n"
                "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
            )

    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities using ScispaCy."""
        if self.nlp is None:
            self.load_model()

        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entities.append(Entity(
                text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=1.0  # ScispaCy doesn't provide confidence scores
            ))

        return entities


class Med7NERExtractor(BaseNERExtractor):
    """
    Med7 medication NER extractor.

    Model: en_core_med7_lg
    Entities: DRUG, DOSAGE, DURATION, FORM, FREQUENCY, ROUTE, STRENGTH
    Optimized for: Medication mentions
    """

    def __init__(self, model_name: str = "med7"):
        super().__init__(model_name)
        self.nlp = None

    def load_model(self):
        """Load Med7 NER model."""
        try:
            import spacy

            # Model: en_core_med7_lg
            # Install: pip install https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl

            logger.info(f"Loading Med7 NER model...")
            self.nlp = spacy.load("en_core_med7_lg")
            logger.info(f"Med7 model loaded")

        except ImportError:
            raise ImportError(
                "Spacy not installed. Install with: pip install spacy"
            )
        except OSError:
            raise OSError(
                "Med7 model not found. Install with:\n"
                "pip install https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl"
            )

    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities using Med7."""
        if self.nlp is None:
            self.load_model()

        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entities.append(Entity(
                text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=1.0  # Med7 doesn't provide confidence scores
            ))

        return entities


class BioBERTNERExtractor(BaseNERExtractor):
    """
    BioBERT NER extractor using HuggingFace pipeline.

    Model: dmis-lab/biobert-base-cased-v1.2 (finetuned for NER)
    Entities: Depends on finetuned model
    Transformer-based: Provides confidence scores
    """

    def __init__(self, model_name: str = "biobert_ner"):
        super().__init__(model_name)
        self.pipeline = None

    def load_model(self):
        """Load BioBERT NER pipeline."""
        try:
            from transformers import pipeline

            # Using dmis-lab/biobert-v1.1 base model finetuned for NER
            # This is a general biomedical NER model
            model_id = "d4data/biomedical-ner-all"

            logger.info(f"Loading BioBERT NER model: {model_id}...")
            logger.info(f"(This will download ~450MB on first run)")

            self.pipeline = pipeline(
                "ner",
                model=model_id,
                aggregation_strategy="simple"  # Aggregate subword tokens
            )

            logger.info(f"BioBERT NER model loaded")

        except ImportError:
            raise ImportError(
                "Transformers not installed. Install with: pip install transformers"
            )

    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities using BioBERT NER."""
        if self.pipeline is None:
            self.load_model()

        # Run NER pipeline
        results = self.pipeline(text)

        entities = []
        for ent in results:
            # HuggingFace NER output format:
            # {'entity_group': 'DISEASE', 'score': 0.99, 'word': 'diabetes', 'start': 10, 'end': 18}
            entities.append(Entity(
                text=ent['word'],
                label=ent['entity_group'],
                start_char=ent['start'],
                end_char=ent['end'],
                confidence=ent['score']
            ))

        return entities


class NERFactory:
    """
    Factory for creating NER extractors.

    Usage:
        extractor = NERFactory.create_extractor("stanza")
        entities = extractor.extract_entities("Patient has diabetes mellitus.")
    """

    _extractors = {
        "stanza": StanzaNERExtractor,
        "stanza_clinical": StanzaNERExtractor,
        "scispacy": ScispaCyNERExtractor,
        "scispacy_bc5cdr": ScispaCyNERExtractor,
        "med7": Med7NERExtractor,
        "biobert": BioBERTNERExtractor,
        "biobert_ner": BioBERTNERExtractor,
    }

    @classmethod
    def create_extractor(cls, model_name: str) -> BaseNERExtractor:
        """
        Create a NER extractor.

        Args:
            model_name: One of "stanza", "scispacy", "med7", "biobert"

        Returns:
            NER extractor instance

        Raises:
            ValueError: If model_name not recognized
        """
        if model_name not in cls._extractors:
            available = ", ".join(cls._extractors.keys())
            raise ValueError(
                f"Unknown NER model: {model_name}. "
                f"Available models: {available}"
            )

        extractor_class = cls._extractors[model_name]
        return extractor_class(model_name=model_name)

    @classmethod
    def list_models(cls) -> List[str]:
        """List available NER models."""
        return list(cls._extractors.keys())


def test_ner_extractors():
    """Test all NER extractors with sample text."""
    sample_text = (
        "Patient has type 2 diabetes mellitus with diabetic retinopathy. "
        "Started on metformin 500mg twice daily and insulin glargine 10 units at bedtime."
    )

    print("Testing NER Extractors")
    print("=" * 60)
    print(f"Input text:\n{sample_text}\n")

    for model_name in ["scispacy"]: #, ["stanza","scispacy", "med7", "biobert"]:
        try:
            print(f"\n{model_name.upper()} Results:")
            print("-" * 60)

            extractor = NERFactory.create_extractor(model_name)
            entities = extractor.extract_entities(sample_text)

            print(f"Found {len(entities)} entities:")
            for ent in entities:
                conf_str = f" (conf: {ent.confidence:.2f})" if ent.confidence < 1.0 else ""
                print(f"  [{ent.label}] {ent.text}{conf_str}")

        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    test_ner_extractors()
