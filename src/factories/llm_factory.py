"""
LLM Factory - Swappable Large Language Model interfaces.

Provides factory pattern for different LLM approaches via Ollama:
- Qwen 2.5 7B: Fast, strong reasoning for medical text
- Llama 3.1 8B: General purpose, good medical knowledge
- Llama 3.2 3B: Lightweight alternative

All LLMs accessed via Ollama local server.

Usage in Phase 4 (Two-layer LLM validation):
1. Layer 1: Evidence existence check
   - Does the clinical note contain evidence for this condition?
   - Binary yes/no with confidence

2. Layer 2: KG inference validity check
   - Is this inference path clinically valid?
   - Validates multi-hop reasoning
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured LLM response."""
    content: str
    confidence: float  # 0.0-1.0
    reasoning: Optional[str] = None
    raw_response: Optional[Dict] = None


class BaseLLM(ABC):
    """Abstract base class for LLM interfaces."""

    def __init__(self, model_name: str, temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        self.client = None

    @abstractmethod
    def load_model(self):
        """Initialize LLM client."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512
    ) -> LLMResponse:
        """
        Generate response from LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with content and confidence
        """
        pass

    def get_model_name(self) -> str:
        """Return the model identifier."""
        return self.model_name


class OllamaLLM(BaseLLM):
    """
    Ollama LLM client for local models.

    Supports:
    - qwen2.5:7b
    - llama3.1:8b
    - llama3.2:3b
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        ollama_host: str = "http://localhost:11434"
    ):
        super().__init__(model_name, temperature)
        self.ollama_host = ollama_host

    def load_model(self):
        """Initialize Ollama client."""
        try:
            import ollama

            self.client = ollama.Client(host=self.ollama_host)

            # Test connection by listing models
            models = self.client.list()
            available_models = [m['name'] for m in models['models']]

            logger.info(f"Connected to Ollama at {self.ollama_host}")
            logger.info(f"Available models: {', '.join(available_models)}")

            # Check if target model is available
            if not any(self.model_name in m for m in available_models):
                logger.warning(
                    f"Model {self.model_name} not found. "
                    f"Pull it with: ollama pull {self.model_name}"
                )

        except ImportError:
            raise ImportError(
                "Ollama Python client not installed. Install with: pip install ollama"
            )
        except Exception as e:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.ollama_host}. "
                f"Ensure Ollama is running (ollama serve). Error: {e}"
            )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512
    ) -> LLMResponse:
        """Generate response using Ollama."""
        if self.client is None:
            self.load_model()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": max_tokens,
                }
            )

            content = response['message']['content']

            # Extract confidence if present in structured output
            confidence = self._extract_confidence(content)

            return LLMResponse(
                content=content,
                confidence=confidence,
                raw_response=response
            )

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    def _extract_confidence(self, content: str) -> float:
        """
        Extract confidence score from LLM output.

        Looks for patterns like:
        - "Confidence: 0.95"
        - "confidence: high" → 0.9
        - "confidence: medium" → 0.6
        - "confidence: low" → 0.3
        """
        content_lower = content.lower()

        # Try to extract numeric confidence
        import re
        match = re.search(r'confidence[:\s]+([0-9.]+)', content_lower)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

        # Try to extract qualitative confidence
        if 'confidence: high' in content_lower or 'high confidence' in content_lower:
            return 0.9
        elif 'confidence: medium' in content_lower or 'medium confidence' in content_lower:
            return 0.6
        elif 'confidence: low' in content_lower or 'low confidence' in content_lower:
            return 0.3

        # Default: assume high confidence if no explicit statement
        return 0.8

    def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[Dict] = None
    ) -> Dict:
        """
        Generate structured JSON response.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            response_format: JSON schema for expected response

        Returns:
            Parsed JSON dictionary
        """
        if response_format:
            # Add JSON schema to system prompt
            schema_prompt = f"\n\nRespond with valid JSON matching this schema:\n{json.dumps(response_format, indent=2)}"
            system_prompt = (system_prompt or "") + schema_prompt

        response = self.generate(prompt, system_prompt)

        # Try to parse JSON from response
        try:
            # Extract JSON from markdown code blocks if present
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.warning(f"Raw response: {response.content}")
            # Return raw content wrapped in dict
            return {"raw_response": response.content, "parse_error": str(e)}


class Qwen25LLM(OllamaLLM):
    """
    Qwen 2.5 7B via Ollama.

    Model: qwen2.5:7b
    Strengths: Fast inference, strong reasoning, good medical knowledge
    """

    def __init__(self, temperature: float = 0.0, ollama_host: str = "http://localhost:11434"):
        super().__init__(
            model_name="qwen2.5:7b",
            temperature=temperature,
            ollama_host=ollama_host
        )


class Llama31LLM(OllamaLLM):
    """
    Llama 3.1 8B via Ollama.

    Model: llama3.1:8b
    Strengths: General purpose, good instruction following
    """

    def __init__(self, temperature: float = 0.0, ollama_host: str = "http://localhost:11434"):
        super().__init__(
            model_name="llama3.1:8b",
            temperature=temperature,
            ollama_host=ollama_host
        )


class Llama32LLM(OllamaLLM):
    """
    Llama 3.2 3B via Ollama.

    Model: llama3.2:3b
    Strengths: Lightweight, faster inference
    """

    def __init__(self, temperature: float = 0.0, ollama_host: str = "http://localhost:11434"):
        super().__init__(
            model_name="llama3.2:3b",
            temperature=temperature,
            ollama_host=ollama_host
        )


class LLMFactory:
    """
    Factory for creating LLM clients.

    Usage:
        llm = LLMFactory.create_llm("qwen2.5")
        response = llm.generate("Does this patient have diabetes?")
    """

    _llms = {
        "qwen2.5": Qwen25LLM,
        "qwen": Qwen25LLM,
        "llama3.1": Llama31LLM,
        "llama": Llama31LLM,
        "llama3.2": Llama32LLM,
    }

    @classmethod
    def create_llm(
        cls,
        model_name: str,
        temperature: float = 0.0,
        ollama_host: str = "http://localhost:11434"
    ) -> BaseLLM:
        """
        Create an LLM client.

        Args:
            model_name: One of "qwen2.5", "llama3.1", "llama3.2"
            temperature: Sampling temperature (0.0 = deterministic)
            ollama_host: Ollama server URL

        Returns:
            LLM client instance

        Raises:
            ValueError: If model_name not recognized
        """
        if model_name not in cls._llms:
            available = ", ".join(cls._llms.keys())
            raise ValueError(
                f"Unknown LLM model: {model_name}. "
                f"Available models: {available}"
            )

        llm_class = cls._llms[model_name]
        return llm_class(temperature=temperature, ollama_host=ollama_host)

    @classmethod
    def list_models(cls) -> List[str]:
        """List available LLM models."""
        return list(cls._llms.keys())


# Prompt templates for Phase 4 validation

EVIDENCE_EXISTENCE_PROMPT = """You are a medical expert reviewing clinical documentation.

Clinical Note:
{clinical_note}

Condition: {condition_name} (ICD-10: {icd10_code})

Question: Does the clinical note contain explicit evidence for this condition?

Analyze the note and respond in JSON format:
{{
  "has_evidence": true/false,
  "confidence": 0.0-1.0,
  "evidence_spans": ["quote from note", ...],
  "reasoning": "brief explanation"
}}

Only return "has_evidence": true if there is explicit mention or strong clinical evidence.
Be conservative - when in doubt, mark as false."""


KG_INFERENCE_VALIDITY_PROMPT = """You are a medical expert validating clinical reasoning.

Inference Path:
{inference_path}

Clinical Context:
{clinical_note}

Question: Is this inference path clinically valid and supported by the note?

Analyze the reasoning chain and respond in JSON format:
{{
  "is_valid": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "explain why the inference is valid or invalid",
  "issues": ["potential problems with the inference", ...]
}}

Validate that:
1. Each step in the inference chain is medically sound
2. The clinical note supports the inference
3. No logical leaps are made

Be conservative - flag any uncertain reasoning."""


def validate_evidence_existence(
    llm: BaseLLM,
    clinical_note: str,
    condition_name: str,
    icd10_code: str
) -> Dict:
    """
    Layer 1 validation: Check if evidence exists in clinical note.

    Args:
        llm: LLM instance
        clinical_note: Clinical text
        condition_name: Condition being validated
        icd10_code: ICD-10 code

    Returns:
        Validation result dictionary
    """
    prompt = EVIDENCE_EXISTENCE_PROMPT.format(
        clinical_note=clinical_note,
        condition_name=condition_name,
        icd10_code=icd10_code
    )

    system_prompt = "You are a medical documentation expert. Be precise and conservative."

    response = llm.generate_structured(
        prompt=prompt,
        system_prompt=system_prompt,
        response_format={
            "has_evidence": "boolean",
            "confidence": "number",
            "evidence_spans": "array",
            "reasoning": "string"
        }
    )

    return response


def validate_kg_inference(
    llm: BaseLLM,
    inference_path: str,
    clinical_note: str
) -> Dict:
    """
    Layer 2 validation: Validate KG inference path.

    Args:
        llm: LLM instance
        inference_path: Description of inference reasoning
        clinical_note: Clinical text

    Returns:
        Validation result dictionary
    """
    prompt = KG_INFERENCE_VALIDITY_PROMPT.format(
        inference_path=inference_path,
        clinical_note=clinical_note
    )

    system_prompt = "You are a clinical reasoning expert. Validate inference logic rigorously."

    response = llm.generate_structured(
        prompt=prompt,
        system_prompt=system_prompt,
        response_format={
            "is_valid": "boolean",
            "confidence": "number",
            "reasoning": "string",
            "issues": "array"
        }
    )

    return response


def test_llm_factory():
    """Test LLM factory with sample prompts."""
    sample_note = (
        "58-year-old male with longstanding type 2 diabetes mellitus. "
        "HbA1c 8.5%. Patient reports poor glycemic control. "
        "Currently on metformin and glipizide."
    )

    print("Testing LLM Factory")
    print("=" * 60)
    print(f"Sample clinical note:\n{sample_note}\n")

    for model_name in ["qwen2.5", "llama3.1"]:
        try:
            print(f"\n{model_name.upper()} Results:")
            print("-" * 60)

            llm = LLMFactory.create_llm(model_name, temperature=0.0)

            # Test evidence existence
            result = validate_evidence_existence(
                llm=llm,
                clinical_note=sample_note,
                condition_name="Type 2 diabetes mellitus",
                icd10_code="E11.65"
            )

            print(f"Evidence existence check:")
            print(f"  Has evidence: {result.get('has_evidence')}")
            print(f"  Confidence: {result.get('confidence')}")
            print(f"  Reasoning: {result.get('reasoning', 'N/A')}")

        except Exception as e:
            print(f"  Error: {e}")
            print(f"  (Make sure Ollama is running and model is pulled)")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    test_llm_factory()
