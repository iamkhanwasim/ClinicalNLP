"""
Configuration settings for the Clinical NLP → ICD-10 Coding POC.

Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
CLINICAL_NOTES_DIR = DATA_DIR / "clinical_notes"
REFERENCE_DATA_DIR = DATA_DIR / "reference"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"

# Reference data files
ICD10_CODES_PATH = os.getenv(
    "ICD10_CODES_PATH",
    str(REFERENCE_DATA_DIR / "icd10_diabetes_codes.json")
)
SNOMED_SUBSET_PATH = os.getenv(
    "SNOMED_SUBSET_PATH",
    str(REFERENCE_DATA_DIR / "snomed_diabetes_subset.json")
)
SNOMED_CROSSWALK_PATH = os.getenv(
    "SNOMED_CROSSWALK_PATH",
    str(REFERENCE_DATA_DIR / "snomed_icd10_crosswalk.json")
)
DIABETES_KG_PATH = os.getenv(
    "DIABETES_KG_PATH",
    str(REFERENCE_DATA_DIR / "diabetes_knowledge_graph.json")
)

# NER Model Configuration
NER_MODEL = os.getenv("NER_MODEL", "en_ner_bc5cdr_md")
NER_MODEL_SECONDARY = os.getenv("NER_MODEL_SECONDARY", "en_core_med7_trf")
USE_SECONDARY_NER = os.getenv("USE_SECONDARY_NER", "false").lower() == "true"

# Resolution Mode
ResolutionMode = Literal["direct_icd10", "snomed"]
RESOLUTION_MODE: ResolutionMode = os.getenv("RESOLUTION_MODE", "direct_icd10")

# Thresholds
FUZZY_MATCH_THRESHOLD = float(os.getenv("FUZZY_MATCH_THRESHOLD", "0.85"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
MIN_ENTITY_LENGTH = int(os.getenv("MIN_ENTITY_LENGTH", "3"))

# LLM Configuration (for Approach 4)
LLMProvider = Literal["groq", "ollama"]
LLM_PROVIDER: LLMProvider = os.getenv("LLM_PROVIDER", "groq")

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Ollama Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Entity type mapping from NER labels to our types
ENTITY_TYPE_MAPPING = {
    "DISEASE": "condition",
    "CHEMICAL": "medication",
    "PROBLEM": "condition",
    "TREATMENT": "procedure",
    "TEST": "lab_value",
    "MEDICATION": "medication",
    "DRUG": "medication",
    "DIAGNOSIS": "condition",
    "SYMPTOM": "condition",
}


def get_clinical_note_path(note_id: str) -> Path:
    """Get the file path for a clinical note by ID."""
    return CLINICAL_NOTES_DIR / f"{note_id}.txt"


def get_ground_truth_path(note_id: str) -> Path:
    """Get the file path for ground truth data by note ID."""
    return GROUND_TRUTH_DIR / f"{note_id}_expected.json"


def validate_paths() -> bool:
    """Validate that required data files exist."""
    required_paths = [
        ICD10_CODES_PATH,
    ]

    missing = []
    for path in required_paths:
        if not Path(path).exists():
            missing.append(path)

    if missing:
        print(f"Warning: Missing required files:")
        for path in missing:
            print(f"  - {path}")
        return False

    return True


# Entity labels that should be treated as conditions
CONDITION_LABELS = {"DISEASE", "PROBLEM", "DIAGNOSIS", "SYMPTOM"}

# Entity labels that should be treated as medications
MEDICATION_LABELS = {"CHEMICAL", "MEDICATION", "DRUG"}

# Minimum confidence for including an extraction in results
MIN_EXTRACTION_CONFIDENCE = 0.5


if __name__ == "__main__":
    # Print configuration for debugging
    print("=== Clinical NLP Configuration ===")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"ICD-10 Codes: {ICD10_CODES_PATH}")
    print(f"Resolution Mode: {RESOLUTION_MODE}")
    print(f"NER Model: {NER_MODEL}")
    print(f"Fuzzy Match Threshold: {FUZZY_MATCH_THRESHOLD}")
    print(f"LLM Provider: {LLM_PROVIDER}")
    print(f"\nValidating paths...")
    if validate_paths():
        print("All required files present")
    else:
        print("Some required files are missing")
