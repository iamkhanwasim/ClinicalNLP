# Clinical NLP → ICD-10 Coding POC

Context-aware clinical NLP system that maps unstructured clinical notes to ICD-10 billing codes with evidence spans.

## Project Overview

This POC is scoped to **diabetes** as the target domain. Three approaches are compared:

| Approach | Extraction Method | Cross-Sentence Context | Key Tradeoff |
|----------|------------------|----------------------|--------------|
| **Approach 3** | NER (BioBERT/ScispaCy) | None (sentence-level only) | Fast, deterministic, but misses context |
| **Approach 3+KG** | NER + Knowledge Graph traversal | Graph-based reasoning | Explainable, structured, but requires KG construction |
| **Approach 4** | NER + LLM context enrichment | LLM reads full document | Flexible, but non-deterministic |

## Features

- **Evidence-based coding**: Every ICD-10 code links to specific text spans in the clinical note
- **Pluggable architecture**: Supports both direct ICD-10 matching and SNOMED CT intermediate layer
- **Multiple approaches**: Compare NER baseline, Knowledge Graph enrichment, and LLM enrichment
- **REST API**: FastAPI endpoints for extraction and comparison
- **Comprehensive evaluation**: Side-by-side approach comparison with metrics

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys (if using Groq for LLM approach)
```

### Run the API

```bash
uvicorn api.main:app --reload
```

Visit http://localhost:8000/docs for interactive API documentation.

## Repository Structure

```
clinical-icd10-coding/
├── data/
│   ├── clinical_notes/       # Sample clinical notes
│   ├── reference/             # ICD-10 codes, SNOMED data, KG
│   └── ground_truth/          # Expected outputs for evaluation
├── src/
│   ├── shared/                # Common components (NER, resolvers, etc.)
│   ├── approach_3/            # Baseline NER → ICD-10 pipeline
│   ├── approach_3_kg/         # NER + Knowledge Graph enrichment
│   └── approach_4/            # NER + LLM enrichment
├── api/                       # FastAPI application
├── evaluation/                # Evaluation metrics and comparison
├── notebooks/                 # Exploratory analysis
└── tests/                     # Unit and integration tests
```

## Tech Stack

- **Python 3.11+**
- **NER**: ScispaCy, Med7
- **String Matching**: rapidfuzz
- **Knowledge Graph**: NetworkX
- **LLM**: Groq API (Llama 3.1 8B) or Ollama
- **API**: FastAPI + Uvicorn
- **Data Models**: Pydantic v2

## Development

### Running Tests

```bash
pytest tests/
```

### Data Preparation

See `clinical-icd10-project-plan.md` for detailed instructions on:
- Downloading ICD-10 codes from CDC
- Obtaining SNOMED CT data (requires UMLS account)
- Creating clinical notes

## License

This is a proof-of-concept project for educational and research purposes.
