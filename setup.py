from setuptools import setup, find_packages

setup(
    name="clinical-icd10-coding",
    version="0.1.0",
    description="Context-aware clinical NLP system for ICD-10 coding",
    author="Clinical NLP Team",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.115.0",
        "uvicorn>=0.30.0",
        "pydantic>=2.9.0",
        "spacy>=3.7.6",
        "scispacy>=0.5.5",
        "rapidfuzz>=3.9.0",
        "networkx>=3.3",
        "ollama>=0.3.0",
        "groq>=0.11.0",
        "pandas>=2.2.0",
        "tabulate>=0.9.0",
        "rich>=13.7.0",
        "pytest>=8.3.0",
        "jupyter>=1.0.0",
        "matplotlib>=3.9.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
