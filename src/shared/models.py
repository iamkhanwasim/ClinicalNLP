"""
Pydantic data models for the Clinical NLP → ICD-10 Coding POC.

Updated architecture with:
- Inference strength tracking ("explicit" | "strong_suggestion" | "weak")
- needs_review flags for non-explicit inferences
- SNOMED CT intermediate layer
- Support for factory pattern (model names tracked)
"""

from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


# Type aliases for clarity
InferenceStrength = Literal["explicit", "strong_suggestion", "weak"]
EntityType = Literal["condition", "medication", "lab_value", "procedure", "symptom", "other"]


class ClinicalNote(BaseModel):
    """
    Represents a parsed clinical note with sections.

    Attributes:
        note_id: Unique identifier for the note
        raw_text: The complete unstructured text
        sections: Dictionary mapping section names to section text
        section_offsets: Character offsets where each section starts in raw_text
        created_at: Timestamp when parsed
    """
    note_id: str
    raw_text: str
    sections: Dict[str, str] = Field(default_factory=dict)
    section_offsets: Dict[str, int] = Field(default_factory=dict)
    created_at: Optional[datetime] = Field(default_factory=datetime.now)

    @field_validator('note_id')
    @classmethod
    def validate_note_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('note_id cannot be empty')
        return v.strip()


class EntitySpan(BaseModel):
    """
    Represents a single entity extracted by NER with its location.

    Attributes:
        text: The extracted entity text
        start_char: Character offset where entity starts in full note
        end_char: Character offset where entity ends in full note
        section: Section name where the entity was found
        entity_type: Type of entity
        label: Original NER label from the model
        source_model: Which NER model produced this entity
        confidence: Optional confidence score from NER model
    """
    text: str
    start_char: int
    end_char: int
    section: str
    entity_type: EntityType
    label: Optional[str] = None
    source_model: str
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator('start_char', 'end_char')
    @classmethod
    def validate_char_positions(cls, v: int) -> int:
        if v < 0:
            raise ValueError('Character positions must be non-negative')
        return v


class NERResult(BaseModel):
    """
    Collection of entities extracted from a clinical note.

    Attributes:
        entities: List of extracted EntitySpan objects
        model_name: Name of the NER model(s) used
        processing_time_ms: Time taken for NER extraction
    """
    entities: List[EntitySpan] = Field(default_factory=list)
    model_name: str
    processing_time_ms: Optional[float] = None

    def get_entities_by_type(self, entity_type: str) -> List[EntitySpan]:
        """Filter entities by type."""
        return [e for e in self.entities if e.entity_type == entity_type]

    def get_entities_by_section(self, section: str) -> List[EntitySpan]:
        """Filter entities by section."""
        return [e for e in self.entities if e.section == section]


class EvidenceSpan(BaseModel):
    """
    Represents a text span that provides evidence for an inference.

    Attributes:
        text: The evidence text from the clinical note
        section: Section where the evidence was found
        char_start: Starting character position in the full note
        char_end: Ending character position in the full note
        reasoning: Explanation of why this is evidence
    """
    text: str
    section: str
    char_start: int
    char_end: int
    reasoning: Optional[str] = None


class SNOMEDConcept(BaseModel):
    """
    Represents a SNOMED CT concept.

    Attributes:
        cui: Concept Unique Identifier from UMLS
        snomed_code: SNOMED CT code
        display: Human-readable display name
        semantic_types: UMLS semantic types (optional, list)
        synonyms: Alternative names for this concept
    """
    cui: str
    snomed_code: str
    display: str
    semantic_types: List[str] = Field(default_factory=list)
    synonyms: List[str] = Field(default_factory=list)


class Entity(BaseModel):
    """
    Simple entity extracted by NER models.

    Used by NER factory extractors.

    Attributes:
        text: Entity mention text
        label: Entity type/label from NER model
        start_char: Character offset start
        end_char: Character offset end
        confidence: Optional confidence score
    """
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float = 1.0


class ICDCode(BaseModel):
    """
    Represents an ICD-10-CM code with metadata.

    Attributes:
        code: ICD-10 code (e.g., "E11.65")
        display: Full text description of the code
        billable: Whether this code can be used for billing
        category: Category prefix (e.g., "E11" for Type 2 diabetes codes)
        hcc: Whether this is an HCC (high-severity) code
    """
    code: str
    display: str
    billable: bool = True
    category: Optional[str] = None
    hcc: bool = False

    @field_validator('code')
    @classmethod
    def validate_icd_code(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('ICD code cannot be empty')
        v = v.strip().upper()
        if not v[0].isalpha():
            raise ValueError('ICD-10 code must start with a letter')
        return v


class ICD10Code(BaseModel):
    """
    ICD-10 code with detailed mapping metadata.

    Used by ICD10Mapper for crosswalk-based mapping.

    Attributes:
        code: ICD-10 code (e.g., "E11.65")
        display: Full text description
        confidence: Overall mapping confidence (0.0-1.0)
        source: Source of the mapping (e.g., "crosswalk")
        billable: Whether this code can be used for billing
        hcc: Whether this is an HCC (high-severity) code
        inference_strength: Strength of the inference
        map_priority: Priority from crosswalk (1 = highest)
        map_rule: Optional mapping rule/condition
        specificity: Code specificity score (higher = more specific)
    """
    code: str
    display: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = "crosswalk"
    billable: bool = True
    hcc: bool = False
    inference_strength: InferenceStrength = "weak"
    map_priority: int = 1
    map_rule: Optional[str] = None
    specificity: int = 0


class EnrichedEntity(BaseModel):
    """
    Entity enriched with additional context from KG or LLM.

    Used in Approach 3+KG and Approach 4.

    Attributes:
        original_entity: The original NER entity
        enriched_condition: Enhanced description with context
        type_specificity: Type 1 | Type 2 | unspecified
        severity: controlled | uncontrolled | unspecified
        complications: List of inferred complications
        evidence: List of evidence spans supporting enrichment
        confidence: Confidence in the enrichment
        inference_strength: How strong is this inference
        enrichment_source: knowledge_graph | llm
        reasoning_trail: Explanation of how enrichment was derived
    """
    original_entity: EntitySpan
    enriched_condition: str
    type_specificity: str = "unspecified"
    severity: str = "unspecified"
    complications: List[str] = Field(default_factory=list)
    evidence: List[EvidenceSpan] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    inference_strength: InferenceStrength
    enrichment_source: Literal["knowledge_graph", "llm"]
    reasoning_trail: str


class ExtractionResult(BaseModel):
    """
    Represents a single condition extracted and coded from a clinical note.

    Attributes:
        condition: Human-readable description of the condition
        snomed_concept: Resolved SNOMED CT concept
        icd10_code: Final ICD-10 code
        confidence: Overall confidence score (0.0 to 1.0)
        inference_strength: Strength of the clinical inference
        needs_review: True if inference_strength is not "explicit"
        review_reason: Why review is needed (if needs_review=True)
        evidence_spans: List of text spans supporting this code
        enrichment_reasoning: Explanation of context-aware enrichment (if any)
        source_entity: Optional reference to the original NER entity
    """
    condition: str
    snomed_concept: SNOMEDConcept
    icd10_code: ICDCode
    confidence: float = Field(ge=0.0, le=1.0)
    inference_strength: InferenceStrength
    needs_review: bool
    review_reason: str = ""
    evidence_spans: List[EvidenceSpan] = Field(default_factory=list)
    enrichment_reasoning: str = ""
    source_entity: Optional[EntitySpan] = None

    @field_validator('needs_review', mode='before')
    @classmethod
    def compute_needs_review(cls, v, info):
        """Auto-compute needs_review based on inference_strength."""
        if v is not None:
            return v
        # If not explicitly set, compute from inference_strength
        strength = info.data.get('inference_strength')
        return strength != 'explicit' if strength else True

    @field_validator('review_reason', mode='before')
    @classmethod
    def compute_review_reason(cls, v, info):
        """Auto-compute review_reason if needed."""
        if v:
            return v
        strength = info.data.get('inference_strength')
        needs_review = info.data.get('needs_review', True)

        if needs_review and strength:
            if strength == 'strong_suggestion':
                return "Inference based on clinical guidelines/knowledge graph (strong suggestion)"
            elif strength == 'weak':
                return "Inference is plausible but not guideline-justified (weak)"
        return ""


class PipelineOutput(BaseModel):
    """
    Complete output from a pipeline run.

    Attributes:
        note_id: ID of the processed note
        approach: Which approach was used
        ner_model: Which NER model was used
        embedding_model: Which embedding model was used
        llm_model: Which LLM (approach_4 only, else null)
        extractions: List of all extracted and coded conditions
        review_required_count: How many extractions need review
        processing_time_ms: Total processing time in milliseconds
        ner_result: Optional reference to the NER extraction result
        metadata: Optional additional metadata
    """
    note_id: str
    approach: Literal["approach_3", "approach_3_kg", "approach_4"]
    ner_model: str
    embedding_model: str
    llm_model: Optional[str] = None
    extractions: List[ExtractionResult] = Field(default_factory=list)
    review_required_count: int = 0
    processing_time_ms: float
    ner_result: Optional[NERResult] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('review_required_count', mode='before')
    @classmethod
    def compute_review_count(cls, v, info):
        """Auto-compute review count from extractions."""
        if v is not None and v != 0:
            return v
        extractions = info.data.get('extractions', [])
        return sum(1 for e in extractions if e.needs_review)

    def get_icd10_codes(self) -> List[str]:
        """Extract just the ICD-10 codes from extractions."""
        return [e.icd10_code.code for e in self.extractions]

    def get_hcc_codes(self) -> List[str]:
        """Extract HCC codes only."""
        return [e.icd10_code.code for e in self.extractions if e.icd10_code.hcc]

    def get_explicit_codes(self) -> List[str]:
        """Extract codes with explicit inference only."""
        return [e.icd10_code.code for e in self.extractions if e.inference_strength == 'explicit']

    def summary(self) -> str:
        """Generate a brief summary of the output."""
        codes = ", ".join(self.get_icd10_codes())
        review = f" ({self.review_required_count} need review)" if self.review_required_count > 0 else ""
        return f"{self.note_id} ({self.approach}): {len(self.extractions)} codes{review} - {codes}"


class ComparisonResult(BaseModel):
    """
    Results from comparing multiple approaches on the same note.

    Attributes:
        note_id: ID of the note being compared
        outputs: Dictionary mapping approach name to PipelineOutput
        metrics: Comparison metrics
    """
    note_id: str
    outputs: Dict[str, PipelineOutput]
    metrics: Dict[str, Any] = Field(default_factory=dict)

    def get_unique_codes_by_approach(self) -> Dict[str, List[str]]:
        """Get codes unique to each approach."""
        all_codes = {}
        for approach, output in self.outputs.items():
            all_codes[approach] = set(output.get_icd10_codes())

        unique = {}
        for approach in all_codes:
            others = set()
            for other_approach, codes in all_codes.items():
                if other_approach != approach:
                    others.update(codes)
            unique[approach] = list(all_codes[approach] - others)

        return unique

    def get_review_rates(self) -> Dict[str, float]:
        """Get needs-review rate per approach."""
        rates = {}
        for approach, output in self.outputs.items():
            if output.extractions:
                rate = output.review_required_count / len(output.extractions)
                rates[approach] = rate
            else:
                rates[approach] = 0.0
        return rates


class KnowledgeGraphEdge(BaseModel):
    """
    Represents an edge in the diabetes knowledge graph.

    Attributes:
        source: Source node ID
        target: Target node ID
        edge_type: Type of relationship
        inference_strength: Strength of the inference this edge supports
        description: Human-readable description
    """
    source: str
    target: str
    edge_type: str
    inference_strength: InferenceStrength
    description: Optional[str] = None


class KnowledgeGraphNode(BaseModel):
    """
    Represents a node in the diabetes knowledge graph.

    Attributes:
        node_id: Unique node identifier
        node_type: Type of node (condition, medication, lab_value, etc.)
        display: Display name
        snomed_cui: Optional SNOMED CUI
        snomed_code: Optional SNOMED code
        synonyms: Alternative names
    """
    node_id: str
    node_type: str
    display: str
    snomed_cui: Optional[str] = None
    snomed_code: Optional[str] = None
    synonyms: List[str] = Field(default_factory=list)


class BenchmarkResult(BaseModel):
    """
    Results from a single benchmark run.

    Attributes:
        note_id: Note being benchmarked
        approach: Approach used
        ner_model: NER model used
        embedding_model: Embedding model used
        llm_model: LLM model used (if applicable)
        output: Pipeline output
        metrics: Evaluation metrics
    """
    note_id: str
    approach: str
    ner_model: str
    embedding_model: str
    llm_model: Optional[str] = None
    output: PipelineOutput
    metrics: Dict[str, float] = Field(default_factory=dict)


class GroundTruthEntry(BaseModel):
    """
    Ground truth entry for evaluation.

    Attributes:
        level: sentence_level | context_aware
        condition: Expected condition description
        icd10_code: Expected ICD-10 code
        icd10_display: Expected code display
        inference_strength: Expected inference strength
        rationale: Why this code is expected
        codable: Whether this should produce a valid code
        hcc: Whether this is an HCC code
        evidence_sections: Sections containing evidence
        key_phrases: Key phrases that support this code
    """
    level: Literal["sentence_level", "context_aware"]
    condition: str
    icd10_code: Optional[str]
    icd10_display: Optional[str] = None
    inference_strength: InferenceStrength = "explicit"
    rationale: str
    codable: bool = True
    hcc: bool = False
    evidence_sections: List[str] = Field(default_factory=list)
    key_phrases: List[str] = Field(default_factory=list)
