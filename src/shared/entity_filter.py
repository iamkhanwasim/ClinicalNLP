"""
Post-NER entity filtering to remove non-clinical entities.

Removes:
- Section headers extracted as entities
- Non-clinical terms (financial constraints, medication costs, etc.)
- Subword fragments from BioBERT NER (##yp, ##lled, ##tension)
- Single-character and very short entities
- Very long spans that are likely parsing errors
- Generic phrases and pronouns

This filter should be applied AFTER NER extraction and BEFORE semantic resolution.
"""

from typing import List
from src.shared.models import EntitySpan


class EntityFilter:
    """Post-NER filter to remove non-clinical entities before SNOMED resolution."""

    # Section headers that NER models sometimes extract as entities
    SECTION_HEADERS = {
        "past_history_medical", "past_medical_history", "history_of_present_illness",
        "review_of_systems", "assessment_plan", "assessment_and_plan",
        "assessment", "plan", "hpi", "pmh", "ros", "medications",
        "allergies", "social_history", "family_history", "vital_signs",
        "physical_exam", "laboratory", "imaging", "procedures",
    }

    # Generic phrases that are not codable clinical entities
    NON_CLINICAL_TERMS = {
        "these symptoms", "the symptoms", "his symptoms", "her symptoms",
        "the patient", "the condition", "this condition",
        "current therapy", "current regimen", "current treatment",
        "current medications", "medication adherence", "medication compliance",
        "medication costs", "medication access", "insulin access",
        "patient assistance programs", "financial constraints",
        "follow up", "follow-up", "return visit",
        "blood work", "lab work", "test results",
    }

    # Minimum entity text length (skip single characters and very short fragments)
    MIN_LENGTH = 3

    # Maximum entity text length (skip very long extracted spans that are likely parsing errors)
    MAX_LENGTH = 80

    def filter_entities(self, entities: List[EntitySpan]) -> List[EntitySpan]:
        """
        Filter out non-clinical entities. Returns filtered list.

        Args:
            entities: List of EntitySpan objects from NER extraction

        Returns:
            Filtered list of EntitySpan objects with garbage entities removed
        """
        filtered = []
        for entity in entities:
            if self._should_keep(entity):
                filtered.append(entity)
        return filtered

    def _should_keep(self, entity: EntitySpan) -> bool:
        """
        Determine if an entity should be kept or filtered out.

        Args:
            entity: EntitySpan to evaluate

        Returns:
            True if entity should be kept, False if it should be filtered out
        """
        text = entity.text.strip()
        text_lower = text.lower().replace("_", " ").strip()

        # Too short or too long
        if len(text) < self.MIN_LENGTH or len(text) > self.MAX_LENGTH:
            return False

        # Section header (check both space and underscore versions)
        if text_lower.replace(" ", "_") in self.SECTION_HEADERS:
            return False
        if text_lower.replace("_", " ") in self.SECTION_HEADERS:
            return False

        # Non-clinical term
        if text_lower in self.NON_CLINICAL_TERMS:
            return False

        # Subword fragments (BioBERT tokenizer artifacts)
        if text.startswith("##"):
            return False

        # Pure numbers without clinical context (e.g., "425" extracted alone)
        # Note: Allow decimal numbers (lab values like "9.2")
        if text.replace(".", "").replace(",", "").isdigit():
            return False

        # Single common words that are not clinical entities
        if len(text_lower.split()) == 1 and text_lower in {
            "presents", "stable", "elevated", "poor", "follow",
            "agents", "counseling", "hp", "un", "li", "h",
        }:
            return False

        return True
