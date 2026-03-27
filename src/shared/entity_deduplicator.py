"""
Entity deduplication to merge duplicate NER entities.

Removes duplicates based on:
- Lowercase entity text
- Entity type

When duplicates are found:
- Keep the entity with highest confidence
- If confidence is equal, keep the first occurrence (earliest char_start)

This filter should be applied AFTER entity filtering and BEFORE semantic resolution.
"""

from typing import List, Tuple
from src.shared.models import EntitySpan


class EntityDeduplicator:
    """Deduplicate NER entities that refer to the same concept."""

    def deduplicate(self, entities: List[EntitySpan]) -> List[EntitySpan]:
        """
        Merge duplicate entities. Keep the best occurrence based on confidence or position.

        Dedup key: lowercase entity text + entity type.
        Two entities with the same text but different types are kept separate
        (e.g., "insulin" as medication vs "insulin" as lab_value).

        Args:
            entities: List of EntitySpan objects from NER extraction (after filtering)

        Returns:
            List of EntitySpan objects with duplicates removed
        """
        seen = {}  # key -> EntitySpan (kept entity)

        for entity in entities:
            key = self._make_dedup_key(entity)

            if key not in seen:
                # First occurrence of this entity
                seen[key] = entity
            else:
                # Duplicate found - decide which one to keep
                existing = seen[key]
                if self._should_replace(existing, entity):
                    seen[key] = entity

        return list(seen.values())

    def _make_dedup_key(self, entity: EntitySpan) -> Tuple[str, str]:
        """
        Create a deduplication key for an entity.

        Args:
            entity: EntitySpan to create key for

        Returns:
            Tuple of (normalized_text, entity_type)
        """
        normalized_text = entity.text.lower().strip()
        return (normalized_text, entity.entity_type)

    def _should_replace(self, existing: EntitySpan, new: EntitySpan) -> bool:
        """
        Decide whether to replace the existing entity with the new one.

        Priority:
        1. Higher confidence wins
        2. If confidence is equal or both None, earlier position wins

        Args:
            existing: Currently kept EntitySpan
            new: New EntitySpan candidate

        Returns:
            True if new entity should replace existing, False otherwise
        """
        # If both have confidence scores, use the higher one
        if existing.confidence is not None and new.confidence is not None:
            if new.confidence > existing.confidence:
                return True
            elif new.confidence < existing.confidence:
                return False
            # If equal confidence, fall through to position check

        # If only one has a confidence score, prefer the one with confidence
        if existing.confidence is None and new.confidence is not None:
            return True
        if existing.confidence is not None and new.confidence is None:
            return False

        # If confidence is equal or both None, keep the earlier occurrence
        # (existing is already in the dict, so don't replace)
        return False
