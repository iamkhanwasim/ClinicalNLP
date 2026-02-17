"""
Clinical note parser that extracts structured sections from unstructured text.

The parser handles markdown-style headers (# and ##) to identify sections
and maintains character offsets for evidence span tracking.
"""

import re
from pathlib import Path
from typing import Dict, Tuple, List
from src.shared.models import ClinicalNote


class NoteParser:
    """
    Parses clinical notes with markdown-style section headers.

    Sections are identified by:
    - Level 1 headers: # Section Name
    - Level 2 headers: ## Subsection Name

    Character offsets are preserved for evidence span tracking.
    """

    def __init__(self):
        # Pattern to match markdown headers
        self.header_pattern = re.compile(r'^(#{1,2})\s+(.+)$', re.MULTILINE)

    def parse_file(self, file_path: str | Path) -> ClinicalNote:
        """
        Parse a clinical note from a text file.

        Args:
            file_path: Path to the clinical note file

        Returns:
            ClinicalNote object with parsed sections

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is empty
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Clinical note not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        if not raw_text.strip():
            raise ValueError(f"Clinical note is empty: {file_path}")

        # Extract note_id from filename (remove .txt extension)
        note_id = file_path.stem

        return self.parse_text(raw_text, note_id)

    def parse_text(self, raw_text: str, note_id: str = "unknown") -> ClinicalNote:
        """
        Parse a clinical note from raw text.

        Args:
            raw_text: The full text of the clinical note
            note_id: Identifier for this note

        Returns:
            ClinicalNote object with parsed sections
        """
        sections = self._extract_sections(raw_text)

        return ClinicalNote(
            note_id=note_id,
            raw_text=raw_text,
            sections=sections
        )

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract sections from text based on markdown headers.

        Sections are identified by # and ## headers. Subsections (##) are
        included in their parent section with a hierarchical key.

        Args:
            text: Full clinical note text

        Returns:
            Dictionary mapping section names to section content
        """
        sections = {}

        # Find all headers with their positions
        headers = list(self.header_pattern.finditer(text))

        if not headers:
            # No headers found - treat entire text as a single section
            sections["full_text"] = text.strip()
            return sections

        # Track the current parent section for subsections
        current_parent = None

        for i, match in enumerate(headers):
            level = len(match.group(1))  # Number of # characters
            section_name = match.group(2).strip()
            start_pos = match.end()  # Start after the header line

            # Determine end position (start of next header or end of text)
            if i + 1 < len(headers):
                end_pos = headers[i + 1].start()
            else:
                end_pos = len(text)

            # Extract section content
            section_content = text[start_pos:end_pos].strip()

            if level == 1:
                # Level 1 header - new main section
                current_parent = section_name
                sections[section_name] = section_content
            elif level == 2 and current_parent:
                # Level 2 header - subsection of current parent
                # Use hierarchical key: Parent_Subsection
                hierarchical_key = f"{current_parent}_{section_name}"
                sections[hierarchical_key] = section_content
            else:
                # Fallback for edge cases
                sections[section_name] = section_content

        return sections

    def get_section_offsets(self, note: ClinicalNote) -> Dict[str, Tuple[int, int]]:
        """
        Get character offsets for each section in the original text.

        Args:
            note: Parsed ClinicalNote object

        Returns:
            Dictionary mapping section names to (start, end) character positions
        """
        offsets = {}
        raw_text = note.raw_text

        for section_name, section_content in note.sections.items():
            # Find the section content in the raw text
            # Use a simple search - this assumes section content is unique
            start = raw_text.find(section_content)
            if start != -1:
                end = start + len(section_content)
                offsets[section_name] = (start, end)

        return offsets

    def get_section_for_offset(
        self,
        note: ClinicalNote,
        char_offset: int
    ) -> str | None:
        """
        Determine which section contains a given character offset.

        Args:
            note: Parsed ClinicalNote object
            char_offset: Character position in the raw text

        Returns:
            Section name containing the offset, or None if not found
        """
        offsets = self.get_section_offsets(note)

        for section_name, (start, end) in offsets.items():
            if start <= char_offset < end:
                return section_name

        return None

    def extract_context_around_offset(
        self,
        text: str,
        offset: int,
        context_chars: int = 100
    ) -> str:
        """
        Extract text context around a character offset.

        Useful for debugging or showing evidence spans in context.

        Args:
            text: Full text
            offset: Character position
            context_chars: Number of characters before/after to include

        Returns:
            Text context around the offset
        """
        start = max(0, offset - context_chars)
        end = min(len(text), offset + context_chars)

        context = text[start:end]

        # Add ellipsis if truncated
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."

        return context

    def get_section_statistics(self, note: ClinicalNote) -> Dict[str, any]:
        """
        Get statistics about the parsed note sections.

        Args:
            note: Parsed ClinicalNote object

        Returns:
            Dictionary with section statistics
        """
        stats = {
            "note_id": note.note_id,
            "total_length": len(note.raw_text),
            "num_sections": len(note.sections),
            "sections": {}
        }

        for section_name, section_content in note.sections.items():
            stats["sections"][section_name] = {
                "length": len(section_content),
                "word_count": len(section_content.split()),
                "line_count": len(section_content.split('\n'))
            }

        return stats


def normalize_section_name(section_name: str) -> str:
    """
    Normalize section names for consistent matching.

    Converts to lowercase, removes special characters, and standardizes
    common abbreviations.

    Args:
        section_name: Raw section name

    Returns:
        Normalized section name
    """
    # Convert to lowercase and remove special characters
    normalized = re.sub(r'[^\w\s]', '', section_name.lower())

    # Replace multiple spaces with single space
    normalized = re.sub(r'\s+', '_', normalized)

    # Common section name mappings
    mappings = {
        "hpi": "history_of_present_illness",
        "pmh": "past_history_medical",
        "psh": "past_history_surgical",
        "ros": "review_of_systems",
        "pe": "physical_exam",
        "a_p": "assessment_plan",
        "ap": "assessment_plan",
    }

    return mappings.get(normalized, normalized)


# Convenience function for quick parsing
def parse_clinical_note(file_path: str | Path) -> ClinicalNote:
    """
    Quick helper to parse a clinical note from a file.

    Args:
        file_path: Path to the clinical note file

    Returns:
        Parsed ClinicalNote object
    """
    parser = NoteParser()
    return parser.parse_file(file_path)


if __name__ == "__main__":
    # Test the parser with sample notes
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from config.settings import CLINICAL_NOTES_DIR

    # Parse all clinical notes
    for note_file in CLINICAL_NOTES_DIR.glob("*.txt"):
        print(f"\n{'='*60}")
        print(f"Parsing: {note_file.name}")
        print('='*60)

        parser = NoteParser()
        note = parser.parse_file(note_file)

        print(f"Note ID: {note.note_id}")
        print(f"Total length: {len(note.raw_text)} characters")
        print(f"Sections found: {len(note.sections)}")
        print("\nSections:")
        for section_name in note.sections:
            section_length = len(note.sections[section_name])
            print(f"  - {section_name}: {section_length} chars")

        # Show statistics
        stats = parser.get_section_statistics(note)
        print(f"\nStatistics:")
        for section_name, section_stats in stats["sections"].items():
            print(f"  {section_name}:")
            print(f"    Words: {section_stats['word_count']}")
            print(f"    Lines: {section_stats['line_count']}")
