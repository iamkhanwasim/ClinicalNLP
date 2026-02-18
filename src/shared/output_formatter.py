"""
Output formatter for pipeline results.

Updated for new architecture with:
- Inference strength tracking ("explicit" | "strong_suggestion" | "weak")
- needs_review flags
- Color-coded console output with rich
- HCC marker display

Formats extraction results as:
- JSON (for API responses)
- Human-readable tables (for review and comparison)
- Markdown reports (for documentation)
- Rich console output with color coding
"""

import json
from typing import List, Dict, Optional
from tabulate import tabulate
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.style import Style
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from src.shared.models import (
    PipelineOutput,
    ExtractionResult,
    EvidenceSpan,
    InferenceStrength
)


class OutputFormatter:
    """
    Format pipeline outputs for different use cases.

    Supports:
    - JSON serialization for API responses
    - Tabular display for human review
    - Comparison tables for evaluating approaches
    - Markdown reports for documentation
    - Rich console output with color coding (if rich available)
    """

    def __init__(self, indent: int = 2, use_color: bool = True):
        """
        Initialize the formatter.

        Args:
            indent: JSON indentation level
            use_color: Enable colored console output (requires rich)
        """
        self.indent = indent
        self.use_color = use_color and RICH_AVAILABLE

        if self.use_color:
            self.console = Console(
                color_system="auto",
                force_terminal=True
            )
        else:
            self.console = None

    def to_json(self, output: PipelineOutput, pretty: bool = True) -> str:
        """
        Convert pipeline output to JSON string.

        Args:
            output: PipelineOutput object
            pretty: Whether to pretty-print with indentation

        Returns:
            JSON string
        """
        indent = self.indent if pretty else None
        return output.model_dump_json(indent=indent, exclude_none=True)

    def to_dict(self, output: PipelineOutput) -> Dict:
        """
        Convert pipeline output to dictionary.

        Args:
            output: PipelineOutput object

        Returns:
            Dictionary representation
        """
        return output.model_dump(exclude_none=True)

    def save_json(self, output: PipelineOutput, file_path: str | Path) -> None:
        """
        Save pipeline output to JSON file.

        Args:
            output: PipelineOutput object
            file_path: Path to save JSON file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json(output, pretty=True))

    def to_table(
        self,
        output: PipelineOutput,
        tablefmt: str = "grid",
        show_evidence: bool = True
    ) -> str:
        """
        Convert pipeline output to human-readable table.

        Args:
            output: PipelineOutput object
            tablefmt: Table format (grid, simple, pipe, etc.)
            show_evidence: Whether to include evidence column

        Returns:
            Formatted table string
        """
        if not output.extractions:
            return "No extractions found."

        # Build table data
        headers = ["Condition", "ICD-10", "Display", "Conf", "Infer", "Review", "HCC"]
        if show_evidence:
            headers.append("Evidence")

        rows = []
        for extraction in output.extractions:
            # Format inference strength
            inf_str = extraction.inference_strength[0].upper()  # E, S, W
            if extraction.inference_strength == "explicit":
                inf_display = f"[OK]{inf_str}"
            elif extraction.inference_strength == "strong_suggestion":
                inf_display = f"[!]{inf_str}"
            else:
                inf_display = f"[!!]{inf_str}"

            row = [
                extraction.condition,
                extraction.icd10_code.code,
                self._truncate(extraction.icd10_code.display, 40),
                f"{extraction.confidence:.2f}",
                inf_display,
                "[!]" if extraction.needs_review else "[OK]",
                "[HCC]" if extraction.icd10_code.hcc else ""
            ]

            if show_evidence:
                evidence_str = self._format_evidence(extraction.evidence_spans)
                row.append(evidence_str)

            rows.append(row)

        return tabulate(rows, headers=headers, tablefmt=tablefmt)

    def to_detailed_table(self, output: PipelineOutput) -> str:
        """
        Create detailed table with all extraction information.

        Args:
            output: PipelineOutput object

        Returns:
            Detailed table string
        """
        if not output.extractions:
            return "No extractions found."

        result = []
        result.append(f"{'='*80}")
        result.append(f"Pipeline Output: {output.note_id}")
        result.append(f"Approach: {output.approach}")
        result.append(f"NER Model: {output.ner_model} | Embedding: {output.embedding_model}")
        if output.llm_model:
            result.append(f"LLM Model: {output.llm_model}")
        result.append(f"Processing Time: {output.processing_time_ms:.2f} ms")
        result.append(f"Total Extractions: {len(output.extractions)}")
        result.append(f"Review Required: {output.review_required_count}")
        result.append(f"{'='*80}\n")

        for i, extraction in enumerate(output.extractions, 1):
            # Inference indicator
            if extraction.inference_strength == "explicit":
                inf_indicator = "[OK]"
            elif extraction.inference_strength == "strong_suggestion":
                inf_indicator = "[!]"
            else:
                inf_indicator = "[!!]"

            result.append(f"{i}. {inf_indicator} {extraction.condition}")
            result.append(f"   ICD-10: {extraction.icd10_code.code} - {extraction.icd10_code.display}")
            result.append(f"   SNOMED: {extraction.snomed_concept.snomed_code} - {extraction.snomed_concept.display}")
            result.append(f"   Confidence: {extraction.confidence:.3f}")
            result.append(f"   Inference Strength: {extraction.inference_strength}")
            result.append(f"   HCC: {'Yes' if extraction.icd10_code.hcc else 'No'}")

            if extraction.needs_review:
                result.append(f"   [!] REVIEW REQUIRED: {extraction.review_reason}")

            if extraction.enrichment_reasoning:
                result.append(f"   Reasoning: {extraction.enrichment_reasoning}")

            if extraction.evidence_spans:
                result.append(f"   Evidence:")
                for j, evidence in enumerate(extraction.evidence_spans, 1):
                    evidence_text = self._truncate(evidence.text, 60)
                    result.append(f"      {j}) \"{evidence_text}\" ({evidence.section})")

            result.append("")  # Blank line between extractions

        return "\n".join(result)

    def create_comparison_table(
        self,
        outputs: List[PipelineOutput],
        tablefmt: str = "grid"
    ) -> str:
        """
        Create side-by-side comparison table for multiple pipeline runs.

        Args:
            outputs: List of PipelineOutput objects
            tablefmt: Table format

        Returns:
            Comparison table string
        """
        if not outputs:
            return "No outputs to compare."

        result = []
        result.append(f"\nComparison for Note: {outputs[0].note_id}")
        result.append("=" * 100)

        # Get all unique ICD-10 codes across approaches
        all_codes = set()
        for output in outputs:
            all_codes.update(output.get_icd10_codes())

        # Build comparison matrix
        approach_names = [output.approach for output in outputs]
        headers = ["ICD-10 Code", "Display"] + approach_names

        rows = []
        for code in sorted(all_codes):
            # Find the display name (use first occurrence)
            display = ""
            for output in outputs:
                for extraction in output.extractions:
                    if extraction.icd10_code.code == code:
                        display = self._truncate(extraction.icd10_code.display, 40)
                        break
                if display:
                    break

            row = [code, display]

            # Check which approaches found this code
            for output in outputs:
                found = any(e.icd10_code.code == code for e in output.extractions)
                row.append("[OK]" if found else "")

            rows.append(row)

        result.append(tabulate(rows, headers=headers, tablefmt=tablefmt))

        # Add statistics
        result.append("\nStatistics:")
        for output in outputs:
            hcc_count = len(output.get_hcc_codes())
            explicit_count = sum(1 for e in output.extractions if e.inference_strength == "explicit")
            result.append(
                f"  {output.approach}: {len(output.extractions)} codes "
                f"({hcc_count} HCC, {explicit_count} explicit, {output.review_required_count} need review)"
            )

        return "\n".join(result)

    def to_markdown_report(
        self,
        outputs: List[PipelineOutput],
        title: str = "Clinical NLP Extraction Report"
    ) -> str:
        """
        Generate markdown report for multiple pipeline outputs.

        Args:
            outputs: List of PipelineOutput objects
            title: Report title

        Returns:
            Markdown formatted report
        """
        md = []
        md.append(f"# {title}\n")

        for output in outputs:
            md.append(f"## Note: {output.note_id}\n")
            md.append(f"**Approach:** {output.approach}  ")
            md.append(f"**NER Model:** {output.ner_model}  ")
            md.append(f"**Embedding Model:** {output.embedding_model}  ")
            if output.llm_model:
                md.append(f"**LLM Model:** {output.llm_model}  ")
            md.append(f"**Processing Time:** {output.processing_time_ms:.2f} ms  ")
            md.append(f"**Total Codes:** {len(output.extractions)}  ")
            md.append(f"**Review Required:** {output.review_required_count}\n")

            if not output.extractions:
                md.append("*No extractions found.*\n")
                continue

            # Table of extractions with inference strength
            headers = ["Condition", "ICD-10", "Display", "Conf", "Inference", "Review", "HCC"]
            rows = []

            for extraction in output.extractions:
                # Inference strength indicator
                if extraction.inference_strength == "explicit":
                    inf_display = "[OK]E"
                elif extraction.inference_strength == "strong_suggestion":
                    inf_display = "[!]S"
                else:
                    inf_display = "[!!]W"

                rows.append([
                    extraction.condition,
                    extraction.icd10_code.code,
                    self._truncate(extraction.icd10_code.display, 30),
                    f"{extraction.confidence:.2f}",
                    inf_display,
                    "[!]" if extraction.needs_review else "[OK]",
                    "[HCC]" if extraction.icd10_code.hcc else ""
                ])

            md.append(tabulate(rows, headers=headers, tablefmt="pipe"))
            md.append("")

            # Evidence details
            md.append("### Evidence Details\n")
            for i, extraction in enumerate(output.extractions, 1):
                md.append(f"**{i}. {extraction.condition}** ({extraction.icd10_code.code})")
                md.append(f"- Inference: {extraction.inference_strength}")
                if extraction.needs_review:
                    md.append(f"- [!] Review Required: {extraction.review_reason}")

                if extraction.evidence_spans:
                    md.append("- Evidence:")
                    for evidence in extraction.evidence_spans:
                        md.append(f"  - \"{self._truncate(evidence.text, 80)}\" `({evidence.section})`")
                md.append("")

        return "\n".join(md)

    def save_comparison_report(
        self,
        outputs: List[PipelineOutput],
        file_path: str | Path,
        format: str = "txt"
    ) -> None:
        """
        Save comparison report to file.

        Args:
            outputs: List of PipelineOutput objects
            file_path: Path to save report
            format: Output format (txt, md, json)
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(
                    [output.model_dump() for output in outputs],
                    f,
                    indent=2,
                    default=str
                )

        elif format == "md":
            md_report = self.to_markdown_report(outputs)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(md_report)

        else:  # txt
            table = self.create_comparison_table(outputs)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(table)

    def display_pipeline_output_rich(
        self,
        output: PipelineOutput,
        show_evidence: bool = False
    ):
        """
        Display pipeline output to console with rich formatting (color-coded).

        Requires rich library. Falls back to plain output if not available.

        Args:
            output: PipelineOutput from pipeline run
            show_evidence: Show evidence spans for each extraction
        """
        if not self.console:
            # Fallback to plain output
            self.print_output(output, detailed=show_evidence)
            return

        # Header panel
        self.console.print()
        self.console.print(
            Panel(
                f"[bold]Pipeline Results: {output.approach.upper()}[/bold]\n"
                f"Note ID: {output.note_id}\n"
                f"NER Model: {output.ner_model} | Embedding: {output.embedding_model}"
                + (f" | LLM: {output.llm_model}" if output.llm_model else ""),
                style="bold blue"
            )
        )

        # Summary stats
        self.console.print(f"\n[bold]Summary:[/bold]")
        self.console.print(f"  Total extractions: {len(output.extractions)}")
        self.console.print(f"  Review required: {output.review_required_count}")
        self.console.print(f"  Processing time: {output.processing_time_ms:.1f}ms")

        # Breakdown by inference strength
        explicit_count = sum(1 for e in output.extractions if e.inference_strength == "explicit")
        strong_count = sum(1 for e in output.extractions if e.inference_strength == "strong_suggestion")
        weak_count = sum(1 for e in output.extractions if e.inference_strength == "weak")

        self.console.print(f"\n[bold]Inference Breakdown:[/bold]")
        self.console.print(f"  [green]Explicit: {explicit_count}[/green]")
        self.console.print(f"  [yellow][!] Strong suggestion: {strong_count}[/yellow]")
        self.console.print(f"  [red][!!] Weak: {weak_count}[/red]")

        # Individual extractions
        if output.extractions:
            self.console.print(f"\n[bold]Extractions:[/bold]")
            for i, extraction in enumerate(output.extractions, 1):
                self.console.print(f"\n[bold cyan]{i}.[/bold cyan]")
                self._display_extraction_rich(extraction, show_evidence)

        # Review required section
        if output.review_required_count > 0:
            self.console.print()
            self.console.print(
                Panel(
                    f"[bold red][!] {output.review_required_count} extraction(s) require review[/bold red]\n"
                    "Please manually verify non-explicit inferences.",
                    style="red"
                )
            )

        self.console.print()

    def _display_extraction_rich(
        self,
        extraction: ExtractionResult,
        show_evidence: bool = False
    ):
        """Display a single extraction with rich formatting."""
        # Determine style based on inference strength
        if extraction.inference_strength == "explicit":
            strength_style = Style(color="green", bold=True)
            strength_icon = "[OK]"
        elif extraction.inference_strength == "strong_suggestion":
            strength_style = Style(color="yellow", bold=True)
            strength_icon = "[!]"
        else:  # weak
            strength_style = Style(color="red", bold=True)
            strength_icon = "[!!]"

        # Condition
        self.console.print(f"  Condition: [bold]{extraction.condition}[/bold]")

        # ICD-10 Code with HCC marker
        icd10_display = f"{extraction.icd10_code.code} | {extraction.icd10_code.display}"
        if extraction.icd10_code.hcc:
            icd10_display += " [HCC]"
        self.console.print(f"  ICD-10: {icd10_display}")

        # SNOMED Code
        snomed_display = f"{extraction.snomed_concept.snomed_code} | {extraction.snomed_concept.display}"
        self.console.print(f"  SNOMED: {snomed_display}")

        # Confidence
        self.console.print(f"  Confidence: [bold]{extraction.confidence:.3f}[/bold]")

        # Inference Strength with icon
        strength_text = Text(
            f"{strength_icon} {extraction.inference_strength.upper()}",
            style=strength_style
        )
        self.console.print(f"  Inference: ", end="")
        self.console.print(strength_text)

        # Review Status
        if extraction.needs_review:
            self.console.print(f"  Review: [bold red][!] REQUIRED[/bold red]")
            self.console.print(f"  Reason: {extraction.review_reason}")

        # Evidence (if requested and available)
        if show_evidence and extraction.evidence_spans:
            self.console.print(f"  Evidence:")
            for j, span in enumerate(extraction.evidence_spans[:3], 1):
                evidence_text = self._truncate(span.text, 60)
                self.console.print(f"    {j}. [{span.section}] {evidence_text}...")

    def _format_evidence(self, evidence_spans: List[EvidenceSpan], max_length: int = 100) -> str:
        """
        Format evidence spans into a compact string.

        Args:
            evidence_spans: List of EvidenceSpan objects
            max_length: Maximum total length

        Returns:
            Formatted evidence string
        """
        if not evidence_spans:
            return ""

        parts = []
        for span in evidence_spans[:2]:  # Show first 2 evidence spans
            text = self._truncate(span.text, 40)
            parts.append(f"\"{text}\" ({span.section})")

        if len(evidence_spans) > 2:
            parts.append(f"(+{len(evidence_spans) - 2} more)")

        result = "; ".join(parts)
        return self._truncate(result, max_length)

    def _truncate(self, text: str, max_length: int) -> str:
        """
        Truncate text to maximum length with ellipsis.

        Args:
            text: Text to truncate
            max_length: Maximum length

        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."

    def print_output(self, output: PipelineOutput, detailed: bool = False, use_rich: bool = True) -> None:
        """
        Print pipeline output to console.

        Args:
            output: PipelineOutput object
            detailed: Whether to show detailed view
            use_rich: Use rich formatting if available
        """
        # Try rich output first if requested and available
        if use_rich and self.console:
            self.display_pipeline_output_rich(output, show_evidence=detailed)
            return

        # Fallback to plain output
        if detailed:
            print(self.to_detailed_table(output))
        else:
            print(f"\nNote: {output.note_id} | Approach: {output.approach}")
            print(f"{'='*80}")
            print(self.to_table(output, tablefmt="simple"))
            print(f"\nProcessing Time: {output.processing_time_ms:.2f} ms")
            print(f"Total Codes: {len(output.extractions)} ({len(output.get_hcc_codes())} HCC)")
            print(f"Review Required: {output.review_required_count}\n")


# Convenience functions
def format_as_json(output: PipelineOutput, pretty: bool = True) -> str:
    """Quick JSON formatting."""
    formatter = OutputFormatter()
    return formatter.to_json(output, pretty=pretty)


def format_as_table(output: PipelineOutput) -> str:
    """Quick table formatting."""
    formatter = OutputFormatter()
    return formatter.to_table(output)


def print_pipeline_output(output: PipelineOutput, detailed: bool = False) -> None:
    """Quick console printing."""
    formatter = OutputFormatter()
    formatter.print_output(output, detailed=detailed)


if __name__ == "__main__":
    # Test the formatter with mock data
    from src.shared.models import (
        EntitySpan,
        EvidenceSpan,
        ExtractionResult,
        PipelineOutput,
        SNOMEDConcept,
        ICDCode
    )

    # Create mock extractions
    extraction1 = ExtractionResult(
        condition="Type 2 diabetes mellitus with hyperglycemia",
        snomed_concept=SNOMEDConcept(
            cui="C0011860",
            snomed_code="44054006",
            display="Type 2 diabetes mellitus",
            synonyms=["NIDDM", "Adult-onset diabetes"]
        ),
        icd10_code=ICDCode(
            code="E11.65",
            display="Type 2 diabetes mellitus with hyperglycemia",
            billable=True,
            hcc=True
        ),
        confidence=0.95,
        inference_strength="explicit",
        needs_review=False,
        evidence_spans=[
            EvidenceSpan(
                text="blood sugar levels have been suboptimal, with a recent reading of 425",
                section="HPI",
                char_start=120,
                char_end=190
            )
        ]
    )

    extraction2 = ExtractionResult(
        condition="Diabetic retinopathy",
        snomed_concept=SNOMEDConcept(
            cui="C0011884",
            snomed_code="4855003",
            display="Diabetic retinopathy",
            synonyms=[]
        ),
        icd10_code=ICDCode(
            code="E11.319",
            display="Type 2 diabetes mellitus with unspecified diabetic retinopathy",
            billable=True,
            hcc=False
        ),
        confidence=0.75,
        inference_strength="strong_suggestion",
        needs_review=True,
        review_reason="Inference based on clinical guidelines (strong suggestion)"
    )

    # Create mock output
    output = PipelineOutput(
        note_id="note_001",
        approach="approach_3",
        ner_model="stanza_clinical",
        embedding_model="sapbert",
        extractions=[extraction1, extraction2],
        processing_time_ms=340.5
    )

    # Test formatters
    formatter = OutputFormatter()

    print("=== JSON Format ===")
    print(formatter.to_json(output))

    print("\n=== Table Format ===")
    print(formatter.to_table(output))

    print("\n=== Detailed Format ===")
    print(formatter.to_detailed_table(output))

    print("\n=== Rich Console Format ===")
    formatter.print_output(output, detailed=True, use_rich=True)
