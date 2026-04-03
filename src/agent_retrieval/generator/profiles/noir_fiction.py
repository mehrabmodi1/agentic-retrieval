from __future__ import annotations

from pathlib import Path

from agent_retrieval.generator.profiles.base import ContentProfile, GenerationContext
from agent_retrieval.schema.experiment import CorpusSpec


class NoirFictionProfile(ContentProfile):
    def generate_folder_structure(self, spec: CorpusSpec) -> list[Path]:
        raise NotImplementedError("Use v2 pool-based generation for noir_fiction")

    def generate_file_prompt(self, path: Path, context: GenerationContext) -> str:
        raise NotImplementedError("Use v2 pool-based generation for noir_fiction")

    def pool_generation_brief(self, target_token_count: int) -> str:
        return (
            "You are generating background files for a noir detective fiction corpus. "
            "Each file should be a Markdown (.md) file containing chapters, scenes, or "
            "supplementary materials for a hardboiled detective novel set in a 1940s American city.\n\n"
            "The story follows a private investigator working a complex case involving "
            "missing persons, corrupt officials, and underworld connections. The narrative "
            "should have a rich cast of characters including suspects, witnesses, informants, "
            "and law enforcement.\n\n"
            "Requirements:\n"
            "- Every file must have a .md extension\n"
            "- Chapter files: 500-1500 words of narrative prose with atmospheric descriptions, "
            "sharp dialogue, and plot progression\n"
            "- Case note files: 100-300 words of terse, factual investigation notes\n"
            "- Evidence log files: structured lists of evidence items with dates and descriptions\n"
            "- Witness statement files: 200-500 words in interview/transcript format\n"
            "- Write in classic noir style: first person or close third, cynical tone, "
            "vivid sensory details, morally ambiguous characters\n"
            "- Include realistic character names, locations, timestamps, and physical descriptions\n"
            "- Do NOT include any comments about the text being generated or synthetic\n"
            f"- Target total output: approximately {target_token_count:,} tokens across all files\n"
        )

    def skeleton(self, target_token_count: int) -> list[dict]:
        tokens_per_section = target_token_count // 8

        return [
            {
                "name": "chapters_1_5",
                "description": "Opening chapters: crime discovered, PI takes the case, initial interviews",
                "target_tokens": tokens_per_section,
                "files": [
                    "chapters/chapter_01.md", "chapters/chapter_02.md",
                    "chapters/chapter_03.md", "chapters/chapter_04.md",
                    "chapters/chapter_05.md",
                ],
            },
            {
                "name": "chapters_6_10",
                "description": "Middle chapters: investigation deepens, false leads, danger increases",
                "target_tokens": tokens_per_section,
                "files": [
                    "chapters/chapter_06.md", "chapters/chapter_07.md",
                    "chapters/chapter_08.md", "chapters/chapter_09.md",
                    "chapters/chapter_10.md",
                ],
            },
            {
                "name": "chapters_11_15",
                "description": "Later chapters: revelations, confrontations, stakes escalate",
                "target_tokens": tokens_per_section,
                "files": [
                    "chapters/chapter_11.md", "chapters/chapter_12.md",
                    "chapters/chapter_13.md", "chapters/chapter_14.md",
                    "chapters/chapter_15.md",
                ],
            },
            {
                "name": "chapters_16_20",
                "description": "Final chapters: climax, resolution, aftermath",
                "target_tokens": tokens_per_section,
                "files": [
                    "chapters/chapter_16.md", "chapters/chapter_17.md",
                    "chapters/chapter_18.md", "chapters/chapter_19.md",
                    "chapters/chapter_20.md",
                ],
            },
            {
                "name": "case_notes",
                "description": "PI's case notebook entries with dates, observations, and theories",
                "target_tokens": tokens_per_section,
                "files": [
                    "case_notes/day_01.md", "case_notes/day_02.md",
                    "case_notes/day_03.md", "case_notes/day_04.md",
                    "case_notes/day_05.md", "case_notes/day_06.md",
                    "case_notes/day_07.md", "case_notes/day_08.md",
                ],
            },
            {
                "name": "evidence",
                "description": "Evidence logs, photographs, forensic reports",
                "target_tokens": tokens_per_section,
                "files": [
                    "evidence/evidence_log.md", "evidence/forensic_report.md",
                    "evidence/phone_records.md", "evidence/financial_records.md",
                    "evidence/photographs.md", "evidence/autopsy_report.md",
                ],
            },
            {
                "name": "witness_statements",
                "description": "Formal and informal witness interviews and statements",
                "target_tokens": tokens_per_section,
                "files": [
                    "witnesses/bartender_statement.md", "witnesses/landlady_statement.md",
                    "witnesses/cab_driver_statement.md", "witnesses/secretary_statement.md",
                    "witnesses/informant_interview.md", "witnesses/detective_debrief.md",
                    "witnesses/neighbor_statement.md",
                ],
            },
            {
                "name": "locations",
                "description": "Location descriptions, maps, and scene reports",
                "target_tokens": tokens_per_section,
                "files": [
                    "locations/crime_scene_report.md", "locations/office_description.md",
                    "locations/warehouse_district.md", "locations/nightclub_report.md",
                    "locations/apartment_search.md", "locations/docks_surveillance.md",
                ],
            },
        ]
