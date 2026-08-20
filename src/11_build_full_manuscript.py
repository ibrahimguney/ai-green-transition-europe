"""
Step 07B - Assemble the full manuscript from modular Markdown sections.

The script keeps analysis-generated Results separate from narrative drafts.
It expects manuscript/results_draft.md to have been generated locally by
src/10_synthesize_results.py.

Output:
    manuscript/manuscript_full.md

Run:
    python src/11_build_full_manuscript.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"

SECTION_FILES = [
    "front_matter.md",
    "introduction_draft.md",
    "literature_review_draft.md",
    "methods_draft.md",
    "results_draft.md",
    "discussion_draft.md",
    "conclusion_draft.md",
]


def main():
    missing = [name for name in SECTION_FILES if not (MANUSCRIPT / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing manuscript section(s): " + ", ".join(missing) + "\n"
            "If results_draft.md is missing, run: python src\\10_synthesize_results.py"
        )

    chunks = []
    for name in SECTION_FILES:
        text = (MANUSCRIPT / name).read_text(encoding="utf-8").strip()
        chunks.append(text)

    appendix = """
# Data and Code Availability

All analysis scripts used to download, harmonise, model, and synthesise the data are version-controlled in the public project repository. Raw and derived data are obtained from Eurostat APIs and can be regenerated from the scripts subject to source availability and revisions.

# Declarations

**Conflict of interest:** To be completed before submission.  
**Funding:** To be completed before submission.  
**Author contributions:** To be completed before submission.  
**AI-use disclosure:** To be adapted to the target journal's current policy before submission.
""".strip()

    full_text = "\n\n".join(chunks + [appendix]) + "\n"
    output = MANUSCRIPT / "manuscript_full.md"
    output.write_text(full_text, encoding="utf-8")

    words = len(full_text.split())
    print("STEP 07B - FULL MANUSCRIPT ASSEMBLY")
    print("=" * 46)
    print(f"Sections assembled: {len(SECTION_FILES)}")
    print(f"Approximate words: {words:,}")
    print(f"Saved: {output}")
    print("\nNext: review title/abstract, Results wording, citations, and journal-specific formatting.")


if __name__ == "__main__":
    main()
