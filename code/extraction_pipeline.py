"""
extraction_pipeline.py
=======================
Citation extraction pipeline used in:

  Murugasen, D. "Citation Hallucination in AI-Assisted Legal Research:
  A Cross-Model, Cross-Domain, Cross-Jurisdictional Empirical Study."
  Computer Law & Security Review (2025).

Requirements:
    pip install spacy pandas
    python -m spacy download en_core_web_sm

Usage:
    python extraction_pipeline.py --input your_model_outputs.csv
    python extraction_pipeline.py --input data/citations_raw.csv --demo
"""

import re
import argparse
import pandas as pd

# ── Regex patterns per jurisdiction ──────────────────────────────────────────

# United States — e.g. "Roe v. Wade, 410 U.S. 113 (1973)"
US_PATTERNS = [
    re.compile(
        r'([A-Z][A-Za-z\s\.\,\']+)\s+v\.?\s+([A-Z][A-Za-z\s\.\,\']+)'
        r',?\s+(\d{1,4})\s+([A-Z][A-Za-z\.\s]+)\s+(\d{1,4})'
        r'\s*\((\d{4})\)',
        re.IGNORECASE
    ),
    re.compile(
        r'([A-Z][A-Za-z\s\.\,\']+)\s+v\.?\s+([A-Z][A-Za-z\s\.\,\']+)'
        r',?\s+No\.?\s*([\d\-]+[A-Z\-]*)\s*\(([A-Za-z\.\s]+)\s+(\d{4})\)',
        re.IGNORECASE
    ),
]

# United Kingdom — e.g. "R v Smith [2023] UKSC 12"
UK_PATTERNS = [
    re.compile(
        r'([A-Z][A-Za-z\s\.\,\'&]+)\s+v\.?\s+([A-Z][A-Za-z\s\.\,\'&]+)'
        r'\s*\[(\d{4})\]\s+([A-Z]{2,8}(?:\s+\d+)?)\s+(\d+)',
        re.IGNORECASE
    ),
    re.compile(
        r'\[(\d{4})\]\s+([A-Z]{2,8}(?:\s+\d+)?)\s+(\d+)',
        re.IGNORECASE
    ),
]

# India — e.g. "State of Maharashtra v. Rajendra (2022) 4 SCC 102"
INDIA_PATTERNS = [
    re.compile(
        r'([A-Z][A-Za-z\s\.\,\']+)\s+v\.?\s+([A-Z][A-Za-z\s\.\,\']+)'
        r'\s*\((\d{4})\)\s+(\d+)\s+([A-Z]+(?:\s+[A-Za-z]+)?)\s+(\d+)',
        re.IGNORECASE
    ),
    re.compile(
        r'([A-Z][A-Za-z\s\.\,\']+)\s+v\.?\s+([A-Z][A-Za-z\s\.\,\']+)'
        r',?\s+AIR\s+(\d{4})\s+([A-Z]+)\s+(\d+)',
        re.IGNORECASE
    ),
    re.compile(
        r'Writ\s+Petition\s+(?:\(Civil\))?\s+No\.?\s*(\d+)\s+of\s+(\d{4})',
        re.IGNORECASE
    ),
]

ALL_PATTERNS = {
    'United States':  US_PATTERNS,
    'United Kingdom': UK_PATTERNS,
    'India':          INDIA_PATTERNS,
}


def extract_citations(text: str, jurisdiction: str) -> list[dict]:
    """
    Extract legal citations from a block of text for a given jurisdiction.

    Parameters
    ----------
    text : str
        Raw model output text
    jurisdiction : str
        One of 'United States', 'United Kingdom', 'India'

    Returns
    -------
    list of dicts with keys: raw_match, jurisdiction, pattern_index
    """
    patterns = ALL_PATTERNS.get(jurisdiction, [])
    found = []
    seen  = set()

    for i, pattern in enumerate(patterns):
        for match in pattern.finditer(text):
            raw = match.group(0).strip()
            if raw not in seen:
                seen.add(raw)
                found.append({
                    'raw_match':     raw,
                    'jurisdiction':  jurisdiction,
                    'pattern_index': i,
                })

    return found


def standardise_citation(raw: str, jurisdiction: str) -> dict:
    """
    Attempt to parse a raw citation string into structured fields.

    Returns dict with: case_name, court, year, jurisdiction, raw
    """
    result = {
        'raw':          raw,
        'case_name':    '',
        'court':        '',
        'year':         '',
        'jurisdiction': jurisdiction,
        'needs_review': False,
    }

    # Extract year (4-digit number between 1800 and 2026)
    year_match = re.search(r'\b(1[89]\d{2}|20[0-2]\d)\b', raw)
    if year_match:
        result['year'] = year_match.group(1)

    # Extract case name (everything before the first comma or bracket or year)
    name_match = re.match(
        r'^([A-Z][A-Za-z\s\.\,\'&]+\s+v\.?\s+[A-Z][A-Za-z\s\.\,\'&]+)',
        raw
    )
    if name_match:
        result['case_name'] = name_match.group(1).strip().rstrip(',')

    # Extract court abbreviation
    if jurisdiction == 'United States':
        court_match = re.search(
            r'\b(SCOTUS|S\.Ct\.|U\.S\.|F\.3d|F\.2d|F\.Supp|'
            r'F\.Supp\.2d|F\.Supp\.3d|[A-Z]{1,4}\. Cir\.)\b',
            raw
        )
        if court_match:
            result['court'] = court_match.group(1)

    elif jurisdiction == 'United Kingdom':
        court_match = re.search(
            r'\b(UKSC|UKHL|EWCA|EWHC|UKFTT|UKUT|UKPC|QB|Ch|Fam|AC)\b',
            raw
        )
        if court_match:
            result['court'] = court_match.group(1)

    elif jurisdiction == 'India':
        court_match = re.search(
            r'\b(SCC|AIR|SCR|Bom|Mad|Cal|Del|All|Ker|Guj|SCWR)\b',
            raw
        )
        if court_match:
            result['court'] = court_match.group(1)

    # Flag for manual review if key fields missing
    if not result['year'] or not result['case_name']:
        result['needs_review'] = True

    return result


def process_file(input_path: str, output_path: str = None, demo: bool = False):
    """
    Process a CSV of model outputs and extract citations.

    Expected input columns: question_id, model, jurisdiction, response_text
    """
    if demo:
        # Create a small demo dataframe
        demo_data = pd.DataFrame([
            {
                'question_id': 'Q001',
                'model': 'GPT-4',
                'jurisdiction': 'United States',
                'response_text': (
                    "The leading case is Terry v. Ohio, 392 U.S. 1 (1968), "
                    "in which the Supreme Court held that a police officer may "
                    "briefly detain a person based on reasonable suspicion. "
                    "See also United States v. Sokolow, 490 U.S. 1 (1989)."
                ),
            },
            {
                'question_id': 'Q035',
                'model': 'Gemini Pro',
                'jurisdiction': 'United Kingdom',
                'response_text': (
                    "The leading authority is R v Jogee [2016] UKSC 8, "
                    "in which the Supreme Court abolished the doctrine of "
                    "parasitic accessory liability. See also R v Ankar "
                    "[2022] EWCA Crim 115."
                ),
            },
            {
                'question_id': 'Q070',
                'model': 'Claude',
                'jurisdiction': 'India',
                'response_text': (
                    "The Supreme Court addressed anticipatory bail in "
                    "Gurbaksh Singh Sibbia v State of Punjab (1980) 2 SCC 565. "
                    "More recently see Sushila Aggarwal v State (NCT of Delhi) "
                    "(2020) 5 SCC 1."
                ),
            },
        ])
        df = demo_data
        print("Running in DEMO mode with 3 sample responses.\n")
    else:
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} model responses from {input_path}\n")

    results = []
    for _, row in df.iterrows():
        text    = str(row.get('response_text', ''))
        jur     = str(row.get('jurisdiction', 'United States'))
        model   = str(row.get('model', ''))
        q_id    = str(row.get('question_id', ''))

        raw_citations = extract_citations(text, jur)

        for rc in raw_citations:
            structured = standardise_citation(rc['raw_match'], jur)
            structured.update({
                'question_id': q_id,
                'model':       model,
            })
            results.append(structured)

    out_df = pd.DataFrame(results)

    if len(out_df) == 0:
        print("No citations extracted.")
        return

    print(f"Extracted {len(out_df)} citations total.")
    print(f"Flagged for manual review: {out_df['needs_review'].sum()}")
    print()
    print(out_df[['question_id', 'model', 'jurisdiction',
                  'case_name', 'court', 'year', 'needs_review']].to_string())

    if output_path:
        out_df.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract legal citations from AI model outputs'
    )
    parser.add_argument(
        '--input',
        default='data/citations_raw.csv',
        help='Path to CSV of model outputs'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Path to save extracted citations CSV (optional)'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run with built-in demo data'
    )
    args = parser.parse_args()
    process_file(args.input, args.output, args.demo)
