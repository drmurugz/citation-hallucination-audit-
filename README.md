# Citation Hallucination in AI-Assisted Legal Research
### Replication Data and Code Repository

This repository contains the dataset, classification data, extraction 
pipeline, and statistical analysis code for the study:

> Murugasen, D. *Citation Hallucination in AI-Assisted Legal Research: 
> A Cross-Model, Cross-Domain, Cross-Jurisdictional Empirical Study.*
> Computer Law & Security Review (2025). 

---

## Repository Structure

```
citation-hallucination-audit/
├── data/
│   ├── questions.csv              # 500 legal questions with domain/jurisdiction labels
│   ├── citations_raw.csv          # All 2,847 extracted citations across 3 models
│   └── classifications.csv        # Three-tier classification for every citation
├── supplementary/
│   ├── Table_S1_full_cells.csv    # 15-cell domain × jurisdiction fabrication breakdown
│   └── codebook.pdf               # Full classification scheme definitions and examples
├── code/
│   ├── extraction_pipeline.py     # spaCy NER + Regex citation extraction code
│   └── statistical_analysis.py    # Chi-square, Fisher's exact, Cramér's V analysis
├── figures/
│   ├── fig2_heatmap.pdf           # Figure 2 — Stratified question distribution
│   └── fig3_verification.pdf      # Figure 3 — Verification database architecture
├── LICENSE                        # CC BY 4.0
└── README.md                      # This file
```

---

## Study Overview

**Research Questions:**
- RQ1: Do GPT-4, Gemini Pro, and Claude differ in hallucination rates?
- RQ2: Does fabrication rate vary across legal domains?
- RQ3: Does fabrication rate vary across jurisdictions?
- RQ4: What types of fabrication errors do models produce?
- RQ5: What are the professional responsibility implications?

**Dataset:** 500 legal questions × 3 models = 2,847 extracted citations

**Domains:** Criminal Law, Contract Law, Intellectual Property, 
Constitutional Law, Data Protection

**Jurisdictions:** United States, United Kingdom, India

**Models Tested:**
- GPT-4-turbo-2024-04-09 (OpenAI)
- Gemini-1.0-pro-001 (Google DeepMind)  
- Claude-3-opus-20240229 (Anthropic)

**Query period:** January–February 2025  
**Temperature:** 0.0 (all models)

---

## Key Findings

| Model | Total Citations | Accurate (%) | Misrepresented (%) | Fabricated (%) |
|---|---|---|---|---|
| GPT-4 | 1,024 | 44.2 | 14.5 | 41.3 |
| Gemini Pro | 872 | 37.6 | 12.4 | 50.0 |
| Claude | 951 | 48.6 | 11.4 | 40.1 |
| **Total** | **2,847** | **43.7** | **12.8** | **43.5** |

---

## Verification Databases

- **United States:** [CourtListener](https://www.courtlistener.com) (REST API)
- **United Kingdom:** [BAILII](https://www.bailii.org) (Web Query)
- **India:** [Indian Kanoon](https://indiankanoon.org) (REST API)

---

## Fabrication Typology

| Type | n | % |
|---|---|---|
| Wholly invented | 486 | 39.2 |
| Real name, wrong details | 374 | 30.2 |
| Jurisdiction transposition | 218 | 17.6 |
| Composite citation | 162 | 13.1 |

---

## Replication

To replicate the statistical analysis:

```bash
pip install pandas scipy statsmodels numpy
python code/statistical_analysis.py
```

To run the extraction pipeline on new model outputs:

```bash
pip install spacy pandas
python -m spacy download en_core_web_sm
python code/extraction_pipeline.py --input your_model_outputs.csv
```

---

## Citation

If you use this dataset or code, please cite:

```
@article{murugasen2025hallucination,
  title   = {Citation Hallucination in AI-Assisted Legal Research: 
             A Cross-Model, Cross-Domain, Cross-Jurisdictional 
             Empirical Study},
  author  = {Murugasen, D.},
  journal = {Computer Law \& Security Review},
  year    = {2025},
  note    = {forthcoming}
}
```

---

## License

This dataset and code are released under the 
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) 
licence. You are free to share and adapt the material for any purpose, 
provided appropriate credit is given.

---

## Contact

Dr. Murugasen — [GitHub Profile](https://github.com/drmurugaz)
