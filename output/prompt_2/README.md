
# Entity Comparison Summary

This report compares named entity extraction results between two JSON files:

- **File 1 [Google Gemini-2.0-flash](./zero-shot-ner_smaller-chunks_google_gemini-2.0-flash-001_20250505_222711.json)**: 
- **File 2 [OpenAI GPT-4o-mini](zero-shot-ner_smaller-chunks_openai_gpt-4o-mini_20250505_184812.json)**: 

## Overall Comparison

- Total entities in **Google Gemini-2.0-flash**: **929**
- Total entities in **OpenAI GPT-4o-mini**: **646**

### Entity Overlap

- **496** entities were extracted by both models (after standardizing text).
- **466** of those had the **same label** assigned in both models.

## Top Label Disagreements

| Google Gemini-2.0-flash Label | GPT-4o-mini Label | Count |
|-------------------------------|------------------|-------|
| PROTEIN | RECEPTOR | 6 |
| TREATMENT | MEASUREMENT | 3 |
| PROTEIN | GENE | 2 |
| GENE | PROTEIN | 2 |
| GENE | CELL | 2 |


## Entity Count by Label Type

This table compares the number of entities extracted for each label:

|                  |   Google Gemini-2.0-flash |   OpenAI GPT-4o-mini |
|:-----------------|--------------------------:|---------------------:|
| ANIMAL_MODEL     |              56 |                   41 |
| BEHAVIOR         |              23 |                    8 |
| BRAIN_REGION     |              68 |                   45 |
| CELL             |             186 |                  103 |
| CELLULAR_PROCESS |               2 |                    0 |
| CHEMICAL         |              25 |                   15 |
| DISORDER         |              14 |                   15 |
| DRUG             |              92 |                   40 |
| EQUIPMENT        |               7 |                    5 |
| FORMAT           |               0 |                    1 |
| GENE             |             172 |                  116 |
| LOCATION         |               0 |                    1 |
| MEASUREMENT      |              46 |                   28 |
| METHOD           |              48 |                   36 |
| NEUROTRANSMITTER |               8 |                   10 |
| ORGANIZATION     |               4 |                    6 |
| PEPTIDE          |               0 |                    1 |
| PROTEIN          |              26 |                   18 |
| PUBLICATION      |              70 |                   70 |
| QUANTITY         |              39 |                   34 |
| RECEPTOR         |               0 |                    7 |
| SOFTWARE         |               0 |                    1 |
| TIME             |              24 |                   27 |
| TREATMENT        |              19 |                   10 |
| VERSION          |               0 |                    8 |