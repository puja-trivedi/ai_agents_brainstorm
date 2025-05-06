# Prompts and Output

## Prompt 1
*Output located [here](./prompt_1)*
```
Task: Perform Named Entity Recognition (NER) with a deep understanding of neuroscience-specific terminology. Use precise and domain-relevant entity labels.
Return the output in a training-ready JSON object for spaCy NER training, provide the entity,start position, end position, and respective label.
Here is an example output: 
```json 'text': 'Drug addiction is a chronic','entities': ['entity': 'Drug addiction','start': 0,'end': 13,'label': 'DISORDER']```
The text is provided here: {neuroscience_text}
```

## Prompt 2
*Output located [here](./prompt_2)*
```
As a highly skilled neuroscience expert with extensive experience in named entity recognition and research publication entity annotation, your task is to meticulously extract and classify all relevant entities from the provided neuroscience text.

For each identified entity, please provide its exact text span, its starting character position, its ending character position, and its corresponding label.

Please ensure that all entities are accurately identified and classified according to their role within the neuroscience domain. Be precise with the start and end character positions.

Identify and classify entities relevant to neuroscience research, some examples include but are NOT limited to:
GENE, PROTEIN, CELL, BRAIN_REGION, NEUROTRANSMITTER, DISORDER, DRUG, TREATMENT, MEASUREMENT, METHOD, ANIMAL_MODEL, BEHAVIOR, CHEMICAL, EQUIPMENT, ORGANIZATION, PUBLICATION, TIME, and QUANTITY.

The final output should be formatted as a training-ready JSON object suitable for spaCy's Named Entity Recognition (NER) training.
Each object in the 'entities' list should contain the 'entity', 'start', 'end', and 'label' keys.

**Input Text:** {neuroscience_text}

**Example Output:**
```json
{{
    "text": "[Portion of the input text containing the entity]",
    "entities": [
    {{"entity": "Extracted Entity 1", "start": [start position], "end": [end position], "label": "ENTITY_LABEL_1"}},
    {{"entity": "Extracted Entity 2", "start": [start position], "end": [end position], "label": "ENTITY_LABEL_2"}}
    ]
}}
```