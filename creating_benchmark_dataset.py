import json
import re
from typing import Dict,Set, Any
from multiprocessing import Pool
import yaml

def find_entity_occurrences(
    entities_dict: Dict[str, Set[str]],
    sentences_dict: Dict[str, Any]
) -> Dict[str, str]:
    """
    Find all occurrences of each entity in the provided sentences.

    Args:
        entities_dict: Mapping of entity string -> set of labels.
        sentences_dict: Dict with a "chunks" key, value is List of dicts each
                        with "id" and "text".

    Returns:
        A dict with two keys:
          - "json": JSON-formatted string of the occurrences list.
          - "yaml": YAML-formatted string of the same.
    """
    # sort the entities_dict by keys
    chunks = sentences_dict.get("chunks", [])

    with Pool() as pool:
        results = pool.starmap(_process_entity, [(entity, labels, chunks) for entity, labels in entities_dict.items()])
    results.sort(
        key=lambda rec: (
            0 if rec['entity'][0].isalpha() else 1,
            rec['entity'].lower()
        )   
    )
    json_output = json.dumps(results, indent=2)
    yaml_output = yaml.dump(results, allow_unicode=True, indent=2, width=80)

    return {
        "json": json_output,
        "yaml": yaml_output
    }


def _process_entity(entity, labels, chunks):
    lower_ent = entity.lower()
    
    # Escape in case `entity` has regex metachars
    e = re.escape(entity)
    # build a “before” group that is either:
    #   1) ^         (start of string, zero‑width)
    #   2) (?<=\s)   (preceded by whitespace, one‑char width)
    #   3) (?<=\()  (preceded by '(', one‑char width)
    before = rf'(?:^|(?<=\s)|(?<=\())'
    # build an “after” lookahead that is:
    #   - whitespace, or one of . , ; : ! ?, or ')', or end‑of‑string
    after  = rf'(?=(?:\s|[.,;:!?]|\)|$))'

    pattern = re.compile(before + e + after, flags=re.IGNORECASE)
    matched = []
    for chunk in chunks:
        text_lower = chunk.get("text", "").lower()
        if pattern.search(text_lower):
            matched.append({
                "text": chunk["text"],
                "id": chunk["id"] 
            })

    return {
        "entity": entity,
        "labels": labels,
        "sentences": matched
    }

if __name__ == "__main__":
    entities = json.load(open("output/prompt_5/benchmark_generation/phillips_merged_filtered_entities_allmodels.json", "r", encoding="utf-8"))
    sentences = json.load(open("data/philips_sentences.json", "r", encoding="utf-8"))
    outputs = find_entity_occurrences(entities, sentences)
    # with open("output.json", "w", encoding="utf-8") as f:
    #     f.write(outputs["json"])
    with open("output.yaml", "w", encoding="utf-8") as f:
        f.write(outputs["yaml"])
