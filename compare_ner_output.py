import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
from tabulate import tabulate  # pip install tabulate

def compare_ner_jsons(filepaths):
    """
    Compare multiple NER JSON files on total entities, unique entities, unique labels, and per-section counts.
    """
    def normalize_entity(text):
        return text.strip().lower()

    def extract_ner_stats(filepath):
        with open(filepath) as f:
            data = json.load(f)

        entities_per_section = defaultdict(int)
        total_entities = 0
        unique_entities = set()
        unique_labels = set()

        for section_key, section in data.get("output", {}).items():
            entities = section.get("entities", section.get("entity",[]))
            entities_per_section[section_key] += len(entities)
            total_entities += len(entities)
            for ent in entities:
                unique_entities.add(normalize_entity(ent["entity"]))
                unique_labels.add(ent["label"])

        return {
            "model": data.get("model_name", "Unknown"),
            "total_entities": total_entities,
            "unique_entities_count": len(unique_entities),
            "unique_labels_count": len(unique_labels),
            "entities_per_section": dict(entities_per_section),
            "unique_entities": unique_entities,
            "unique_labels": unique_labels
        }

    all_stats = [extract_ner_stats(fp) for fp in filepaths]

    # Summary table
    summary_df = pd.DataFrame([{
        "Model": stat["model"],
        "# Total Entities": stat["total_entities"],
        "# Unique Entities": stat["unique_entities_count"],
        "# Unique Labels": stat["unique_labels_count"]
    } for stat in all_stats]).sort_values(by="Model")
    print("\n===== 📊 Summary of NER Statistics =====\n")
    print(tabulate(summary_df, headers="keys", tablefmt="grid", showindex=False))

    # labels in all models
    all_labels = set()
    for stat in all_stats:
        all_labels.update(stat["unique_labels"])
    print("\n===== 📊 Labels in All Models =====\n")
    print(f"Total number of unique labels across all models: {len(all_labels)}")
    print(f"Unique labels across all models: {all_labels}")

    print("\n===== 📊 Labels Unique to Each Model =====\n")
    for stat in all_stats:
        print(f"MODEL: {stat['model']}")
        labels_unique_to_model = stat["unique_labels"]
        for other_stat in all_stats:
            if stat["model"] != other_stat["model"]:
                labels_unique_to_model -= other_stat["unique_labels"]
        print(f"Number of unique labels only contained in {stat['model']}: {len(labels_unique_to_model)}")
        print(f"Unique labels for {stat['model']}: {labels_unique_to_model}\n")
    return summary_df


def main():
    filepaths = [
        "output/creating_benchmark/lin_anthropic_claude-3.7-sonnet_20250507_214630.json",
        "output/creating_benchmark/lin_deepseek_deepseek-chat-v3-0324:free_20250507_200333.json",
        "output/creating_benchmark/lin_google_gemini-2.0-flash-001_20250507_192808.json",
        "output/creating_benchmark/lin_openai_gpt-4o-mini_20250507_192255.json"
    ]

    summary_df = compare_ner_jsons(filepaths)





if __name__ == "__main__":
    main()
