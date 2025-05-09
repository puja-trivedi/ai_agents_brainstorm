import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
from tabulate import tabulate  # pip install tabulate
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from upsetplot import from_contents, UpSet
import seaborn as sns  # small exception just for heatmap!

def compare_ner_jsons(filepaths):
    """
    Compare multiple NER JSON files on total entities, unique entities, unique labels, and per-section counts.
    """
    def normalize_entity(text):
        return text.strip().lower()

    def extract_ner_stats(filepath):
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)

        entities_per_section = defaultdict(int)
        total_entities = 0
        unique_entities = set()
        unique_labels = set()

        for section_key, section in data.get("results", {}).items():
            entities = section.get("entities", [])
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

    stats = all_stats
    # suppose stats is your list of 4 dicts
    df_summary = pd.DataFrame([{
        "model": s["model"],
        "total": s["total_entities"],
        "unique_entities": s["unique_entities_count"],
        "unique_labels": s["unique_labels_count"]
    } for s in stats]).set_index("model")

    # 1) Summary table
    print(df_summary)

    # 2) Section counts
    sec_df = (
        pd.DataFrame({s["model"]: s["entities_per_section"] 
                    for s in stats})
        .fillna(0).astype(int)
    )
    sec_df.plot.bar(figsize=(8,5))
    plt.ylabel("Entities per section")
    plt.show()

    # 3) Jaccard similarity
    def jaccard(a, b):
        a, b = set(a), set(b)
        return len(a & b) / len(a | b) if a or b else 0

    models = [s["model"] for s in stats]
    jac = pd.DataFrame(index=models, columns=models, dtype=float)
    for i, j in combinations(models, 2):
        si = next(s for s in stats if s["model"] == i)
        sj = next(s for s in stats if s["model"] == j)
        score = jaccard(si["unique_entities"], sj["unique_entities"])
        jac.at[i, j] = jac.at[j, i] = score
    for m in models:
        jac.at[m, m] = 1.0

    sns.heatmap(jac, annot=True, cmap="viridis")
    plt.title("Entity Set Jaccard Similarity")
    plt.show()

    # 4) UpSet plot (requires upsetplot)

    contents = {s["model"]: s["unique_entities"] for s in stats}
    up = from_contents(contents)
    UpSet(up).plot()
    plt.show()


    # 5) Label-Set Overlap (Jaccard)
    label_sets = {s["model"]: set(s["unique_labels"]) for s in all_stats}

    # compute Jaccard matrix
    models = list(label_sets)
    jac_labels = pd.DataFrame(index=models, columns=models, dtype=float)
    for i,j in combinations(models, 2):
        a, b = label_sets[i], label_sets[j]
        jac_labels.at[i,j] = jac_labels.at[j,i] = len(a&b)/len(a|b)
    for m in models:
        jac_labels.at[m,m] = 1.0

    sns.heatmap(jac_labels, annot=True, cmap="Oranges")
    plt.title("Label‑Set Jaccard Similarity")
    plt.show()

    # # labels in all models
    # all_labels = set()
    # for stat in all_stats:
    #     all_labels.update(stat["unique_labels"])
    # print("\n===== 📊 All Unique Labels Aggregated Across Models =====\n")
    # print(f"Total number of unique labels across all models: {len(all_labels)}")
    # print(f"Unique labels across all models: {sorted(all_labels)}")

    # # labels that appear in all models
    # print("\n===== 📊 Labels Common to All Models =====\n")
    # common_labels = set.intersection(*(stat["unique_labels"] for stat in all_stats))
    # print(f"Total number of common labels across all models: {len(common_labels)}")
    # print(f"Common labels across all models: {sorted(common_labels)}")

    # print("\n===== 📊 Labels Unique to Each Model =====\n")
    # for stat in all_stats:
    #     print(f"MODEL: {stat['model']}")
    #     labels_unique_to_model = stat["unique_labels"]
    #     for other_stat in all_stats:
    #         if stat["model"] != other_stat["model"]:
    #             labels_unique_to_model -= other_stat["unique_labels"]
    #     print(f"Number of unique labels only contained in {stat['model']}: {len(labels_unique_to_model)}")
    #     print(f"Unique labels for {stat['model']}: {sorted(labels_unique_to_model)}\n") 


    return summary_df


def main():
    filepaths = [
        "output/prompt_5/phillips_claude-3.7-sonnet_2025-05-08_23-47-46.json",
        "output/prompt_5/phillips_deepseek-chat-v3-0324_2025-05-09_00-09-01.json",
        "output/prompt_5/phillips_gemini-2.0-flash-001_2025-05-08_23-42-35.json",
        "output/prompt_5/phillips_gpt-4o-mini_2025-05-08_23-32-22.json"
    ]

    summary_df = compare_ner_jsons(filepaths)





if __name__ == "__main__":
    main()
