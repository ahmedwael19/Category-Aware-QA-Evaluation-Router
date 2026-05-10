"""Build a blinded annotation CSV for the SEMANTIC/HYBRID spot-check.

Annotators receive the conversation and the evaluation prompt but not the
GPT-5.2 answer, so their judgements validate the judge without anchoring.

Writes the answer key and per-annotator templates; per-annotator files are
created only if they do not already exist, so re-running this script does
not destroy returned annotations.

Run: python -m scripts.artifacts.generate_semantic_spotcheck
"""

import pandas as pd

from config import DATA_DIR
from evaluation.system_runner import format_conversation


def main() -> None:
    df = pd.read_csv(DATA_DIR / "evaluation_with_llm.csv")
    judged = df[df["ground_truth_source"] == "gpt5.2_judge"].copy()

    hybrid = judged[judged["category"] == "HYBRID"]
    sem_yes = judged[(judged["category"] == "SEMANTIC") & (judged["ground_truth"] == "yes")]
    sem_no = judged[(judged["category"] == "SEMANTIC") & (judged["ground_truth"] == "no")]

    n_yes = min(15, len(sem_yes))
    n_no = min(30 - n_yes, len(sem_no))

    semantic_balanced = pd.concat([
        sem_yes.sample(n=n_yes, random_state=42),
        sem_no.sample(n=n_no, random_state=42),
    ])

    sample = pd.concat([hybrid, semantic_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Spot-check sample: {len(sample)} pairs")
    print(f"  HYBRID:   {(sample['category'] == 'HYBRID').sum()}")
    print(f"  SEMANTIC: {(sample['category'] == 'SEMANTIC').sum()}")

    rows = []
    for i, (_, row) in enumerate(sample.iterrows()):
        rows.append({
            "id": i + 1,
            "source_idx": row["idx"],
            "category": row["category"],
            "prompt": row["prompt"],
            "conversation": format_conversation(row["conversation_json"]),
            "annotator_answer": "",
            "notes": "",
        })
    anno_df = pd.DataFrame(rows)

    anno_dir = DATA_DIR / "annotations"
    anno_dir.mkdir(parents=True, exist_ok=True)

    for name in ("semantic_spotcheck_annotator_1.csv", "semantic_spotcheck_annotator_2.csv"):
        out = anno_dir / name
        if out.exists():
            print(f"  skipping {name} (already exists)")
        else:
            anno_df.to_csv(out, index=False)
            print(f"  wrote {name}")

    key_df = sample[["idx", "category", "prompt", "ground_truth"]].rename(
        columns={"idx": "source_idx", "ground_truth": "gpt5_2_answer"},
    )
    key_path = anno_dir / "semantic_spotcheck_answer_key.csv"
    key_df.to_csv(key_path, index=False)
    print(f"Wrote {key_path.name}")


if __name__ == "__main__":
    main()
