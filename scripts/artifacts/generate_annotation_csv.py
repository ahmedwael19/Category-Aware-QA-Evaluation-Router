"""Build the blank annotation-study CSV and answer key from the synthetic dataset.

Produces the template that was sent to annotators (20 prompts per top
category, stratified) plus a hidden answer key. Overwrites
`data/annotations/annotation_study.csv` and `annotation_answer_key.csv`; the
per-annotator files (`annotation_study_annotator_{1,2}.csv`) are created only
if they do not already exist, so re-running this script does not destroy
returned annotations.

Run: python -m scripts.artifacts.generate_annotation_csv
"""

import pandas as pd

from config import DATA_DIR, DATASET_FULL


def main() -> None:
    print(f"Using: {DATASET_FULL}")
    df = pd.read_csv(DATASET_FULL)
    print(f"Total: {len(df)}")

    samples = [
        df[df["top_category"] == cat].sample(n=20, random_state=42)
        for cat in sorted(df["top_category"].unique())
    ]
    sample_df = pd.concat(samples).sample(frac=1, random_state=42).reset_index(drop=True)

    anno = pd.DataFrame({
        "id": range(1, len(sample_df) + 1),
        "prompt": sample_df["prompt"].values,
        "annotator_category": "",
        "annotator_confidence": "",
        "notes": "",
    })
    anno_dir = DATA_DIR / "annotations"
    anno_dir.mkdir(parents=True, exist_ok=True)

    blank_path = anno_dir / "annotation_study.csv"
    anno.to_csv(blank_path, index=False)
    print(f"Wrote {blank_path}")

    for name in ("annotation_study_annotator_1.csv", "annotation_study_annotator_2.csv"):
        out = anno_dir / name
        if out.exists():
            print(f"  skipping {name} (already exists)")
        else:
            anno.to_csv(out, index=False)
            print(f"  wrote {name}")

    key = pd.DataFrame({
        "id": range(1, len(sample_df) + 1),
        "prompt": sample_df["prompt"].values,
        "true_category": sample_df["top_category"].values,
        "fine_category": sample_df["category"].values,
        "generation_method": sample_df["subcategory"].values,
    })
    key_path = anno_dir / "annotation_answer_key.csv"
    key.to_csv(key_path, index=False)
    print(f"Wrote {key_path}")

    print(f"\nPer-category counts:\n{sample_df['top_category'].value_counts().sort_index().to_string()}")


if __name__ == "__main__":
    main()
