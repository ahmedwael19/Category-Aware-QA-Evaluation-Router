"""Paths, thresholds, and model constants for the thesis router project."""

from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"
NOTEBOOKS_DIR = ROOT / "notebooks"
SCRIPTS_DIR = ROOT / "scripts"

DATASET_PREFIX = "synthetic_final_20260304_205519_c9e4df6d"
DATASET_FULL = DATA_DIR / f"{DATASET_PREFIX}_full.csv"
DATASET_TRAIN = DATA_DIR / f"{DATASET_PREFIX}_train.csv"
DATASET_VAL = DATA_DIR / f"{DATASET_PREFIX}_val.csv"
DATASET_TEST = DATA_DIR / f"{DATASET_PREFIX}_test.csv"
DATASET_METADATA = DATA_DIR / f"{DATASET_PREFIX}_metadata.json"

EVAL_DATASET = DATA_DIR / "evaluation_dataset.csv"
EVAL_WITH_LLM = DATA_DIR / "evaluation_with_llm.csv"
HOLDOUT_CSV = DATA_DIR / "holdout_cross_generator.csv"

MODEL_TFIDF_LR = MODELS_DIR / "router_model_tfidf_lr.joblib"
MODEL_CALIBRATED = MODELS_DIR / "router_model_calibrated.joblib"
TFIDF_VECTORIZER = MODELS_DIR / "router_tfidf_vectorizer.joblib"
LABEL_ENCODER = MODELS_DIR / "router_label_encoder.joblib"
EMBEDDINGS_TRAIN = MODELS_DIR / "embeddings" / "train.npy"
EMBEDDINGS_VAL = MODELS_DIR / "embeddings" / "val.npy"
EMBEDDINGS_TEST = MODELS_DIR / "embeddings" / "test.npy"

RESULTS_ROUTER = RESULTS_DIR / "router"
RESULTS_E2E = RESULTS_DIR / "e2e"
RESULTS_REASONING = RESULTS_DIR / "reasoning"
RESULTS_VALIDATION = RESULTS_DIR / "validation"
RESULTS_ANALYSES = RESULTS_DIR / "analyses"

FEATURE_COL = "prompt"
TARGET_COL = "top_category"
CLASSES = [
    "HYBRID", "SEMANTIC", "SYMBOLIC_COUNT",
    "SYMBOLIC_METADATA", "SYMBOLIC_TIME", "UNSUPPORTED",
]

SEED = 42

SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.15
SPLIT_TEST = 0.15

# none / light / medium / heavy.
NOISE_WEIGHTS = {
    "none": 0.60,
    "light": 0.25,
    "medium": 0.10,
    "heavy": 0.05,
}

# Reject the synthetic dataset if any single token predicts a category more
# than this fraction of the time. Guards against shortcut-learnable benchmarks.
KEYWORD_LEAKAGE_THRESHOLD = 0.25

# Reject the dataset if a keyword/regex baseline exceeds this Macro-F1.
KEYWORD_BASELINE_REJECTION = 0.80

# Cosine threshold (multilingual-e5-large-instruct) above which two
# LLM-generated prompts are treated as near-duplicates and one is dropped.
SEMANTIC_DEDUP_COSINE = 0.99

# TF-IDF cosine threshold for the post-hoc near-duplicate audit.
# Flagged for inspection, not removed.
NEAR_DUP_TFIDF_COSINE = 0.95

# Confidence threshold for the routed path. Selected on the validation set as
# the smallest tau achieving >= TAU_MIN_ACCURACY on routed prompts at
# >= TAU_MIN_COVERAGE coverage.
TAU_DEFAULT = 0.85
TAU_MIN_ACCURACY = 0.99
TAU_MIN_COVERAGE = 0.80

# Expected Calibration Error target (Guo et al., 2017).
ECE_TARGET = 0.05
ECE_BINS = 15

# McNemar's test with continuity correction; Holm-Bonferroni across pairs.
SIGNIFICANCE_LEVEL = 0.05

BOOTSTRAP_N = 5000
BOOTSTRAP_CI = 0.95

TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)

EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
EMBEDDING_DIM = 1024

GENERATION_MODEL = "gpt-5.2"
EVALUATION_MODEL = "gpt-4o-mini"
REASONING_MODELS = ["o1", "o3-mini"]
STANDARD_MODELS = ["gpt-4o"]
