"""End-to-end evaluation harness.

- deterministic: Conversation/Message model and check_* functions
- conversation_generator: synthetic conversation generation
- prompt_templates: LLM prompt templates (binary, reject, ground truth)
- answer_parser: LLM output normalisation (yes/no/reject/invalid)
- dataset_builder: evaluation triples from the test split
- system_runner: deterministic eval + LLM baselines
- comparison: full router-based system comparison
- reliability: 5-run LLM reliability test
- latency: uncached latency measurement
- statistics: McNemar, bootstrap CIs, per-category accuracy
"""
