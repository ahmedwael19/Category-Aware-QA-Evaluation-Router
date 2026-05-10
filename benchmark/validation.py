"""Post-generation validation metrics for the synthetic benchmark dataset.

Runs TTR, keyword leakage, style distribution, length distribution,
noise uniformity, and generation method checks.
"""

import re

_KEYWORDS = {
    "SYMBOLIC_TIME": ["minute", "hour", "second", "sla", "duration", "response time"],
    "SYMBOLIC_METADATA": ["metadata", "field", "label"],
    "SYMBOLIC_COUNT": ["count", "how many", "number of"],
    "SEMANTIC_EMPATHY": ["empathy", "empathetic", "compassion"],
    "SEMANTIC_TONE": ["tone", "professionalism", "polite"],
    "SEMANTIC_SOLUTION": ["solution", "resolve", "resolution"],
    "SEMANTIC_GREETING": ["greeting", "greet", "welcome"],
    "SEMANTIC_CLOSING": ["closing", "close", "farewell"],
    "SEMANTIC_COMPREHENSION": ["comprehension", "comprehend"],
}

def validate(df):
    """Run all validation metrics on the generated dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: prompt, category, subcategory, style, noise.

    Prints results to stdout. Does not return a value.
    """
    total = len(df)

    # 1. Type-Token Ratio (TTR) per category -- lexical diversity
    print("\n1. TYPE-TOKEN RATIO (lexical diversity)")
    print("-" * 60)
    for cat in sorted(df["category"].unique()):
        cat_prompts = df[df["category"] == cat]["prompt"]
        all_tokens = []
        for p in cat_prompts:
            all_tokens.extend(re.findall(r'\w+', p.lower()))
        types = len(set(all_tokens))
        tokens = len(all_tokens)
        ttr = types / tokens if tokens > 0 else 0
        print(f"  {cat:<25} TTR={ttr:.3f} ({types:,} types / {tokens:,} tokens)")

    # 2. Keyword leakage (forbidden words in own category)
    print("\n2. KEYWORD LEAKAGE (forbidden words in own category, threshold: 25%)")
    print("-" * 60)
    any_high = False
    for cat, kws in _KEYWORDS.items():
        cat_prompts = df[df["category"] == cat]["prompt"]
        if len(cat_prompts) == 0:
            continue
        for kw in kws:
            hits = sum(1 for p in cat_prompts if kw in p.lower())
            pct = hits / len(cat_prompts) * 100
            if pct > 25:
                print(f"  WARNING {cat}: '{kw}' in {pct:.0f}% of prompts")
                any_high = True
    if not any_high:
        print("  All categories pass (<25% leakage)")

    # 3. Style distribution
    print("\n3. STYLE DISTRIBUTION")
    print("-" * 60)
    q = (df["style"] == "question").sum()
    print(f"  Question: {q/total*100:.1f}% (target: ~50%)")
    print(f"  Instruction: {(total-q)/total*100:.1f}% (target: ~50%)")

    # 4. Length distribution
    print("\n4. LENGTH DISTRIBUTION")
    print("-" * 60)
    lens = df["prompt"].str.len()
    print(f"  Mean: {lens.mean():.0f} | Median: {lens.median():.0f} | Max: {lens.max()}")
    for lo, hi, label in [(0, 100, "Short"), (100, 500, "Medium"), (500, 1000, "Long")]:
        n = ((lens >= lo) & (lens < hi)).sum()
        print(f"  {label} ({lo}-{hi}): {n:,} ({n/total*100:.1f}%)")

    # 5. Noise distribution per category (chi-squared test)
    print("\n5. NOISE DISTRIBUTION PER CATEGORY")
    print("-" * 60)
    from scipy.stats import chi2_contingency
    ct = df.groupby(["category", "noise"]).size().unstack(fill_value=0)
    try:
        chi2, p, dof, _ = chi2_contingency(ct.values)
        print(f"  Chi-squared test for noise uniformity: chi2={chi2:.1f}, p={p:.4f}, dof={dof}")
        if p < 0.05:
            print("  WARNING: Noise distribution significantly differs across categories")
        else:
            print("  OK: No significant noise-category correlation")
    except Exception as e:
        print(f"  Could not compute chi-squared: {e}")
    print(ct.to_string())

    # 6. Noise indicators (surface-level)
    print("\n6. NOISE INDICATORS")
    print("-" * 60)
    ps = df["prompt"].tolist()
    lc = sum(1 for p in ps if p and p[0].islower())
    np_ = sum(1 for p in ps if p and p.rstrip()[-1:] not in ".?!")
    dbl = sum(1 for p in ps if "??" in p or "..." in p)
    inf = sum(1 for p in ps if any(w in p.lower() for w in ["plz", "gonna", "wanna", "cuz", " u "]))
    print(f"  Lowercase start: {lc/total*100:.1f}%")
    print(f"  Missing punct:   {np_/total*100:.1f}%")
    print(f"  Double punct:    {dbl/total*100:.1f}%")
    print(f"  Informal words:  {inf/total*100:.1f}%")

    # 7. Generation method distribution
    print("\n7. GENERATION METHOD DISTRIBUTION")
    print("-" * 60)
    method_ct = df.groupby(["category", "subcategory"]).size().unstack(fill_value=0)
    print(method_ct.to_string())

    print(f"\nTOTAL: {total:,} prompts | {df['category'].nunique()} categories")
