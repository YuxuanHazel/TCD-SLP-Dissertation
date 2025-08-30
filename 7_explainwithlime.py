"""
Explain translationese predictions with SHAP/LIME and feed explanations into your prompt generator.

What this script does
---------------------
1) Rebuilds the exact same df_balanced/train/dev/test split as in `5_filter_high_probability.py` so that
   features/indices align perfectly (critical for correct explanations).
2) Loads the trained MLP and the StandardScaler you used (the scaler is re-fit on train just like before).
3) For each sentence in `translationese_test_subset.csv` (high-prob translationese), computes a **local explanation**
   using **LIME (Tabular)** over the 14 handcrafted features **while holding the sentence's BERT embedding fixed**.
   - This mirrors how you interpret: we only vary interpretable features, not the 768-dim embeddings.
4) (Optional, slower) Computes **Kernel SHAP** per instance over the 14 features with the same "fixed-embedding" trick.
   Off by default; enable via CLI flag.
5) Writes an enriched CSV with per-sentence top-k features and their signed importances (for downstream prompts).

Notes
-----
• You can control whether to run LIME and/or SHAP; LIME is much faster and good for per-sentence prompts.
• This file does not change your classifier. It only explains predictions consistently with your current pipeline.
• Assumes the same paths as your current code. Adjust PATHS below if needed.

Usage
-----
python 7_explain_with_shap_lime.py \
  --top_k 3 \
  --run_lime 1 \
  --run_shap 0 \
  --shap_samples 500

Outputs
-------
• result/translationese_explanations.csv  (per-instance local importance over 14 features)
• result/prompts_for_llm_from_lime.csv   (drop-in replacement for 6_generate_prompts_from_translationese.py)

Dependencies
------------
pip install numpy pandas joblib scikit-learn lime shap
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Optional explainers
try:
    import shap  # type: ignore
except Exception:
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer  # type: ignore
except Exception:
    LimeTabularExplainer = None

# ========================
# PATH CONFIG (edit if needed)
# ========================
FULL_DF_CSV = Path(r"D:\Liuyuxuan\MUC\Slp\dissertation\新建文件夹\result\2. feature_dataset.csv")
BERT_NPY    = Path(r"result/bert_embeddings_downsample.npy")
MODEL_PKL   = Path(r"D:\Liuyuxuan\MUC\Slp\dissertation\新建文件夹\trained_models\mlp.pkl")
HIGH_PROB_CSV = Path(r"result/translationese_test_subset.csv")

EXPLAIN_OUT_CSV = Path(r"D:\Liuyuxuan\MUC\Slp\dissertation\新建文件夹\result\tiaocanshiyunxing\retrive_src_from_lime_v4.csv")
PROMPTS_OUT_CSV = Path(r"D:\Liuyuxuan\MUC\Slp\dissertation\新建文件夹\result\tiaocanshiyunxing\/prompts_for_llm_from_lime_v4.csv")



# ========================
# Feature schema (must match training)
# ========================
HANDCRAFTED_COLS = [
    "sent_length", "clause_depth", "passive_ratio", "subordination_ratio", "nominalization_ratio",
    "ttr", "lexical_density", "repetition_frequency", "avg_word_length", "func_content_ratio",
    "connective_count", "cohesive_marker_ratio", "pronoun_ratio", "punctuation_density",
]

ISSUE_LABELS = {
    "sent_length": "too long",
    "passive_ratio": "too passive",
    "connective_count": "too many connectives",
    "func_content_ratio": "too many function words",
    "punctuation_density": "punctuation heavy",
    "repetition_frequency": "repetition",
    "pronoun_ratio": "too many pronouns",
    "subordination_ratio": "too many subordinate clauses",
    "nominalization_ratio": "too much nominalization",
    "ttr": "low lexical variety",
    "lexical_density": "overly dense",
    "clause_depth": "deeply nested syntax",
    "cohesive_marker_ratio": "too many cohesive markers",
    "avg_word_length": "words too long",
}
ISSUE_SOLUTIONS = {
    "sent_length": "Split long sentences once.",
    "passive_ratio": "Prefer active if natural.",
    "connective_count": "Cut redundant connectives.",
    "func_content_ratio": "Trim filler function words.",
    "punctuation_density": "Simplify heavy punctuation.",
    "repetition_frequency": "Remove needless repetition.",
    "pronoun_ratio": "Reduce pronoun overuse.",
    "subordination_ratio": "Limit excessive subordination.",
    "nominalization_ratio": "Turn nouns back into verbs.",
    "ttr": "Vary wording slightly.",
    "lexical_density": "Loosen dense noun phrases.",
    "clause_depth": "Flatten nested clauses.",
    "cohesive_marker_ratio": "Keep only necessary markers.",
    "avg_word_length": "Prefer simpler common words.",
}

# ========================
# Utility
# ========================

def build_df_balanced(full_df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    df = full_df.copy()
    df["row_id"] = range(len(df))
    df_tec = df[df["source"] == "TEC"]
    df_nec = df[df["source"] == "NEC"]
    nec_sample = df_nec.sample(n=len(df_tec) * 5, random_state=random_state)
    df_balanced = pd.concat([df_tec, nec_sample]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_balanced["row_id"] = range(len(df_balanced))
    return df_balanced


def split_with_row_tracking(X: np.ndarray, y: np.ndarray, row_ids: pd.Series, random_state: int = 42):
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X, y, row_ids, test_size=0.30, stratify=y, random_state=random_state
    )
    X_dev, X_test, y_dev, y_test, idx_dev, idx_test = train_test_split(
        X_temp, y_temp, idx_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )
    return (X_train, y_train, idx_train), (X_dev, y_dev, idx_dev), (X_test, y_test, idx_test)


def make_predict_fn_for_instance(model, scaler: StandardScaler, fixed_embed: np.ndarray):

    def f(z14: np.ndarray) -> np.ndarray:
        z = np.array(z14, dtype=float, copy=True)

        z_std = scaler.transform(z)

        n = z_std.shape[0]
        emb = np.repeat(fixed_embed.reshape(1, -1), n, axis=0)
        X_full = np.hstack([z_std, emb])
        # 预测翻译腔概率        p_trans = model.predict_proba(X_full)[:, 1]  # P(translationese)
        p_no_trans = 1.0 - p_trans                    # P(no_translationese)
        return np.column_stack([p_no_trans, p_trans])
    return f



def run_lime_for_instance(lime_explainer, instance_raw14: np.ndarray, predict_fn, feature_names: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
    exp = lime_explainer.explain_instance(
        data_row=instance_raw14,
        predict_fn=predict_fn,
        num_features=top_k,
    )
    # exp.as_list() returns pairs like ("feature <= 0.12", weight). We map back to clean names by parsing.
    results: List[Tuple[str, float]] = []
    for name, weight in exp.as_list():
        # Heuristic: find feature name prefix
        clean = None
        for feat in feature_names:
            if name.startswith(feat) or feat in name:
                clean = feat
                break
        clean = clean or name
        results.append((clean, float(weight)))
    return results


def run_kernel_shap_for_instance(shap_background14: np.ndarray, instance_raw14: np.ndarray, predict_fn, feature_names: List[str], shap_samples: int = 500) -> List[Tuple[str, float]]:
    if shap is None:
        raise ImportError("shap is not installed. pip install shap")
    explainer = shap.KernelExplainer(predict_fn, shap_background14)
    shap_values = explainer.shap_values(instance_raw14.reshape(1, -1), nsamples=shap_samples)
    # shap_values is a list when there are multiple classes; for binary with proba of class1, get array
    if isinstance(shap_values, list):
        sv = np.array(shap_values)[1].reshape(-1)  # class 1
    else:
        sv = np.array(shap_values).reshape(-1)
    return sorted([(feature_names[i], float(sv[i])) for i in range(len(feature_names))], key=lambda x: abs(x[1]), reverse=True)


def build_issue_string(top_feats: List[str]) -> str:
    labels = [ISSUE_LABELS.get(f, f) for f in top_feats]
    return ", ".join(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--run_lime", type=int, default=1)
    parser.add_argument("--run_shap", type=int, default=0)
    parser.add_argument("--shap_samples", type=int, default=500)
    args = parser.parse_args()

    # 1) Load full data and rebuild df_balanced exactly as before
    full_df = pd.read_csv(FULL_DF_CSV)
    df_balanced = build_df_balanced(full_df, random_state=42)
    y = (df_balanced["source"] == "TEC").astype(int).values

    X_hand = df_balanced[HANDCRAFTED_COLS].values.astype(float)
    X_bert = np.load(BERT_NPY.as_posix())
    assert X_hand.shape[0] == X_bert.shape[0], "Handcrafted rows must match BERT embeddings"

    # 2) Train/dev/test split with row tracking (same random_state)
    (Xtr_hand, y_train, idx_train), (Xdev_hand, y_dev, idx_dev), (Xte_hand, y_test, idx_test) = split_with_row_tracking(
        X_hand, y, df_balanced["row_id"], random_state=42
    )
    (Xtr_bert, _, _), (_, _, _), (Xte_bert, _, _) = split_with_row_tracking(
        X_bert, y, df_balanced["row_id"], random_state=42
    )

    # 3) Standardize handcrafted features like in training
    scaler = StandardScaler()
    Xtr_hand_std = scaler.fit_transform(Xtr_hand)
    Xte_hand_std = scaler.transform(Xte_hand)

    # 4) Load trained model
    model = joblib.load(MODEL_PKL)

    # 5) Load the high-probability subset produced earlier
    df_hp = pd.read_csv(HIGH_PROB_CSV)
    if "row_id" not in df_hp.columns:
        raise ValueError("translationese_test_subset.csv must include 'row_id' so we can align to test set")

    # Only keep rows that are actually in the test split (safety)
    test_row_ids = set(idx_test.tolist())
    df_hp = df_hp[df_hp["row_id"].isin(test_row_ids)].copy()
    if df_hp.empty:
        raise RuntimeError("No rows from high-prob subset align with current test split. Check random_state and input files.")

    # Map row_id -> test index position so we can fetch standardized 14 features and the fixed 768-dim embedding
    rowid_to_pos = {rid: pos for pos, rid in enumerate(idx_test.tolist())}

    # Build LIME background (over 14 raw features) using training RAW features (not standardized)
    lime_explainer = None
    if args.run_lime:
        if LimeTabularExplainer is None:
            raise ImportError("lime is not installed. pip install lime")
        lime_explainer = LimeTabularExplainer(
            training_data=Xtr_hand,  # RAW values
            feature_names=HANDCRAFTED_COLS,
            discretize_continuous=True,
            mode="classification",
            class_names=["no_translationese", "translationese"],

        )

    # For SHAP background (RAW 14-dim). We'll rely on predict_fn to standardize internally
    shap_background14 = None
    if args.run_shap:
        if shap is None:
            raise ImportError("shap is not installed. pip install shap")
        # Use a modest background sample for speed
        bg_idx = np.random.RandomState(42).choice(Xtr_hand.shape[0], size=min(100, Xtr_hand.shape[0]), replace=False)
        shap_background14 = Xtr_hand[bg_idx]

    records = []

    for _, row in df_hp.iterrows():
        rid = int(row["row_id"])  # row id in df_balanced
        pos = rowid_to_pos[rid]

        # Pull RAW 14 features and RAW 768 for this test item
        instance_raw14 = Xte_hand[pos]
        instance_embed768 = Xte_bert[pos]

        # Build predict_fn that standardizes 14 features and concatenates fixed embedding
        predict_fn = make_predict_fn_for_instance(model, scaler, instance_embed768)

        explanations: Dict[str, float] = {}

        if lime_explainer is not None:
            pairs = run_lime_for_instance(
                lime_explainer=lime_explainer,
                instance_raw14=instance_raw14,
                predict_fn=predict_fn,
                feature_names=HANDCRAFTED_COLS,
                top_k=max(args.top_k, 5),  # pull a few more then trim
            )
            for name, w in pairs:
                explanations[name] = explanations.get(name, 0.0) + float(w)

        if args.run_shap and shap_background14 is not None:
            shap_pairs = run_kernel_shap_for_instance(
                shap_background14=shap_background14,
                instance_raw14=instance_raw14,
                predict_fn=predict_fn,
                feature_names=HANDCRAFTED_COLS,
                shap_samples=args.shap_samples,
            )
            # merge with (signed) shap values
            for name, w in shap_pairs:
                explanations[name] = explanations.get(name, 0.0) + float(w)

        # Rank by absolute weight and keep top_k
        if not explanations:
            # fallback: use standardized z-score magnitudes if no explainer ran
            z = scaler.transform(instance_raw14.reshape(1, -1)).reshape(-1)
            idxs = np.argsort(np.abs(z))[::-1][: args.top_k]
            top_feats = [HANDCRAFTED_COLS[i] for i in idxs]
            top_weights = [float(z[i]) for i in idxs]
        else:
            ranked = sorted(explanations.items(), key=lambda x: abs(x[1]), reverse=True)
            top = ranked[: args.top_k]
            top_feats = [t[0] for t in top]
            top_weights = [float(t[1]) for t in top]

        issues = build_issue_string(top_feats)

        rec = {
            **row.to_dict(),
            "top_features": ",".join(top_feats),
            "top_weights": ",".join([f"{w:.4f}" for w in top_weights]),
            "issues_from_expl": issues,
        }
        records.append(rec)

    out_df = pd.DataFrame(records)
    EXPLAIN_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(EXPLAIN_OUT_CSV, index=False, encoding="utf-8-sig")

    def build_solution_text(top_features_str: str) -> str:
        feats = [f.strip() for f in str(top_features_str).split(",") if f.strip()]
        sols = []
        seen = set()
        for f in feats:
            s = ISSUE_SOLUTIONS.get(f, "")
            if s and s not in seen:
                sols.append(s)
                seen.add(s)
        # 回退：如果没有匹配到任何解决方案，给一个通用指令
        return " ".join(sols) if sols else "Apply minimal edits to improve fluency."

    # 由 top_features 衍生出 solutions 文本
    out_df["solutions_from_expl"] = out_df["top_features"].apply(build_solution_text)

    # Also emit a prompts CSV in the same format your 6_* script expects

    #     ========v2
    # PROMPT_TEMPLATE = (
#     "[INST] You are an expert English editor.\n"
#     "The following sentence shows signs of translationese: \"{sentence}\"\n"
#     "Key issues detected: {issues}.\n"
#     "Suggested revision: Address the above issues to improve fluency while preserving the original style and voice.\n"
#     "Rules:\n"
#     "- Preserve the original meaning and all factual content\n"
#     "- Preserve named entities (people, places), numbers, colors, and body-part details\n"
#     "- Preserve negation and modality (not/never/might/would/I think, etc.)\n"
#     "- Maintain the original grammatical voice (active/passive), tense, aspect, and person unless ungrammatical or unclear\n"
#     "- Keep the original style, tone, and register of the input sentence\n"
#     "- Make minimal necessary edits to resolve the listed issues\n"
#     "- Do NOT add or omit information\n"
#     "- Do NOT include explanations in the output\n"
#     "- Output only the revised sentence on a single line, without quotes\n"
#     "[/INST]"
# )

#     ========v3
#     PROMPT_TEMPLATE = (
#     "[INST] You are an expert English editor.\n"
#     "The following sentence shows signs of translationese: \"{sentence}\"\n"
#     "Key issues detected: {issues}.\n"
#     "Suggested fixes: {solutions}\n"
#     "Rules:\n"
#     "- Preserve meaning and all factual details\n"
#     "- Preserve names, numbers, colors, and body-part details\n"
#     "- Preserve negation and modality (not/never/might/would/I think, etc.)\n"
#     "- Keep the original style and register; maintain the original grammatical voice (active/passive), tense, aspect, and person unless ungrammatical\n"
#     "- Make minimal edits; do not add or remove information\n"
#     "- Output only the revised sentence on a single line, without quotes\n"
#     "[/INST]"
# )

    # =========v4
    PROMPT_TEMPLATE = (
        "[INST] You are an expert English editor.\n"
        "The following sentence shows signs of translationese: \"{sentence}\"\n"
        "Suggested fixes: {solutions}\n"
        "Rules:\n"
        "- Preserve the original meaning and all factual details\n"
        "- Preserve names, numbers, colors, and body-part details\n"
        "- Preserve negation and modality (not/never/might/would/I think, etc.)\n"
        "- Keep the original style and register; maintain the original grammatical voice (active/passive), tense, aspect, and person unless ungrammatical\n"
        "- Make minimal necessary edits following the suggested fixes\n"
        "- Do NOT add or omit information\n"
        "- Output only the revised sentence on a single line, without quotes\n"
        "[/INST]"
    )

    def build_solution_text(top_features_str: str) -> str:
        feats = [f.strip() for f in str(top_features_str).split(",") if f.strip()]
        sols = [ISSUE_SOLUTIONS.get(f, "") for f in feats]
        sols = [s for s in sols if s]
        # 简洁合并：空格拼在一起即可（或用 "; " 更清楚）
        return " ".join(sols)

    out_df["solutions_from_expl"] = out_df["top_features"].apply(build_solution_text)


    prompts_df = pd.DataFrame({
        "sentence": out_df["sentence"],
        # 保留 issues 方便分析（可选）
        "issues": out_df.get("issues_from_expl", np.nan),
        "solutions": out_df["solutions_from_expl"],
        "prompt": [
            PROMPT_TEMPLATE.format(
                sentence=" ".join(str(s).split()),
                solutions=sol
            )
            for s, sol in zip(out_df["sentence"], out_df["solutions_from_expl"])
        ],
        "pred_prob": out_df.get("pred_prob", np.nan),
    })
    prompts_df.to_csv(PROMPTS_OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"✅ Saved local explanations to: {EXPLAIN_OUT_CSV}")
    print(f"✅ Saved LLM prompts (from explanations) to: {PROMPTS_OUT_CSV}")


if __name__ == "__main__":
    main()
