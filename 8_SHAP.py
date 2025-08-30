"""
MLP_SHAP_Final120.py
---------------------
- Extract translationese sentences from final.csv
- Match feature_dataset.csv to extract 14 linguistic features
- Load the corresponding BERT embeddings to construct a 782-dimensional input
- Use Kernel SHAP to analyze the global importance of MLP predictions for translationese
- Output as CSV + bar chart
"""


import os
import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========== 文件路径 ==========
feature_csv = r"D:\Liuyuxuan\MUC\Slp\dissertation\新建文件夹\result\2. feature_dataset.csv"
final_csv = r"D:\Liuyuxuan\MUC\Slp\dissertation\新建文件夹\result\tiaocanshiyunxing\comet\final.csv"
model_path = r"D:\Liuyuxuan\MUC\Slp\dissertation\新建文件夹\trained_models\mlp.pkl"
bert_path = r"D:\Liuyuxuan\MUC\Slp\dissertation\新建文件夹\result\bert_embeddings_downsample.npy"
output_dir = r"D:\Liuyuxuan\MUC\Slp\dissertation\新建文件夹\新建文件夹\result\shap_data_mlp"
os.makedirs(output_dir, exist_ok=True)



# ========== Load data ==========
print("Loading feature data and final samples...")
df_feat = pd.read_csv(feature_csv)
df_final = pd.read_csv(final_csv)

# ========== Load corresponding BERT embeddings ==========
matched_df = df_feat[df_feat["sentence"].isin(df_final["sentence"])].copy()
matched_df.reset_index(drop=True, inplace=True)
print(f"✅ Successfully matched {len(matched_df)} samples")


# ========== Extract features ==========
handcrafted_cols = [
    "sent_length", "clause_depth", "passive_ratio", "subordination_ratio", "nominalization_ratio",
    "ttr", "lexical_density", "repetition_frequency", "avg_word_length", "func_content_ratio",
    "connective_count", "cohesive_marker_ratio", "pronoun_ratio", "punctuation_density"
]

# 🔒 Missing value protection: avoid SHAP errors
if matched_df[handcrafted_cols].isnull().any().any():
    print("⚠️ Warning: NaN values found, filling with 0")
    matched_df[handcrafted_cols] = matched_df[handcrafted_cols].fillna(0)

X_hand = matched_df[handcrafted_cols].values

# ========== Load corresponding BERT embeddings ==========
print(" Load corresponding BERT embeddings...")
bert_all = np.load(bert_path)  # 与 feature_dataset.csv 对应
matched_indices = df_feat[df_feat["sentence"].isin(df_final["sentence"])].index
bert_matched = bert_all[matched_indices]
np.save("result/bert_embeddings_matched.npy", bert_matched)


X = np.hstack([X_hand, bert_matched])
print("final X.shape:", X.shape)


print("Loading MLP model...")
mlp = joblib.load(model_path)
print("dimension:", mlp.n_features_in_)

def predict_fn(x):
    return mlp.predict_proba(x)


# background
bert_bg = bert_matched.mean(axis=0, keepdims=True)  # (1, 768)

def predict_with_14(X14):

    n = X14.shape[0]
    X_full = np.hstack([X14, np.repeat(bert_bg, n, axis=0)])
    return mlp.predict_proba(X_full)

bg14 = matched_df[handcrafted_cols].iloc[:50].values
explainer14 = shap.KernelExplainer(predict_with_14, bg14, link="logit")
shap_values_14 = explainer14.shap_values(X_hand, nsamples=2000)

sv14 = shap_values_14[1]
if sv14.shape[0] != X_hand.shape[0]:
    sv14 = sv14.T

mean_shap = np.abs(sv14).mean(axis=0)



importance_df = pd.DataFrame({
    "Feature": handcrafted_cols,
    "Mean SHAP": mean_shap
}).sort_values("Mean SHAP", ascending=True)




csv_path = os.path.join(output_dir, "mlp_shap_importance_final120.csv")
img_path = os.path.join(output_dir, "shap_mlp_final120.png")

importance_df.to_csv(csv_path, index=False)
print(f"feature imortance CSV 保存至: {csv_path}")

plt.figure(figsize=(8, 6))
plt.barh(importance_df["Feature"], importance_df["Mean SHAP"], color="lightblue")
plt.title("MLP Classifier - SHAP Feature Importance (Final 120 Samples)")
plt.xlabel("Mean |SHAP value|")
plt.tight_layout()
plt.savefig(img_path, dpi=300)
plt.clf()
print(f"SHAP illustration is saved to the : {img_path}")


import matplotlib.ticker as mtick

plt.figure(figsize=(9, 6))
plt.barh(importance_df["Feature"], importance_df["Mean SHAP"], color="lightblue")
plt.title("MLP Classifier - SHAP Feature Importance (Final 120 Samples)")
plt.xlabel("Mean |SHAP value|")
plt.gca().xaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
plt.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
plt.tight_layout()
plt.savefig(img_path, dpi=300)



# 2) 14 个特征与预测概率的相关性（粗略 sanity check）
p1 = mlp.predict_proba(X)[:, 1]
corrs = [np.corrcoef(X_hand[:, i], p1)[0,1] for i in range(14)]
