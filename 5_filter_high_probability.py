# Reconstruct the downsampled df_balanced (1:5 ratio)
# Load the BERT embeddings saved during training (bert_embeddings.npy)
# Combine 14 handcrafted features with BERT features (782 dimensions)
# Use the trained MLP model to make predictions on the test set
# Filter sentences with high translationese probability (prob > 0.7)
# Save as translationese_test_subset.csv for post-editing tasks

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ========================
# Step 1: Load full data
# ========================
df = pd.read_csv("D:\Liuyuxuan\MUC\Slp\dissertation\新建文件夹\\result\\2. feature_dataset.csv")
df["row_id"] = range(len(df))

# ========================
# Step 2: Rebuild df_balanced
# ========================
df_tec = df[df["source"] == "TEC"]
df_nec = df[df["source"] == "NEC"]
nec_sample = df_nec.sample(n=len(df_tec)*5, random_state=42)
df_balanced = pd.concat([df_tec, nec_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
df_balanced["row_id"] = range(len(df_balanced))

# ========================
# Step 3: Prepare features
# ========================
handcrafted_cols = [
    "sent_length", "clause_depth", "passive_ratio", "subordination_ratio", "nominalization_ratio",
    "ttr", "lexical_density", "repetition_frequency", "avg_word_length", "func_content_ratio",
    "connective_count", "cohesive_marker_ratio", "pronoun_ratio", "punctuation_density"
]

X_hand = df_balanced[handcrafted_cols].values
X_bert = np.load("result/bert_embeddings_downsample.npy")
X = np.hstack([X_hand, X_bert])
y = (df_balanced["source"] == "TEC").astype(int).values

# ========================
# Step 4: Split with row tracking
# ========================
X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
    X, y, df_balanced["row_id"], test_size=0.30, stratify=y, random_state=42
)
X_dev, X_test, y_dev, y_test, idx_dev, idx_test = train_test_split(
    X_temp, y_temp, idx_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# ========================
# Step 5: Standardize handcrafted features
# ========================
scaler = StandardScaler()
X_train[:, :14] = scaler.fit_transform(X_train[:, :14])
X_test[:, :14] = scaler.transform(X_test[:, :14])

# ========================
# Step 6: Load model & predict
# ========================
model = joblib.load("D:\Liuyuxuan\MUC\Slp\dissertation\新建文件夹\\trained_models\mlp.pkl")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ========================
# Step 7: Restore test sentences
# ========================
df_test = df_balanced.loc[idx_test].copy()
df_test["pred_prob"] = y_prob
df_test["pred_label"] = y_pred

# ========================
# Step 8: Filter high translationese
# ========================
df_translationese = df_test[df_test["pred_prob"] > 0.8]
df_translationese = df_test[
    (df_test["pred_prob"] > 0.8) &
    (df_test["source"] == "TEC")
]
# ========================
# Step 9: Save result
# ========================
output_path = "result/translationese_test_subset.csv"
df_translationese.to_csv(output_path, index=False)
print(f"✅ Post-editing candidate sentences saved to: {output_path}")

