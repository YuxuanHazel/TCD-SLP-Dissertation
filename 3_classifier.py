x# ===========================================
# Code A: Classifiers (Logistic Regression / RF / MLP)
# NEC
# ===========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import joblib
import os

# ==========================
# 1.
# ==========================
df = pd.read_csv("result/feature_dataset.csv")

df_tec = df[df["source"] == "TEC"]
df_nec = df[df["source"] == "NEC"]

print(f"原始数据：TEC={len(df_tec)}, NEC={len(df_nec)}")

# ==========================
# 2. NEC downsample(1:5)
# ==========================
nec_sample = df_nec.sample(n=len(df_tec)*5, random_state=42)
df_balanced = pd.concat([df_tec, nec_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"采样后数据：{df_balanced['source'].value_counts().to_dict()}")

# ==========================
# 3. data
# ==========================
handcrafted_cols = [
    "sent_length","clause_depth","passive_ratio","subordination_ratio","nominalization_ratio",
    "ttr","lexical_density","repetition_frequency","avg_word_length","func_content_ratio",
    "connective_count","cohesive_marker_ratio","pronoun_ratio","punctuation_density"
]

X_hand = df_balanced[handcrafted_cols].values
y = (df_balanced["source"] == "TEC").astype(int).values
sentences = df_balanced["sentence"].tolist()

# ==========================
# 4. BERT embeddings (CLS)
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_path = "/root/autodl-tmp/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"

tokenizer = BertTokenizer.from_pretrained(model_path)
bert = BertModel.from_pretrained(model_path)
bert.to(device)
bert.eval()


def get_bert_embeddings(sentences, batch_size=32):
    all_embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="提取 BERT 向量"):
        batch = sentences[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = bert(**enc)
        cls_vecs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_vecs)
    return np.vstack(all_embeddings)

bert_cache_file = "result/bert_embeddings_downsample.npy"

if os.path.exists(bert_cache_file):
    # ✅ Detected cache file, load BERT embeddings directly...
    print(f"✅ Detected cache file {bert_cache_file}, loading BERT embeddings...")
    X_bert = np.load(bert_cache_file)
else:
    # 🚀 No cache file found, start extracting BERT embeddings...
    print("🚀 No cache file found, extracting BERT embeddings...")
    X_bert = get_bert_embeddings(sentences)
    np.save(bert_cache_file, X_bert)
    # ✅ BERT embeddings saved
    print(f"✅ BERT embeddings have been saved to {bert_cache_file}")


# combined feature (14 dimensions handcrafted + 768d BERT)
X = np.hstack([X_hand, X_bert])

# ==========================
# 5. dataset (70/15/15)
# ==========================
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

print(f"Train={len(y_train)}, Dev={len(y_dev)}, Test={len(y_test)}")

# ==========================
# 6. standarization
# ==========================
scaler = StandardScaler()
X_train[:, :len(handcrafted_cols)] = scaler.fit_transform(X_train[:, :len(handcrafted_cols)])
X_dev[:, :len(handcrafted_cols)] = scaler.transform(X_dev[:, :len(handcrafted_cols)])
X_test[:, :len(handcrafted_cols)] = scaler.transform(X_test[:, :len(handcrafted_cols)])

# ==========================
# 7. evaluate
# ==========================
def evaluate_model(name, model, X, y):
    y_pred = model.predict(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred)
    }
# ==========================
# 8. training
# ==========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    dev_scores = evaluate_model(name, model, X_dev, y_dev)
    test_scores = evaluate_model(name, model, X_test, y_test)
    
    print(f"\n=== {name} ===")
    print(f"Dev: {dev_scores}")
    print(f"Test: {test_scores}")
    
    results.append(f"=== {name} ===")
    results.append(f"Dev: {dev_scores}")
    results.append(f"Test: {test_scores}")
    results.append("")

with open("result/classifier_results.txt", "w", encoding="utf-8") as f:
    for line in results:
        f.write(line + "\n")



# ==========================
# 9. 保存模型和结果
# ==========================
joblib.dump(models["Logistic Regression"], "result/logistic_regression.pkl")
joblib.dump(models["Random Forest"], "result/random_forest.pkl")
joblib.dump(models["MLP"], "result/mlp.pkl")

with open("result/classifier_results.txt", "w", encoding="utf-8") as f:
    for line in results:
        f.write(line + "\n")

print("\n✅ All evaluation results and model files have been saved to the result folder")


