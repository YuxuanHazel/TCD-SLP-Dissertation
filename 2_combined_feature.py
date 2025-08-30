# ===========================================
# Linguistic Feature Extraction Script


# ===== 1. dependence=====
# !pip install spacy matplotlib seaborn scipy pandas tqdm -q
# !python -m spacy download en_core_web_sm

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pandas as pd
import numpy as np
import spacy
from collections import Counter
from tqdm import tqdm
from scipy.stats import ttest_ind

# ==========================
# 1.
# ==========================
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 3000000

# ==========================
# 2.
# ==========================
connectives = {"however", "therefore", "moreover", "because", "although", "but"}
cohesive_markers = {"and", "but", "so", "or"}

# ==========================
# 3.
# ==========================
def extract_features(sentence):
    doc = nlp(sentence)
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    content_words = [t for t in doc if t.pos_ in ["NOUN","VERB","ADJ","ADV"]]
    function_words = [t for t in doc if t.pos_ in ["PRON","DET","ADP","AUX","CCONJ","SCONJ"]]

    # ---- syntactic features ----
    sent_length = len(tokens)
    clause_depth = max([len(list(t.ancestors)) for t in doc]) if tokens else 0
    verbs = [t for t in doc if t.pos_ == "VERB"]
    passive_verbs = [t for t in verbs if "pass" in t.dep_]
    passive_ratio = len(passive_verbs) / len(verbs) if verbs else 0
    sub_clauses = [t for t in doc if t.dep_ in ["mark","advcl","ccomp"]]
    all_clauses = [t for t in doc if t.dep_ in ["ROOT","ccomp","advcl","xcomp"]]
    subordination_ratio = len(sub_clauses) / len(all_clauses) if all_clauses else 0
    nominals = [t for t in doc if t.pos_ == "NOUN" and t.text.endswith(("tion","ment","ness","ity"))]
    nouns = [t for t in doc if t.pos_ == "NOUN"]
    nominalization_ratio = len(nominals) / len(nouns) if nouns else 0

    # ---- lexical features ----
    ttr = len(set(tokens)) / len(tokens) if tokens else 0
    lexical_density = len(content_words) / len(tokens) if tokens else 0
    counter = Counter(tokens)
    repetition_frequency = sum(c for c in counter.values() if c > 1) / len(tokens) if tokens else 0
    avg_word_length = np.mean([len(t) for t in tokens]) if tokens else 0
    func_content_ratio = len(function_words) / len(content_words) if content_words else 0

    # ---- structural features ----
    connective_count = sum(1 for t in tokens if t in connectives)
    cohesive_marker_ratio = sum(1 for t in tokens if t in cohesive_markers) / len(tokens) if tokens else 0
    pronoun_ratio = len([t for t in doc if t.pos_ == "PRON"]) / len(tokens) if tokens else 0
    punctuation_density = len([t for t in doc if t.is_punct]) / len(doc) if len(doc) else 0

    return [sent_length, clause_depth, passive_ratio, subordination_ratio, nominalization_ratio,
            ttr, lexical_density, repetition_frequency, avg_word_length, func_content_ratio,
            connective_count, cohesive_marker_ratio, pronoun_ratio, punctuation_density]

feature_names = [
    "sent_length","clause_depth","passive_ratio","subordination_ratio","nominalization_ratio",
    "ttr","lexical_density","repetition_frequency","avg_word_length","func_content_ratio",
    "connective_count","cohesive_marker_ratio","pronoun_ratio","punctuation_density"
]

# ==========================
# 4.
# ==========================
def split_into_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

# ==========================
# 5.
# ==========================
def load_tec_txt(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    sentences = split_into_sentences(text)
    return pd.DataFrame({"sentence": sentences, "source": "TEC"})

def load_nec_folder(folder_path):
    all_sentences = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                sentences = split_into_sentences(text)
                all_sentences.extend(sentences)
    return pd.DataFrame({"sentence": all_sentences, "source": "NEC"})

# ==========================
# 6.
# ==========================
tec_path = "TEC.txt"
nec_folder = "NEC"
output_folder = "result"

os.makedirs(output_folder, exist_ok=True)

# ==========================
# 7.
# ==========================
tec_df = load_tec_txt(tec_path)
nec_df = load_nec_folder(nec_folder)
df = pd.concat([tec_df, nec_df]).reset_index(drop=True)

print(f"✅ TEC sentences: {len(tec_df)}, NEC sentences: {len(nec_df)}")

# ==========================
# 8. 提取特征
# ==========================
all_features = []
for sent in tqdm(df["sentence"], desc="Extracting features"):
    all_features.append(extract_features(sent))

features_df = pd.DataFrame(all_features, columns=feature_names)
df = pd.concat([df, features_df], axis=1)

# ==========================
# 9.
# ==========================
desc_stats = df.groupby("source")[feature_names].agg(["mean","std"]).round(3)
ttest_results = {}
for feature in feature_names:
    tec_values = df[df["source"]=="TEC"][feature]
    nec_values = df[df["source"]=="NEC"][feature]
    t_stat, p_val = ttest_ind(tec_values, nec_values, equal_var=False)
    ttest_results[feature] = p_val
ttest_df = pd.DataFrame.from_dict(ttest_results, orient="index", columns=["p-value"]).round(4)

# ==========================
# 10.
# ==========================
feature_csv_path = os.path.join(output_folder, "feature_dataset.csv")
txt_result_path = os.path.join(output_folder, "linguistic_analysis_results.txt")

df.to_csv(feature_csv_path, index=False)
with open(txt_result_path, "w", encoding="utf-8") as f:
    f.write("=== Linguistic Feature Descriptive Statistics ===\n\n")
    f.write(desc_stats.to_string())
    f.write("\n\n=== T-test Results (p-values) ===\n\n")
    f.write(ttest_df.to_string())
    f.write("\n\nNote: p < 0.05 indicates significant difference between TEC and NEC.\n")

print(f"✅ feature_dataset.csv is done: {feature_csv_path}")
print(f"✅ statisics and t-test result is saved as : {txt_result_path}")
