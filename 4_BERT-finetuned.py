# ===========================================
# Fine-tuned BERT for TEC vs NEC 分类
# 离线运行 / 下采样 / 三分法划分
# ===========================================

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from tqdm import tqdm

# ==========================
# 0. devicd and model
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/root/autodl-tmp/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"

# ==========================
# 1. data loading
# ==========================
df = pd.read_csv("result/feature_dataset.csv")

df_tec = df[df["source"] == "TEC"]
df_nec = df[df["source"] == "NEC"]

nec_sample = df_nec.sample(n=len(df_tec)*5, random_state=42)
df_balanced = pd.concat([df_tec, nec_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"采样后：{df_balanced['source'].value_counts().to_dict()}")

sentences = df_balanced["sentence"].tolist()
labels = (df_balanced["source"] == "TEC").astype(int).values

# ==========================
# 2. 三分法
# ==========================
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    sentences, labels, test_size=0.30, stratify=labels, random_state=42
)
dev_texts, test_texts, dev_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42
)

# ==========================
# 3. Tokenizer & Dataset
# ==========================
tokenizer = BertTokenizer.from_pretrained(model_path)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

batch_size = 16
train_dataset = TextDataset(train_texts, train_labels, tokenizer)
dev_dataset   = TextDataset(dev_texts, dev_labels, tokenizer)
test_dataset  = TextDataset(test_texts, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader   = DataLoader(dev_dataset, batch_size=batch_size)
test_loader  = DataLoader(test_dataset, batch_size=batch_size)

# ==========================
# 4. Fine-tuned BERT
# ==========================
class BertClassifier(nn.Module):
    def __init__(self, bert_model):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 向量
        x = self.dropout(cls_output)
        return self.classifier(x)

bert = BertModel.from_pretrained(model_path)
model = BertClassifier(bert).to(device)

# ==========================
# 5. optimizer & Loss & Scheduler
# ==========================
epochs = 3
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()


total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# ==========================
# 6.
# ==========================
def train_epoch(model, dataloader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs, dim=1)
            preds.extend(pred.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return {
        "Accuracy": accuracy_score(true_labels, preds),
        "Precision": precision_score(true_labels, preds),
        "Recall": recall_score(true_labels, preds),
        "F1": f1_score(true_labels, preds)
    }

# ==========================
# 7.
# ==========================
results = []

for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)
    dev_scores = evaluate(model, dev_loader)

    log_line = f"Epoch {epoch + 1}/{epochs} | Train Loss={train_loss:.4f} | Dev={dev_scores}"
    print("\n" + log_line)
    results.append(log_line)

# ==========================
# 8.
# ==========================
test_scores = evaluate(model, test_loader)
final_line = f"\n✅ Final Test Scores: {test_scores}"
print(final_line)
results.append(final_line)

# 保存训练日志
os.makedirs("result", exist_ok=True)
with open("result/fine_tuned_bert_results.txt", "w", encoding="utf-8") as f:
    for line in results:
        f.write(line + "\n")

print("✅ Training log has been saved to result/fine_tuned_bert_results.txt")


# ==========================
# 9. save model & Tokenizer
# ==========================
torch.save(model.state_dict(), "result/finetuned_bert_translationese.pt")
torch.save(model, "result/finetuned_bert_translationese_full.pt")
tokenizer.save_pretrained("result/finetuned_bert_tokenizer")

print("✅ Model weights, full model, and tokenizer have been saved")

