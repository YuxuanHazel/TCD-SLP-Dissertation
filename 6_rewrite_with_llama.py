import pandas as pd
import torch
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

# ========================
# 参数配置
# ========================
BATCH_SIZE = 10
INPUT_CSV = "/root/run/retrive_src_from_lime.csv"
TEMP_CSV = "/root/run/result/revised_sentences_llama_temp_50_t=3.csv"
FINAL_CSV = "/root/run/result/revised_sentences_llama_50_t=3.csv"
MODEL_PATH = "/root/autodl-fs/llama-2-7b-chat-hf"

MAX_RETRY = 2
TEMPERATURE = 0.3
MAX_LEN_FACTOR = 1.3

# ========================

# ========================
def load_llama():
    print("Loading……")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


llama = load_llama()

# ========================

# ========================
df = pd.read_csv(INPUT_CSV)
if "revised_sentence" not in df.columns:
    df["revised_sentence"] = ""

done_mask = df["revised_sentence"].notna() & df["revised_sentence"].astype(str).str.strip().ne("")
start_index = done_mask.sum()
total = len(df)

print(f" Total {total}, processed {start_index}, remaining {total - start_index}")


# ========================

# ========================
def generate_revised(prompt, orig_sentence):
    for attempt in range(MAX_RETRY + 1):
        try:
            max_new_tokens = max(10, int(len(orig_sentence.split()) * MAX_LEN_FACTOR))
            # Chat 格式 prompt
            chat_prompt = f"<s>[INST] Rewrite the following sentence to make it read more like native English:\n{orig_sentence}\n[/INST]"
            response = llama(chat_prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=TEMPERATURE, top_p=0.9)[0][
                "generated_text"]#true or false

            # Remove possible instruction part

            if "[/INST]" in response:
                revised = response.split("[/INST]")[-1].strip()
            else:
                revised = response.strip()

            # Simple quality check: length and whether the full original sentence is included
            if len(revised) < 3 or revised.lower() == orig_sentence.lower():
                print(f"Retry {attempt + 1}: generated content looks suspicious")

                continue

            return revised
        except Exception as e:
            print(f"Generation error (retry {attempt + 1}): {e}")

            time.sleep(1)
    return ""


# ========================

# ========================
for i in range(start_index, total, BATCH_SIZE):
    end = min(i + BATCH_SIZE, total)
    print(f"\n processing {i + 1} to {end} ……")

    for j in range(i, end):
        orig_sentence = df.at[j, "prompt"]
        revised = generate_revised(df.at[j, "prompt"], orig_sentence)
        df.at[j, "revised_sentence"] = revised
        print(f"[{j + 1}/{total}] done")

        time.sleep(0.1)


    df.to_csv(TEMP_CSV, index=False)
    print(f"Intermediate results saved to: {TEMP_CSV}")

    if i > 0 and i % 20 == 0:
        del llama
        torch.cuda.empty_cache()
        llama = load_llama()

# ========================

# ========================
df.to_csv(FINAL_CSV, index=False)
print(f"\n all result is saved to the ：{FINAL_CSV}")
