import os
import glob
import pandas as pd
import numpy as np
from comet import download_model, load_from_checkpoint

# ==== 路径配置 ====
csv_path = r"D:\Liuyuxuan\MUC\Slp\dissertation\新建文件夹\result\tiaocanshiyunxing\comet\final.csv"
save_path = r"D:\Liuyuxuan\MUC\Slp\dissertation\新建文件夹\result\tiaocanshiyunxing\comet\final_comet.csv"
local_model_root = r"D:\Liuyuxuan\MUC\Slp\dissertation\新建文件夹\trained_models\models--Unbabel--wmt22-cometkiwi-da"

# ==== 列配置 ====
rename_to_numeric = True
src_candidates = ["col_5"]
mt_orig_col = "col_0"
mt_revised_col = "col_6"




qe_model_name = "Unbabel/wmt22-cometkiwi-da"
batch_size = 8
use_gpu = 0  # CPU

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def find_src_column(df):
    for c in src_candidates:
        if c in df.columns:
            return c
    for guess in ["col_1", "col_2"]:
        if guess in df.columns:
            return guess
    raise ValueError("未找到源文列，请检查 src_candidates。")

def resolve_local_checkpoint(root_dir):
    if not root_dir or not os.path.exists(root_dir):
        return None
    patterns = [
        os.path.join(root_dir, "**", "checkpoints", "*.ckpt"),
        os.path.join(root_dir, "**", "*.ckpt"),
    ]
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        if matches:
            return matches[0]
    return root_dir

def load_qe_model():
    ckpt = resolve_local_checkpoint(local_model_root)
    if ckpt:
        try:
            print(f"Trying to load model from local checkpoint: {ckpt}")
            return load_from_checkpoint(ckpt)
        except Exception as e:
            print(f"Local load failed, falling back to online download. Reason: {e}")
    print(f"Downloading model online: {qe_model_name}")
    mpath = download_model(qe_model_name)
    return load_from_checkpoint(mpath)


def main():

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if rename_to_numeric:
        df.columns = [f"col_{i}" for i in range(len(df.columns))]

    src_col = find_src_column(df)
    for c in [src_col, mt_orig_col, mt_revised_col]:
        if c not in df.columns:
            raise ValueError(f"找不到列：{c}")


    model = load_qe_model()


    data_orig = [{"src": s, "mt": m} for s, m in zip(df[src_col], df[mt_orig_col])]
    data_rev  = [{"src": s, "mt": m} for s, m in zip(df[src_col], df[mt_revised_col])]

    scores_orig = model.predict(data_orig, batch_size=batch_size, gpus=use_gpu).scores
    scores_rev  = model.predict(data_rev,  batch_size=batch_size, gpus=use_gpu).scores

    df["cometkiwi_original"]    = scores_orig
    df["cometkiwi_revised"]     = scores_rev
    df["cometkiwi_improvement"] = df["cometkiwi_revised"] - df["cometkiwi_original"]

    ensure_dir(save_path)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"✅ result is saved to：{save_path}")


    impr = df["cometkiwi_improvement"].dropna()
    summary = {
        "n_pairs": int(impr.shape[0]),
        "mean_delta": float(impr.mean()),
        "median_delta": float(impr.median()),
        "win_rate": float((impr > 0).mean()),
        "degrade_rate": float((impr < 0).mean()),
    }

    try:
        from scipy import stats
        t, p = stats.ttest_rel(df["cometkiwi_revised"], df["cometkiwi_original"], nan_policy="omit")
        summary["paired_t"] = float(t)
        summary["p_value"] = float(p)
    except Exception as e:
        print("⚠️ scipy is not installed, skipping t-test.", e)

    if impr.std(ddof=1) not in (0, np.nan):
        summary["cohens_d_paired"] = float(impr.mean() / impr.std(ddof=1))

    sum_path = os.path.splitext(save_path)[0] + "_summary.csv"
    pd.DataFrame([summary]).to_csv(sum_path, index=False, encoding="utf-8-sig")
    print("== COMETKiwi Improvement Statistics ==")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"📊 Summary saved to: {sum_path}")

    if __name__ == "__main__":
        main()
