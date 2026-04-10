#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Threshold sweep -> best threshold & confusion matrix (publication-quality PNG)
+ metrics: AUROC, AUPRC, F1, MCC, Balanced Accuracy

Usage (binary classification):
  python best_threshold_confmat_plus.py \
    --csv sample_probs_train_joint.csv \
    --outdir ./figs_thresh \
    --pos_class cancer \
    --label_col true_label

Optional:
  --prob_col prob_cancer        # 予測確率列を明示（既定は自動検出）
  --steps 2001                  # 閾値分割数（既定 1001）
  --bins 5                      # 閾値軸の主目盛数（既定 5）
  --thr_by acc                  # 閾値選択基準: acc|f1|mcc|balacc（既定 acc）
  --dpi 400
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV path")
    ap.add_argument("--outdir", default="./figs_thresh", help="Output directory")
    ap.add_argument("--pos_class", required=True, help="Positive class name (e.g., 'cancer')")
    ap.add_argument("--label_col", default="true_label", help="True label column")
    ap.add_argument("--prob_col", default=None, help="Probability column for POSITIVE class (auto-detect if None)")
    ap.add_argument("--steps", type=int, default=10001, help="Number of thresholds swept in [0,1]")
    ap.add_argument("--bins", type=int, default=5, help="Major ticks for threshold axis (plot)")
    ap.add_argument("--thr_by", choices=["acc","f1","mcc","balacc"], default="acc",
                    help="Criterion to pick the best threshold (default: acc)")
    ap.add_argument("--dpi", type=int, default=400, help="DPI for PNG")
    return ap.parse_args()

# ---------- Style ----------
def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })

# ---------- Utils ----------
def autodetect_prob_col(df: pd.DataFrame, pos_class: str):
    cands = [f"prob_{pos_class}", "prob", "score", "y_prob", "p"]
    for c in cands:
        if c in df.columns:
            return c
    # 'prob_*' が2本だけなら pos_class と一致する方を使う
    prob_like = [c for c in df.columns if c.lower().startswith("prob_")]
    if len(prob_like) == 2:
        for c in prob_like:
            if c.lower() == f"prob_{pos_class.lower()}":
                return c
    raise ValueError("予測確率列を自動検出できません。--prob_col で明示指定してください。")

def to_binary_labels(y_true_raw, pos_class: str):
    return np.array([1 if str(t)==str(pos_class) else 0 for t in y_true_raw], dtype=int)

def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray, n_steps: int=1001):
    thresholds = np.linspace(0.0, 1.0, n_steps)
    accs = np.zeros_like(thresholds)
    f1s  = np.zeros_like(thresholds)
    mccs = np.zeros_like(thresholds)
    bals = np.zeros_like(thresholds)
    tps = np.zeros_like(thresholds, dtype=int)
    fps = np.zeros_like(thresholds, dtype=int)
    fns = np.zeros_like(thresholds, dtype=int)
    tns = np.zeros_like(thresholds, dtype=int)

    for i, thr in enumerate(thresholds):
        y_pred = (y_prob >= thr).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        tps[i], fps[i], fns[i], tns[i] = tp, fp, fn, tn
        accs[i] = (tp + tn) / (len(y_true) if len(y_true)>0 else 1)
        # 0/0 回避のため try/except
        try:
            f1s[i]  = f1_score(y_true, y_pred, zero_division=0)
        except Exception:
            f1s[i] = 0.0
        try:
            mccs[i] = matthews_corrcoef(y_true, y_pred)
        except Exception:
            mccs[i] = 0.0
        try:
            bals[i] = balanced_accuracy_score(y_true, y_pred)
        except Exception:
            bals[i] = 0.0

    table = pd.DataFrame({
        "threshold": thresholds,
        "acc": accs,
        "f1": f1s,
        "mcc": mccs,
        "balacc": bals,
        "TP": tps, "FP": fps, "FN": fns, "TN": tns,
    })
    return table

def pick_best_row(table: pd.DataFrame, criterion: str):
    # 同値なら閾値の小さい方（再現性重視）
    idx = int(table[criterion].values.argmax())
    return table.iloc[idx]

def plot_results(table: pd.DataFrame, criterion: str,
                 auroc: float, auprc: float,
                 pos_class: str, out_png: str, bins: int, dpi: int):
    set_style()
    # ベスト行
    best = pick_best_row(table, criterion)
    best_thr = float(best["threshold"])
    best_acc = float(best["acc"])
    cm = dict(TP=int(best["TP"]), FP=int(best["FP"]), FN=int(best["FN"]), TN=int(best["TN"]))
    f1   = float(best["f1"])
    mcc  = float(best["mcc"])
    bala = float(best["balacc"])

    fig = plt.figure(figsize=(10.5, 4.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.28)

    # ---- 左：Metric vs Threshold（主曲線は基準メトリクス、薄線でACCも表示）
    ax1 = fig.add_subplot(gs[0,0])
    x = table["threshold"].values
    # 主曲線
    main_y = table[criterion].values
    ax1.plot(x, main_y, lw=2.2, color="#2ca02c", label=criterion.upper())
    # 参考線：ACC（基準がACCでないとき）
    if criterion != "acc":
        ax1.plot(x, table["acc"].values, lw=1.4, color="#1f77b4", alpha=0.7, label="ACC")
    # ベスト点
    ax1.scatter([best_thr], [best[criterion]], color="#d62728", s=42, zorder=3)
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel(criterion.upper())
    ax1.set_title(f"{criterion.upper()} vs. Threshold")
    ax1.set_xlim(0,1)
    ax1.set_ylim(0,1)
    ax1.set_xticks(np.linspace(0,1,bins))
    ax1.set_yticks(np.linspace(0,1,6))
    ax1.legend(loc="lower center")

    # サブ注記（AUROC/AUPRC/ベスト閾値）
    text = (f"AUROC  = {auroc:.3f}\n"
            f"AUPRC  = {auprc:.3f}\n"
            f"best thr({criterion}) = {best_thr:.3f}\n"
            f"ACC={best_acc:.3f} | F1={f1:.3f}\n"
            f"MCC={mcc:.3f} | BalAcc={bala:.3f}")
    ax1.text(0.02, 0.98, text, transform=ax1.transAxes,
             va="top", ha="left", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", lw=0.6))

    # ---- 右：混同行列
    ax2 = fig.add_subplot(gs[0,1])
    TP, FP, FN, TN = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
    mat = np.array([[TP, FN],
                    [FP, TN]], dtype=float)  # 上段：True POS, 下段：True NEG
    tot = mat.sum()
    mat_norm = mat / tot if tot>0 else mat
    im = ax2.imshow(mat, cmap="Blues", vmin=0, vmax=max(1, mat.max()))
    for spine in ax2.spines.values():
        spine.set_visible(True); spine.set_linewidth(0.8)
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f"{int(mat[i,j])}\n({mat_norm[i,j]*100:.1f}%)",
                     ha="center", va="center",
                     color=("white" if mat_norm[i,j] > 0.6 else "black"),
                     fontsize=10)
    ax2.set_xticks([0,1]); ax2.set_yticks([0,1])
    ax2.set_xticklabels([f"Pred {pos_class}", f"Pred not-{pos_class}"])
    ax2.set_yticklabels([f"True {pos_class}", f"True not-{pos_class}"])
    ax2.set_title(f"Confusion Matrix @ thr={best_thr:.3f}")

    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Count")

    fig.suptitle("Best Threshold & Confusion Matrix (with AUROC/AUPRC)", y=1.02, fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)

    return best  # 返してテキスト出力に使う

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 読み込み
    df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]
    cmap = {c.lower(): c for c in df.columns}

    # ラベル列
    if args.label_col not in df.columns:
        lc = args.label_col.lower()
        if lc in cmap:
            args.label_col = cmap[lc]
        else:
            raise ValueError(f"Label column '{args.label_col}' not found. Found: {df.columns.tolist()}")

    # 確率列
    prob_col = args.prob_col
    if prob_col is None:
        prob_col = autodetect_prob_col(df, args.pos_class)
    elif prob_col not in df.columns:
        lc = prob_col.lower()
        if lc in cmap:
            prob_col = cmap[lc]
        else:
            raise ValueError(f"Probability column '{args.prob_col}' not found. Found: {df.columns.tolist()}")

    # ベクトル化
    y_true = to_binary_labels(df[args.label_col].values, args.pos_class)
    y_prob = pd.to_numeric(df[prob_col], errors="coerce").fillna(0.0).clip(0,1).values

    # AUROC / AUPRC（閾値に依存しない）
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    # 閾値掃引（ACC/F1/MCC/BalAcc & CM）
    table = sweep_thresholds(y_true, y_prob, n_steps=args.steps)
    # 全閾値の表も保存（解析用）
    table.to_csv(os.path.join(args.outdir, "metrics_at_thresholds.csv"), index=False)

    # 選択基準でベスト閾値を決定
    best = pick_best_row(table, args.thr_by)

    # 図を出力（左：基準メトリクス曲線＋注記、右：CM）
    out_png = os.path.join(args.outdir, "best_threshold_confmat.png")
    best = plot_results(table, args.thr_by, auroc, auprc, args.pos_class, out_png, args.bins, args.dpi)

    # テキストまとめ
    with open(os.path.join(args.outdir, "best_threshold_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"pos_class: {args.pos_class}\n")
        f.write(f"label_col: {args.label_col}\n")
        f.write(f"prob_col : {prob_col}\n")
        f.write(f"threshold_selection: {args.thr_by}\n")
        f.write("\n--- THRESHOLD-INDEPENDENT ---\n")
        f.write(f"AUROC : {auroc:.6f}\n")
        f.write(f"AUPRC : {auprc:.6f}\n")
        f.write("\n--- AT BEST THRESHOLD ---\n")
        f.write(f"best_threshold: {float(best['threshold']):.6f}\n")
        f.write(f"ACC : {float(best['acc']):.6f}\n")
        f.write(f"F1  : {float(best['f1']):.6f}\n")
        f.write(f"MCC : {float(best['mcc']):.6f}\n")
        f.write(f"BalAcc: {float(best['balacc']):.6f}\n")
        f.write(f"TP={int(best['TP'])}  FP={int(best['FP'])}  FN={int(best['FN'])}  TN={int(best['TN'])}\n")

    print(f"[OK] AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
    print(f"[OK] Best thr by {args.thr_by}: {float(best['threshold']):.4f} | ACC={float(best['acc']):.4f} | F1={float(best['f1']):.4f} | MCC={float(best['mcc']):.4f} | BalAcc={float(best['balacc']):.4f}")
    print(f"[SAVED] {out_png}")
    print(f"[SAVED] metrics_at_thresholds.csv / best_threshold_metrics.txt")

if __name__ == "__main__":
    main()
