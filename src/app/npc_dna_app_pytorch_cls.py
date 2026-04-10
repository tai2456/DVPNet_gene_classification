# npc_dna_app_pytorch_cls.py
import os
import numpy as np
import streamlit as st
import torch
from datetime import datetime
import pandas as pd  


from npc_dna_core_pytorch_cls import (
    NPC_VECTORS_MC,
    build_dataloaders_vectors_multiclass,
    fit_multiclass_vectors,
    evaluate_multiclass_vectors,
    save_checkpoint_vectors,
    load_checkpoint_vectors,
)

st.set_page_config(page_title="NPC – Folder-of-vectors (.npy) per SAMPLE + RAT-SPN", layout="wide")
st.title("NPC — 1 sample = 1 folder (vectors), class-conditional PCs (RAT‑SPN)")

# =============================================================================
# Sidebar: config
# =============================================================================
st.sidebar.header("1) Data folders (SAMPLE = a folder that contains many .npy)")
train_root = st.sidebar.text_input("Train root", "./vec_data/train", key="sd_train_root")
val_root   = st.sidebar.text_input("Validation root (optional)", "", key="sd_val_root")
batch_size = st.sidebar.number_input("Batch size (list of sample-folders per iter)", 1, 64, 8, 1, key="sd_batch")
st.sidebar.header("2) PCs / Training")
pc_depth   = st.sidebar.number_input("PC depth", 1, 12, 5, 1, key="sd_pc_depth")
pc_latents = st.sidebar.number_input("PC latents (L)", 4, 1024, 64, 4, key="sd_pc_latents")
pc_reps    = st.sidebar.number_input("PC repetitions (R)", 1, 16, 3, 1, key="sd_pc_reps")
pc_pieces  = st.sidebar.number_input("Leaf pieces (P)", 1, 8, 2, 1, key="sd_pc_pieces")

ce_lr      = st.sidebar.number_input("CE LR", 1e-6, 1.0, 2e-3, format="%.6f", key="sd_ce_lr")
epochs     = st.sidebar.number_input("Epochs (core will loop)", 1, 2000, 10, 1, key="sd_epochs")
log_every  = st.sidebar.number_input("Log every N batches (train)", 1, 100000, 1000, 1, key="sd_log_every")

st.sidebar.header("3) Checkpoints")
ckpt_dir   = st.sidebar.text_input("Checkpoint directory", "./checkpoints/vec_cls", key="sd_ckpt_dir")
resume_from= st.sidebar.text_input("Resume from (path to .pt; optional)", "", key="sd_resume_from")

CONST_Q = 1
CONST_K = 1  # model hparam only（実際の K は入力から動的）

def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)
    st.write(f"`{ts}` {msg}")

# =============================================================================
# Step A) Build dataloaders
# =============================================================================
st.header("Step A) Build dataloaders")
if st.button("Scan dataset & build", key="btn_scan_build"):
    try:
        _log(f"Scanning dataset… train={train_root}, val={val_root or '(none)'}")
        dl_tr, dl_va, spec = build_dataloaders_vectors_multiclass(
            train_root=train_root.strip(),
            val_root=val_root.strip() or None,
            batch_size=int(batch_size),      # list of sample-folders per iter（更新はサンプル単位）
            num_workers=0,
        )
        st.session_state.dl = (dl_tr, dl_va)
        st.session_state.spec = spec
        _log(f"Spec: classes={spec['class_names']} (C={spec['num_classes']}), D={spec['D']} (K varies per sample)")
        st.success(f"Classes = {spec['class_names']} (C={spec['num_classes']}), D={spec['D']}")
    except Exception as e:
        _log(f"[ERROR] Scan/build failed: {e}")
        st.error(f"Failed: {e}")

# =============================================================================
# Step B) Initialize model
# =============================================================================
st.header("Step B) Initialize model")
if st.button("Init model", key="btn_init_model"):
    spec = st.session_state.get("spec")
    if not spec:
        st.warning("Scan dataset first.")
    else:
        try:
            D = int(spec["D"])
            _log(f"Initializing model with cfg K={CONST_K}, q={CONST_Q}, D={D} ... (runtime K is dynamic per sample)")
            model = NPC_VECTORS_MC(
                num_classes=int(spec["num_classes"]),
                K=CONST_K, q=CONST_Q, d=D,
                mixture_M=4,
                learn_class_prior=True,
                pc_depth_patch=int(pc_depth),
                pc_latents_patch=int(pc_latents),
                pc_repetitions_patch=int(pc_reps),
                pc_pieces_patch=int(pc_pieces),
            )
            st.session_state.model = model

            dev = next(model.parameters()).device
            nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
            _log(f"Model ready on {dev}. Trainable params={nparams:,}")
            st.success(f"Model ready. C={model.C}, D={D} (q=1, K variable per sample)")
        except Exception as e:
            _log(f"[ERROR] Init failed: {e}")
            st.error(f"Init failed: {e}")

# =============================================================================
# Step C) Train (core loops epochs; app calls fit() ONCE)
# =============================================================================
st.header("Step C) Train")
if st.button("Train now", key="btn_train_now"):
    model = st.session_state.get("model")
    dl_tr, dl_va = st.session_state.get("dl", (None, None))
    spec = st.session_state.get("spec")
    if not model or not dl_tr or not spec:
        st.warning("Please scan dataset and init model first.")
    else:
        def log_fn(m: str):
            _log(m)
        try:
            os.makedirs(ckpt_dir or "./checkpoints", exist_ok=True)
            resume_path = resume_from.strip() or None
            _log(f"Training start: epochs={int(epochs)} (core-controlled), ce_lr={float(ce_lr)}, ckpt_dir={ckpt_dir}, resume={resume_path or '(none)'}")
            _log(f"  train samples = {len(dl_tr.dataset.samples):,}")
            if dl_va: _log(f"  val   samples = {len(dl_va.dataset.samples):,}")

            # IMPORTANT: call fit() ONCE; core handles the epoch loop
            fit_multiclass_vectors(
                model=model,
                dl_tr=dl_tr,
                dl_va=dl_va,
                epochs=int(epochs),
                ce_lr=float(ce_lr),
                checkpoint_dir=ckpt_dir,
                save_every=5,
                resume_from=resume_path,
                log_every=int(log_every),
                log_fn=log_fn,
            )
            _log("Training finished successfully.")
            st.success("Training done.")
        except Exception as e:
            _log(f"[ERROR] Training failed: {e}")
            st.error(f"Training failed: {e}")

# =============================================================================
# Manual Save / Load (optional)
# =============================================================================
c1, c2 = st.columns(2)
with c1:
    if st.button("Manual snapshot save (model+opt=NULL)", key="btn_manual_save"):
        model = st.session_state.get("model")
        dl_tr, _ = st.session_state.get("dl", (None, None))
        if model is None or dl_tr is None:
            st.warning("Init model and build dataloaders first.")
        else:
            try:
                path = os.path.join(ckpt_dir or ".", "manual.pt")
                _log(f"Manual save to {path}")
                save_checkpoint_vectors(model, optimizer=None, epoch=0, best_acc=0.0,
                                        path=path,
                                        class_names=getattr(dl_tr.dataset, "classes", None))
                st.success(f"Saved → {path}")
            except Exception as e:
                _log(f"[ERROR] Save failed: {e}")
                st.error(f"Save failed: {e}")

with c2:
    if st.button("Load checkpoint", key="btn_load_ckpt"):
        try:
            path = resume_from.strip()
            if not path:
                st.warning("Set 'Resume from' path first.")
            else:
                _log(f"Loading checkpoint from {path}")
                model_loaded, meta = load_checkpoint_vectors(path)
                st.session_state.model = model_loaded
                class_names = meta.get("class_names", None)
                if class_names:
                    st.session_state.spec = {
                        "class_names": class_names,
                        "num_classes": len(class_names),
                        "D": int(meta["hparams"]["D"]),
                        "K": int(meta["hparams"]["K"]),
                        "q": int(meta["hparams"]["q"]),
                    }
                _log(f"Loaded ckpt: epoch={meta.get('epoch','?')}, best_acc={meta.get('best_acc','?')}")
                st.success(f"Loaded: {os.path.basename(path)} (epoch={meta.get('epoch','?')}, best_acc={meta.get('best_acc','?')})")
        except Exception as e:
            _log(f"[ERROR] Load failed: {e}")
            st.error(f"Load failed: {e}")

# =============================================================================
# Step D) Inference (one SAMPLE = one folder of .npy)
# =============================================================================
st.header("Step D) Inference (one SAMPLE folder)")
sample_dir = st.text_input("Path to a sample folder (contains many .npy vectors)", "", key="ti_sample_dir")
topk_show  = st.number_input("Show top-K genes by posterior (pred class)", 1, 200, 20, 1, key="sd_topk")

def _vec_from_any(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 2:
        return arr.mean(axis=0, dtype=np.float32)
    if arr.ndim == 3:
        return arr.mean(axis=(0,1), dtype=np.float32)
    raise RuntimeError(f"Unsupported ndim={arr.ndim}")

if st.button("Predict sample folder", key="btn_predict_sample"):
    model = st.session_state.get("model")
    spec  = st.session_state.get("spec")
    if not model or not spec:
        st.warning("Init model first.")
    else:
        try:
            if not sample_dir or not os.path.isdir(sample_dir):
                st.warning("Please input a valid sample folder path that contains .npy files.")
                st.stop()

            _log(f"Prediction start: sample_dir={sample_dir}")
            files = sorted([f for f in os.listdir(sample_dir) if f.lower().endswith(".npy")])
            if not files:
                raise RuntimeError("No .npy files found in the folder.")
            vecs = []
            gene_names = []
            for fn in files:
                try:
                    arr = np.load(os.path.join(sample_dir, fn), allow_pickle=False)
                    vecs.append(_vec_from_any(np.asarray(arr)))
                    gene_names.append(os.path.splitext(fn)[0])
                except Exception:
                    continue
            if not vecs:
                raise RuntimeError("No valid vectors to infer.")

            V = np.stack(vecs, axis=0)  # (K,D)
            D = int(spec["D"])
            if V.shape[1] != D:
                raise RuntimeError(f"D mismatch: found D={V.shape[1]} in sample, model D={D}")

            # x:(1,K,1,D), m:(1,K)
            device = next(model.parameters()).device
            x = torch.from_numpy(V.astype(np.float32)).view(1, V.shape[0], 1, D).to(device)
            m = torch.ones((1, V.shape[0]), dtype=torch.float32, device=device)

            _log("  running model.predict_from_feats() ...")
            with torch.no_grad():
                pred, prob, _ = model.predict_from_feats(x, m)
            classes = spec.get("class_names", [str(i) for i in range(model.C)])
            p = prob[0].detach().cpu().numpy().tolist()
            _log(f"  sample-level prob={p}")
            st.write("**Sample-level probabilities:** " + ", ".join([f"{classes[i]}={p[i]:.3f}" for i in range(len(classes))]))
            st.write(f"**Predicted class:** {classes[int(pred[0].item())]}")

            # per-gene posterior (keep-only) for predicted class
            with torch.no_grad():
                scores_log, _t_idx = model.per_bin_keep_only_posteriors_from_feats(x, m, target="pred")  # (1,K)
            scores = scores_log[0].detach().cpu().numpy()  # log posterior per gene vector
            order = np.argsort(-scores)[:int(topk_show)]
            st.subheader("Top genes by posterior (predicted class)")
            for rank, idx in enumerate(order, 1):
                st.write(f"{rank:2d}. {gene_names[idx]}  (log-post={scores[idx]:.4f})")

            _log("Prediction done.")
        except Exception as e:
            _log(f"[ERROR] Inference failed: {e}")
            st.error(f"Inference failed: {e}")
# =============================================================================
# Step E) Batch evaluate a directory (same structure as train) and export CSV
# =============================================================================
st.header("Step E) Batch evaluate & export CSV")

eval_root = st.text_input("Eval root (same class/sample folder structure as train)", "", key="sd_eval_root")
topk_gene_export = st.number_input("Top-K genes per sample (predicted & other-max)", 1, 200, 20, 1, key="sd_topk_export")
export_dir = st.text_input("Export directory (CSV will be written here)", "./exports", key="sd_export_dir")
csv_a_name = st.text_input("CSV A filename (sample-level probs)", "sample_probs.csv", key="sd_csv_a")
csv_b_name = st.text_input("CSV B filename (gene-level TopK)", "sample_gene_topk.csv", key="sd_csv_b")

def _scan_eval_samples(root: str):
    """
    Return list of (sample_dir, true_label_name) scanning 'root'.
    We treat "deepest directories that contain at least one .npy" as samples.
    true_label_name = the first-level subdir under root.
    """
    out = []
    if not os.path.isdir(root):
        raise FileNotFoundError(f"{root} not found.")
    # first-level dirs are class names
    cls_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not cls_dirs:
        raise RuntimeError(f"No class directories found under {root}.")
    cls_dirs = sorted(cls_dirs)
    for cname in cls_dirs:
        croot = os.path.join(root, cname)
        for dirpath, dirnames, filenames in os.walk(croot):
            if any(fn.lower().endswith(".npy") for fn in filenames):
                out.append((dirpath, cname))
    if not out:
        raise RuntimeError(f"No '.npy'-containing sample folders under {root}.")
    return out

# クリックで一括推論＆CSV出力
if st.button("Run batch evaluate & export CSV", key="btn_batch_export"):
    model = st.session_state.get("model")
    spec  = st.session_state.get("spec")
    if not model or not spec:
        st.warning("Init model first（Step A → Step B）。")
        st.stop()
    if not eval_root or not os.path.isdir(eval_root):
        st.warning("Eval root に有効なパスを指定してください。")
        st.stop()

    try:
        os.makedirs(export_dir, exist_ok=True)
        path_csv_a = os.path.join(export_dir, csv_a_name)
        path_csv_b = os.path.join(export_dir, csv_b_name)

        # 1) サンプル一覧（(sample_dir, true_label_name)）
        samples = _scan_eval_samples(eval_root)
        classes = spec.get("class_names", [str(i) for i in range(model.C)])
        class_to_idx = {c:i for i,c in enumerate(classes)}
        D_model = int(spec["D"])

        # 出力バッファ
        rows_a = []  # sample-level probabilities
        rows_b = []  # gene-level topK (pred & other-max)

        device = next(model.parameters()).device
        grp = model.multi.groups["vec"]  # access class-conditional group (for per-gene log-post)

        # 小ヘルパ：np.load から (D,) ベクトルへ
        def _vec_from_any(arr: np.ndarray) -> np.ndarray:
            if arr.ndim == 1:  return arr.astype(np.float32, copy=False)
            if arr.ndim == 2:  return arr.mean(axis=0, dtype=np.float32)
            if arr.ndim == 3:  return arr.mean(axis=(0,1), dtype=np.float32)
            raise RuntimeError(f"Unsupported ndim={arr.ndim}")

        st.write(f"Found {len(samples)} samples under `{eval_root}`")
        prog = st.progress(0.0)

        for si, (sample_dir, true_label_name) in enumerate(samples, start=1):
            try:
                files = sorted([f for f in os.listdir(sample_dir) if f.lower().endswith(".npy")])
                if not files:
                    _log(f"[SKIP] no .npy in {sample_dir}")
                    continue

                vecs, gene_names = [], []
                for fn in files:
                    try:
                        arr = np.load(os.path.join(sample_dir, fn), allow_pickle=False)
                        v = _vec_from_any(np.asarray(arr))
                        vecs.append(v); gene_names.append(os.path.splitext(fn)[0])
                    except Exception:
                        continue
                if not vecs:
                    _log(f"[SKIP] no valid vectors in {sample_dir}")
                    continue

                V = np.stack(vecs, axis=0)  # (K,D)
                if V.shape[1] != D_model:
                    _log(f"[SKIP] D mismatch in {sample_dir}: sample D={V.shape[1]}, model D={D_model}")
                    continue

                x = torch.from_numpy(V.astype(np.float32)).view(1, V.shape[0], 1, D_model).to(device)
                m = torch.ones((1, V.shape[0]), dtype=torch.float32, device=device)

                with torch.no_grad():
                    # ---- 1) sample-level probability ----
                    pred, prob, logp = model.predict_from_feats(x, m)  # prob:(1,C)
                    p = prob[0].detach().cpu().numpy()
                    pred_idx = int(pred[0].item())
                    pred_name = classes[pred_idx]

                    # true label idx (あればマップ。未知なら None)
                    true_idx = class_to_idx.get(true_label_name, None)

                    # CSV A: 1 サンプル 1 行
                    rec = {
                        "sample_path": sample_dir,
                        "sample_name": os.path.basename(sample_dir),
                        "true_label": true_label_name,
                        "true_label_idx": true_idx if true_idx is not None else -1,
                        "pred_label": pred_name,
                        "pred_label_idx": pred_idx,
                    }
                    for ci, cname in enumerate(classes):
                        rec[f"prob_{cname}"] = float(p[ci])
                    rows_a.append(rec)

                    # ---- 2) gene-level log posterior per class (B=1) ----
                    # llk_allc: (B,C,K)
                    llk_allc = grp.per_patch_loglik(x)  # log-likelihood per gene per class
                    prior_log = model.multi._log_prior().view(1, model.C, 1)
                    joint_keep = torch.nan_to_num(llk_allc, neginf=-1e9, posinf=1e9) + prior_log
                    logpost_keep = joint_keep - torch.logsumexp(joint_keep, dim=1, keepdim=True)  # (1,C,K)

                    # predicted-class の per-gene
                    logpost_pred = logpost_keep[0, pred_idx, :]  # (K,)
                    # other の最大クラスと値
                    mask = torch.ones(model.C, dtype=torch.bool, device=device)
                    mask[pred_idx] = False
                    other_vals, other_arg = logpost_keep[0, mask, :].max(dim=0)  # (K,), (K,)
                    # other 側のクラス index を実クラス index に写像
                    other_class_indices = torch.arange(model.C, device=device)[mask][other_arg]  # (K,)

                    # ---- Top-K 選定（pred / other-max）----
                    K = V.shape[0]
                    k = int(min(topk_gene_export, K))

                    # predicted
                    vals_pred, idx_pred = torch.topk(logpost_pred, k=k, largest=True, sorted=True)
                    # other-max
                    vals_other, idx_other = torch.topk(other_vals, k=k, largest=True, sorted=True)

                    # CSV B: long-form（サンプル×2k 行）
                    for rank, (gi, v) in enumerate(zip(idx_pred.tolist(), vals_pred.tolist()), start=1):
                        rows_b.append({
                            "sample_path": sample_dir,
                            "sample_name": os.path.basename(sample_dir),
                            "true_label": true_label_name,
                            "pred_label": pred_name,
                            "list_type": "predicted",         # 予測ラベルの条件付き確率 TopK
                            "class_name": pred_name,
                            "gene": gene_names[gi],
                            "rank": rank,
                            "logpost": float(v),
                            "prob": float(np.exp(v)),
                        })
                    for rank, (gi, v) in enumerate(zip(idx_other.tolist(), vals_other.tolist()), start=1):
                        other_cls_idx = int(other_class_indices[gi].item())
                        rows_b.append({
                            "sample_path": sample_dir,
                            "sample_name": os.path.basename(sample_dir),
                            "true_label": true_label_name,
                            "pred_label": pred_name,
                            "list_type": "other_max",         # 予測ラベル以外で最大の条件付き確率 TopK
                            "class_name": classes[other_cls_idx],
                            "gene": gene_names[gi],
                            "rank": rank,
                            "logpost": float(v),
                            "prob": float(np.exp(v)),
                        })

            except Exception as ex:
                _log(f"[WARN] failed at {sample_dir}: {ex}")
                continue
            finally:
                prog.progress(si / max(1, len(samples)))

        # ---- CSV 出力 ----
        df_a = pd.DataFrame(rows_a)
        df_b = pd.DataFrame(rows_b)
        df_a.to_csv(path_csv_a, index=False, encoding="utf-8")
        df_b.to_csv(path_csv_b, index=False, encoding="utf-8")
        _log(f"[EXPORT] Wrote A: {path_csv_a}  ({len(df_a)} rows)")
        _log(f"[EXPORT] Wrote B: {path_csv_b}  ({len(df_b)} rows)")
        st.success(f"Exported CSVs to:\n- {path_csv_a}\n- {path_csv_b}")

        # ダウンロードボタン（任意）
        with open(path_csv_a, "rb") as fa:
            st.download_button("Download CSV A (sample probs)", data=fa, file_name=os.path.basename(path_csv_a), mime="text/csv")
        with open(path_csv_b, "rb") as fb:
            st.download_button("Download CSV B (gene TopK)", data=fb, file_name=os.path.basename(path_csv_b), mime="text/csv")

    except Exception as e:
        _log(f"[ERROR] Batch export failed: {e}")
        st.error(f"Failed: {e}")
