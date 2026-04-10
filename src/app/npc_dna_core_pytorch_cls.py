# npc_dna_core_pytorch_cls.py
# -----------------------------------------------------------------------------
# Multi-class classification on folders of pre-encoded vectors (.npy) + class-
# conditional probabilistic circuits (RAT-SPN).
#  - 1 sample = 1 folder that contains many .npy vectors (genes) => K vectors.
#  - Each .npy can be (D,), (K,D), (K,q,D); we average over K/q to unify (D,).
#  - Dataset returns variable-length per-sample tensors: x:(K,1,D), m:(K,)
#  - DataLoader uses an identity collate -> batch is a list of samples.
#  - Training updates are STRICTLY per-sample (one optimizer step per sample).
#  - Epoch control lives HERE (core), not in the app. App calls fit() ONCE.
#  - Checkpointing: every N epochs + best-by-accuracy (model+optimizer+hparams).
#  - AMP (mixed precision) supported to reduce memory (enabled by default on CUDA).
# -----------------------------------------------------------------------------

from __future__ import annotations
import os, math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Speed hint for convolutions (not strictly used here but safe)
torch.backends.cudnn.benchmark = True

# ======= Utilities ============================================================

def _pick_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# RAT-SPN building blocks
# =============================================================================

class _TorchRATSPNLeaf(nn.Module):
    def __init__(self, var_idx: int, num_latents: int, num_pieces: int):
        super().__init__()
        self.v = int(var_idx)
        self.L = int(num_latents)
        self.P = int(num_pieces)
        self.mu = nn.Parameter(0.10 * torch.randn(self.L, self.P))
        self.log_sigma = nn.Parameter(torch.zeros(self.L, self.P))
        self.p_logits = nn.Parameter(torch.zeros(self.L, self.P))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x[:, self.v].view(-1, 1, 1)
        mu = self.mu.view(1, self.L, self.P)
        ls = self.log_sigma.view(1, self.L, self.P)
        logw = F.log_softmax(self.p_logits, dim=-1).view(1, self.L, self.P)
        inv_s2 = torch.exp(-2.0 * ls)
        logN = -0.5 * ((z - mu) ** 2) * inv_s2 - ls - 0.5 * math.log(2.0 * math.pi)
        return torch.logsumexp(logw + logN, dim=-1)  # (N, L)


class _TorchRATSPNLeafBlock(nn.Module):
    def __init__(self, scope: List[int], num_latents: int, num_pieces: int):
        super().__init__()
        self.leaves = nn.ModuleList([_TorchRATSPNLeaf(i, num_latents, num_pieces) for i in scope])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = None
        for leaf in self.leaves:
            v = leaf(x)
            out = v if out is None else (out + v)
        return out


class _TorchRATSPNProdSum(nn.Module):
    def __init__(self, left: nn.Module, right: nn.Module, num_latents: int):
        super().__init__()
        self.left = left; self.right = right; self.L = int(num_latents)
        self.mix_logits = nn.Parameter(0.01 * torch.randn(self.L, self.L))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.left(x); b = self.right(x)
        s = a + b
        logW = F.log_softmax(self.mix_logits, dim=-1)
        return torch.logsumexp(s.unsqueeze(1) + logW.unsqueeze(0), dim=-1)  # (N,L)


class _TorchRATSPN(nn.Module):
    def __init__(self, num_vars: int, depth: int = 5, num_latents: int = 64,
                 num_repetitions: int = 3, num_pieces: int = 2, leaf_group: int = 1, seed: int = 42):
        super().__init__()
        self.D = int(num_vars); self.depth=int(depth); self.L=int(num_latents)
        self.R=int(num_repetitions); self.P=int(num_pieces); self.leaf_group=int(max(1,leaf_group))
        roots=[]
        gen=torch.Generator()
        for r in range(self.R):
            gen.manual_seed(seed + 1000*r)
            scope=list(range(self.D))
            roots.append(self._build_tree(scope, self.depth, gen))
        self.roots = nn.ModuleList(roots)
        self.root_channel_logits = nn.Parameter(torch.zeros(self.R, self.L))
        self.rep_logits = nn.Parameter(torch.zeros(self.R))

    def _build_tree(self, scope: List[int], depth: int, gen: torch.Generator) -> nn.Module:
        if depth<=0 or len(scope)<=self.leaf_group:
            return _TorchRATSPNLeafBlock(scope, self.L, self.P)
        perm = torch.randperm(len(scope), generator=gen).tolist()
        scope = [scope[i] for i in perm]
        mid = len(scope)//2
        left_scope, right_scope = scope[:mid], scope[mid:]
        if len(left_scope)==0 or len(right_scope)==0:
            return _TorchRATSPNLeafBlock(scope, self.L, self.P)
        left = self._build_tree(left_scope, depth-1, gen)
        right= self._build_tree(right_scope, depth-1, gen)
        return _TorchRATSPNProdSum(left, right, self.L)

    def forward_ll(self, x_flat: torch.Tensor) -> torch.Tensor:
        outs=[]
        for r,root in enumerate(self.roots):
            ch = root(x_flat)  # (N,L)
            log_alpha = F.log_softmax(self.root_channel_logits[r:r+1,:], dim=-1)
            s = torch.logsumexp(ch + log_alpha, dim=-1)  # (N,)
            outs.append(s.unsqueeze(1))
        outs = torch.cat(outs, dim=1)  # (N,R)
        log_beta = F.log_softmax(self.rep_logits.view(1,self.R), dim=-1)
        return torch.logsumexp(outs + log_beta, dim=1)  # (N,)

    @torch.no_grad()
    def param_stats(self) -> Dict[str,float]:
        vals=[]
        for p in self.parameters():
            if p is not None and p.numel()>0 and p.dtype.is_floating_point:
                vals.append(p.detach().flatten())
        if not vals:
            return {"mean":0.0,"std":0.0,"min":0.0,"max":0.0,"norm":0.0}
        t=torch.cat(vals)
        return {"mean":float(t.mean().item()),"std":float(t.std(unbiased=False).item()),
                "min":float(t.min().item()),"max":float(t.max().item()),
                "norm":float(t.norm().item())}


# =============================================================================
# SPN over per-vector attributes (q*d); here q=1, but kept generic
# =============================================================================

class SPNJointGMMAttrVec(nn.Module):
    def __init__(self, K: int, q: int, d: int, M: int,
                 pc_depth_patch: int = 5, pc_latents_patch: int = 64,
                 pc_repetitions_patch: int = 3, pc_pieces_patch: int = 2):
        super().__init__()
        self.K, self.q, self.d, self.M = int(K), int(q), int(d), int(M)
        self.num_vars = int(self.q * self.d)
        self.patch_pc = _TorchRATSPN(
            num_vars=self.num_vars, depth=int(pc_depth_patch),
            num_latents=int(pc_latents_patch),
            num_repetitions=int(pc_repetitions_patch),
            num_pieces=int(pc_pieces_patch), leaf_group=1, seed=42
        )

    def _standardize(self, x_flat: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mu = x_flat.mean(0, keepdim=True)
            std= x_flat.std (0, keepdim=True).clamp_min(1e-3)
        return (x_flat - mu) / std

    def per_patch_loglik(self, x_bkqd: torch.Tensor) -> torch.Tensor:
        B,K,q,d = x_bkqd.shape
        x_flat = x_bkqd.reshape(B*K, q*d)
        x_flat = self._standardize(x_flat)
        ll = self.patch_pc.forward_ll(x_flat)     # (B*K,)
        ll = torch.nan_to_num(ll, neginf=-100.0, posinf=100.0) / float(self.num_vars)
        return ll.view(B, K)

    def per_patch_loglik_masked(self, x_bkqd: torch.Tensor, mask_bk: Optional[torch.Tensor]) -> torch.Tensor:
        B,K,q,d = x_bkqd.shape
        x_flat = x_bkqd.reshape(B*K, q*d)
        if mask_bk is None:
            return self.per_patch_loglik(x_bkqd)
        mask = mask_bk.reshape(B*K) > 0.5
        ll_full = torch.full((B*K,), fill_value=-100.0, device=x_bkqd.device, dtype=x_bkqd.dtype)
        if mask.any():
            xv = x_flat[mask]
            xv = self._standardize(xv)
            llv = self.patch_pc.forward_ll(xv)
            llv = torch.nan_to_num(llv, neginf=-100.0, posinf=100.0) / float(self.num_vars)
            ll_full[mask] = llv
        return ll_full.view(B, K)

    def forward_log_p_x(self, x_bkqd: torch.Tensor, reduce: str="mean") -> torch.Tensor:
        ll = self.per_patch_loglik(x_bkqd)
        return ll.mean(-1) if reduce=="mean" else ll.sum(-1)


class _ClassConditionalGroup(nn.Module):
    def __init__(self, num_classes: int, K: int, q: int, d: int, M: int,
                 pc_depth_patch: int = 5, pc_latents_patch: int = 64,
                 pc_repetitions_patch: int = 3, pc_pieces_patch: int = 2):
        super().__init__()
        self.C = int(num_classes)
        self.circuits = nn.ModuleList([
            SPNJointGMMAttrVec(K=K, q=q, d=d, M=M,
                               pc_depth_patch=pc_depth_patch,
                               pc_latents_patch=pc_latents_patch,
                               pc_repetitions_patch=pc_repetitions_patch,
                               pc_pieces_patch=pc_pieces_patch)
            for _ in range(self.C)
        ])

    @torch.no_grad()
    def per_patch_loglik(self, x_bkqd: torch.Tensor) -> torch.Tensor:
        outs=[]
        for c in range(self.C):
            llk = self.circuits[c].per_patch_loglik(x_bkqd)  # (B,K)
            outs.append(torch.nan_to_num(llk, neginf=-1e9, posinf=1e9).unsqueeze(1))
        return torch.cat(outs, dim=1)  # (B,C,K)

    def sum_loglik_masked(self, x_bkqd: torch.Tensor, mask_bk: Optional[torch.Tensor]) -> torch.Tensor:
        outs=[]
        for c in range(self.C):
            llk = self.circuits[c].per_patch_loglik_masked(x_bkqd, mask_bk)  # (B,K)
            if mask_bk is None:
                s = llk.mean(-1)
            else:
                num = (llk * mask_bk).sum(-1)
                den = mask_bk.sum(-1).clamp_min(1.0)
                s = num / den
            outs.append(torch.nan_to_num(s, neginf=-1e9, posinf=1e9).unsqueeze(1))
        return torch.cat(outs, dim=1)   # (B,C)


class SPNMultiAttrGroupsMC(nn.Module):
    def __init__(self, groups: Dict[str,_ClassConditionalGroup], num_classes: int,
                 learn_class_prior: bool=True):
        super().__init__()
        assert len(groups)==1, "Single group only."
        self.groups = nn.ModuleDict(groups)
        self.group_names = list(groups.keys())
        self.C = int(num_classes)
        self.class_prior_logits = nn.Parameter(torch.zeros(self.C)) if learn_class_prior else None

    def _log_prior(self)->torch.Tensor:
        if self.class_prior_logits is None:
            return torch.zeros(self.C, device=next(self.parameters()).device)
        return F.log_softmax(self.class_prior_logits, dim=-1)

    def joint_logp_per_class_masked(self, feats: Dict[str,torch.Tensor], masks: Dict[str,Optional[torch.Tensor]]) -> torch.Tensor:
        logs=[]
        for g in self.group_names:
            group = self.groups[g]
            x = feats[g]
            m = masks.get(g, None)
            l = group.sum_loglik_masked(x, m)     # (B,C)
            logs.append(torch.nan_to_num(l, neginf=-1e9, posinf=1e9))
        joint = torch.stack(logs, 0).sum(0) + self._log_prior().view(1,self.C)
        return torch.nan_to_num(joint, neginf=-1e9, posinf=1e9)

    @staticmethod
    def _log_softmax_rows(x: torch.Tensor) -> torch.Tensor:
        j = x - x.max(dim=1, keepdim=True).values
        return j - torch.logsumexp(j, dim=1, keepdim=True)
    def posterior_logprobs(self, feats: Dict[str, torch.Tensor],
                           masks: Optional[Dict[str, Optional[torch.Tensor]]] = None,
                           tau: float = 20.0) -> torch.Tensor:
        """
        joint = log p(y) + Σ_g mean_{k∈mask}(log p_g^c(x_{g,k}))
        安定化: 行最大 + prior を floor にして logaddexp → 行方向 log-softmax
        """
        masks = masks or {g: None for g in self.group_names}
        joint = self.joint_logp_per_class_masked(feats, masks)  # (B,C)
        joint = torch.nan_to_num(joint, neginf=-1e30, posinf=1e30)
        row_max = joint.max(dim=1, keepdim=True).values         # (B,1)
        prior_log = self._log_prior().view(1, self.C)           # (1,C)
        log_floor = row_max + prior_log                         # (B,C)
        joint_floored = torch.logaddexp(joint, log_floor)
        j = joint_floored - joint_floored.max(dim=1, keepdim=True).values
        logZ = torch.logsumexp(j, dim=1, keepdim=True)
        logp = j - logZ
        return torch.nan_to_num(logp, neginf=-30.0, posinf=0.0)


# =============================================================================
# Model (no encoder)
# =============================================================================

class NPC_VECTORS_MC(nn.Module):
    def __init__(self,
                 num_classes: int,
                 K: int, q: int, d: int,
                 mixture_M: int = 4,
                 learn_class_prior: bool = True,
                 pc_depth_patch: int = 5,
                 pc_latents_patch: int = 64,
                 pc_repetitions_patch: int = 3,
                 pc_pieces_patch: int = 2,
                 device: Optional[str] = None):
        super().__init__()
        self.C = int(num_classes)
        self.K = int(K)     # kept for hparams
        self.q = int(q)
        self.D = int(d)
        self.M = int(max(1, mixture_M))

        grp = _ClassConditionalGroup(num_classes=self.C, K=self.K, q=self.q, d=self.D, M=self.M,
                                     pc_depth_patch=pc_depth_patch, pc_latents_patch=pc_latents_patch,
                                     pc_repetitions_patch=pc_repetitions_patch, pc_pieces_patch=pc_pieces_patch)
        self.multi = SPNMultiAttrGroupsMC(groups={"vec": grp}, num_classes=self.C,
                                          learn_class_prior=True if (self.C>=2 and learn_class_prior) else False)
        self.group_names = ["vec"]
        self.to(_pick_device(device))


    @torch.no_grad()
    def predict_from_feats(self, x_bkqd: torch.Tensor, mask_bk: Optional[torch.Tensor]):
        logp = self.log_posteriors_from_feats(x_bkqd, mask_bk)
        prob = logp.exp()
        pred = prob.argmax(dim=1)
        return pred, prob, logp
    
    def log_posteriors_from_feats(self, x_bkqd: torch.Tensor, mask_bk: Optional[torch.Tensor]) -> torch.Tensor:
        feats = {"vec": x_bkqd}
        masks = {"vec": mask_bk}
        return self.multi.posterior_logprobs(feats, masks=masks)  # (B,C)

    def loss_and_stats_from_feats(self, x_bkqd: torch.Tensor, mask_bk: Optional[torch.Tensor], y: torch.Tensor,
                                  class_weights: Optional[torch.Tensor]=None):
        device = next(self.parameters()).device
        B = x_bkqd.size(0)
        y = y.long().to(device)
        logp = self.log_posteriors_from_feats(x_bkqd, mask_bk)    # (B,C)
        ce_per = -logp[torch.arange(B, device=device), y]
        if class_weights is not None:
            sw = class_weights.to(device)[y]
            ce = (ce_per * sw).sum() / sw.sum().clamp_min(1e-12)
        else:
            ce = ce_per.mean()
        with torch.no_grad():
            prob = logp.exp()
            pred = prob.argmax(dim=1)
            acc = (pred==y).float().mean().item()
        return ce, {"acc":acc}

    @torch.no_grad()
    def per_bin_keep_only_posteriors_from_feats(self, x_bkqd: torch.Tensor, mask_bk: Optional[torch.Tensor],
                                                target: str|int="pred") -> Tuple[torch.Tensor, torch.Tensor]:
        # NOTE: use K from input, not self.K (variable-length per sample)
        B, K, _, _ = x_bkqd.shape
        feats = {"vec": x_bkqd}
        masks = {"vec": mask_bk}
        joint = self.multi.joint_logp_per_class_masked(feats, masks)  # (B,C)
        logp = SPNMultiAttrGroupsMC._log_softmax_rows(joint)
        pred = logp.argmax(dim=1)
        t_idx = pred if (isinstance(target,str) and target=="pred") else torch.full_like(pred, int(target))

        prior_log = self.multi._log_prior().view(1, self.C, 1)
        joint_keep = torch.zeros(B, self.C, K, device=logp.device)
        grp = self.multi.groups["vec"]
        llk_allc = grp.per_patch_loglik(x_bkqd)  # (B,C,K)
        joint_keep += llk_allc
        joint_keep = joint_keep + prior_log
        logpost_keep = joint_keep - torch.logsumexp(joint_keep, dim=1, keepdim=True)  # (B,C,K)
        scores = logpost_keep[torch.arange(B, device=logp.device), t_idx, :]          # (B,K)
        return scores, t_idx


# =============================================================================
# Dataset: 1 sample = 1 folder with many .npy
# =============================================================================

class SampleDirDataset(Dataset):
    """
    Root/
      <classA>/.../<sampleA_folder_containing_npys>/*.npy
      <classB>/.../<sampleB_folder_containing_npys>/*.npy
    We treat the deepest directories that contain at least one .npy as "samples".
    samples = [(sample_dir_path, class_index)]
    __getitem__ loads ALL .npy in that directory (lazy), converts to (K,1,D) + mask(K,)
    """
    def __init__(self, root: str):
        super().__init__()
        self.root = root
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"{self.root} not found.")

        # top-level classes = first-level subdirs
        classes = sorted([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))])
        if not classes:
            raise RuntimeError(f"No class directories found under {self.root}.")
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        self.classes = classes

        # collect sample directories (deepest dirs that contain at least one .npy)
        samples: List[Tuple[str,int]] = []
        for c in classes:
            croot = os.path.join(self.root, c)
            for dirpath, dirnames, filenames in os.walk(croot):
                if any(fn.lower().endswith(".npy") for fn in filenames):
                    samples.append((dirpath, self.class_to_idx[c]))
        if not samples:
            raise RuntimeError(f"No .npy-containing folders under {self.root}.")
        self.samples = samples

        # determine D from one file
        probe_dir = self.samples[0][0]
        probe_files = [f for f in os.listdir(probe_dir) if f.lower().endswith(".npy")]
        probe = np.load(os.path.join(probe_dir, probe_files[0]), allow_pickle=False, mmap_mode="r")
        if   probe.ndim == 1: D = probe.shape[0]
        elif probe.ndim == 2: D = probe.shape[1]
        elif probe.ndim == 3: D = probe.shape[2]
        else: raise RuntimeError(f"Unsupported ndim={probe.ndim} in {probe_files[0]}")
        self.D = int(D)

    @staticmethod
    def _vec_from_any(arr: np.ndarray) -> np.ndarray:
        # (D,) -> (D,)
        # (K,D) -> mean over K -> (D,)
        # (K,q,D) -> mean over K,q -> (D,)
        if arr.ndim == 1:
            return arr.astype(np.float32, copy=False)
        if arr.ndim == 2:
            return arr.mean(axis=0, dtype=np.float32)
        if arr.ndim == 3:
            return arr.mean(axis=(0,1), dtype=np.float32)
        raise RuntimeError(f"Unsupported ndim={arr.ndim}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sample_dir, y = self.samples[idx]
        files = sorted([f for f in os.listdir(sample_dir) if f.lower().endswith(".npy")])
        vecs: List[np.ndarray] = []
        for fn in files:
            try:
                v = np.load(os.path.join(sample_dir, fn), allow_pickle=False)
                v = self._vec_from_any(np.asarray(v))
                vecs.append(v)
            except Exception:
                # skip malformed
                continue
        if not vecs:
            raise RuntimeError(f"No valid vectors in {sample_dir}")

        # K vectors -> tensor (K,1,D) and mask(K,)
        V = np.stack(vecs, axis=0)  # (K, D)
        if V.shape[1] != self.D:
            raise RuntimeError(f"D mismatch in {sample_dir}: got {V.shape[1]}, expected {self.D}")
        x = torch.from_numpy(V.astype(np.float32)).view(V.shape[0], 1, self.D)  # (K,1,D)
        m = torch.ones((V.shape[0],), dtype=torch.float32)                       # (K,)
        return x, m, int(y), sample_dir


def build_dataloaders_vectors_multiclass(
    train_root: str,
    val_root: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 0,
):
    ds_tr = SampleDirDataset(train_root)
    ds_va = SampleDirDataset(val_root) if (val_root and os.path.isdir(val_root)) else None

    # NEW: パディング＋マスク collate（B並列＋K並列）
    def collate_pad(batch):
        # batch: List[(x:(K,1,D), m:(K,), y:int, path:str)]
        B = len(batch)
        Ks = [b[0].shape[0] for b in batch]
        Kmax = max(Ks)
        D = batch[0][0].shape[2]
        x_bkqd = torch.zeros(B, Kmax, 1, D, dtype=torch.float32)
        m_bk   = torch.zeros(B, Kmax, dtype=torch.float32)
        y_b    = torch.empty(B, dtype=torch.long)
        paths  = []
        for i, (x_kqd, m_k, y, p) in enumerate(batch):
            K = x_kqd.shape[0]
            x_bkqd[i, :K, :, :] = x_kqd
            m_bk[i, :K] = m_k
            y_b[i] = int(y)
            paths.append(p)
        return x_bkqd, m_bk, y_b, paths

    kwargs = dict(
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, collate_fn=collate_pad, persistent_workers=(num_workers > 0)
    )
    dl_tr = DataLoader(ds_tr, **kwargs)
    kwargs.update(shuffle=False)
    dl_va = DataLoader(ds_va, **kwargs) if ds_va is not None else None

    spec = {
        "class_names": ds_tr.classes,
        "num_classes": len(ds_tr.classes),
        "D": int(ds_tr.D),
        "K": 1,  # hparam保持用（実際のKは毎バッチのKmaxで動的）
        "q": 1,
    }
    return dl_tr, dl_va, spec



# =============================================================================
# Train / Evaluate (strict per-sample updates; EPOCHS CONTROLLED HERE)
# =============================================================================

def _make_param_groups_for_vectors(model: NPC_VECTORS_MC, ce_lr: float, pc_lr: Optional[float]=None):
    pc_lr = pc_lr or ce_lr
    params_prior=[]
    if getattr(model.multi,"class_prior_logits",None) is not None:
        params_prior.append(model.multi.class_prior_logits)
    pc_params=[]
    grp = model.multi.groups["vec"]
    for c in range(model.C):
        pc_params += list(grp.circuits[c].patch_pc.parameters())
    param_groups=[]
    if params_prior: param_groups.append({"params": params_prior, "lr": ce_lr})
    if pc_params:    param_groups.append({"params": pc_params, "lr": pc_lr})
    if not param_groups: raise RuntimeError("No trainable parameters found.")
    return param_groups

def fit_multiclass_vectors(
    model: NPC_VECTORS_MC,
    dl_tr: DataLoader,
    dl_va: Optional[DataLoader] = None,
    epochs: int = 10,
    device: Optional[str] = None,
    ce_lr: float = 2e-3,
    pc_lr: Optional[float] = None,
    checkpoint_dir: Optional[str] = None,
    save_every: int = 5,
    resume_from: Optional[str] = None,
    log_every: int = 1000,
    log_fn=lambda m: None,
):
    """
    EPOCHS ARE FULLY CONTROLLED HERE (app should call this ONCE).
    Batched training: (B, Kmax, 1, D) + (B, Kmax mask) → single backward/step per batch.

    進捗出力:
      - 各エポック開始時/終了時
      - 各バッチ(=各ステップ)ごとに print() で詳細を出力
    """
    import time

    dev = _pick_device(device)
    model = model.to(dev).train()

    # optimizer
    param_groups = _make_param_groups_for_vectors(model, ce_lr=ce_lr, pc_lr=pc_lr)
    opt = torch.optim.Adam(param_groups)

    # resume
    start_epoch = 1
    best_acc = 0.0
    if resume_from:
        mdl, meta = load_checkpoint_vectors(resume_from, map_location=dev.type)
        model.load_state_dict(mdl.state_dict(), strict=False)
        if meta.get("optimizer_state", None):
            try:
                opt.load_state_dict(meta["optimizer_state"])
            except Exception:
                pass
        start_epoch = int(meta.get("epoch", 0)) + 1
        best_acc = float(meta.get("best_acc", 0.0))
        msg = f"[resume] from={resume_from} start_epoch={start_epoch} best_acc={best_acc:.4f}"
        print(msg, flush=True); log_fn(msg)

    # class weighting (over samples)
    class_weights = None
    try:
        ys = [y for _, y in dl_tr.dataset.samples]  # (sample_dir, y)
        if ys:
            C = int(model.C)
            cnt = torch.bincount(torch.tensor(ys, dtype=torch.long), minlength=C).float()
            N   = cnt.sum()
            cnt[cnt==0] = 1.0
            w = (N / (C * cnt))
            w = w * (C / w.sum().clamp_min(1e-12))
            class_weights = w.to(dev)
            msg = f"[class-weighting] {w.tolist()}"
            print(msg, flush=True); log_fn(msg)
    except Exception as e:
        msg = f"[class-weighting] WARN: {e}"
        print(msg, flush=True); log_fn(msg)

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    stop_epoch = start_epoch + int(epochs) - 1
    msg = f"[train] epochs={epochs} (from {start_epoch} to {stop_epoch}) device={dev}"
    print(msg, flush=True); log_fn(msg)

    # ----- epoch loop -----
    for ep in range(start_epoch, stop_epoch + 1):
        t_ep0 = time.time()
        model.train()
        tr_loss_sum, tr_acc_sum, n_batches = 0.0, 0.0, 0
        nbatches = len(dl_tr)
        msg = f"[train] epoch {ep}/{stop_epoch} : batches={nbatches}"
        print(msg, flush=True); log_fn(msg)

        for bi, (x_bkqd, m_bk, y_b, _paths) in enumerate(dl_tr, start=1):
            t_b0 = time.time()

            # (B,Kmax,1,D) / (B,Kmax) / (B,)
            x = x_bkqd.to(dev, non_blocking=True)
            m = m_bk.to(dev, non_blocking=True)
            y = y_b.to(dev, non_blocking=True)

            # ---- バッチ統計（Kの分布など）を計算 ----
            with torch.no_grad():
                # 実データK（maskの和）
                k_used = m.sum(dim=1).to(torch.int32)  # (B,)
                B = int(y.numel())
                Kmax = int(x.size(1))
                k_min = int(k_used.min().item())
                k_max = int(k_used.max().item())
                k_mean = float(k_used.float().mean().item())

            # ---- forward / backward / step ----
            opt.zero_grad(set_to_none=True)
            loss, st = model.loss_and_stats_from_feats(x, m, y, class_weights=class_weights)
            loss.backward()
            opt.step()

            # ---- ログ（各ステップごとに必ず print）----
            t_b1 = time.time()
            step_msg = (
                f"[step] epoch {ep}/{stop_epoch}  batch {bi}/{nbatches}  "
                f"B={B}  Kmax={Kmax}  K[min/mean/max]=[{k_min}/{k_mean:.1f}/{k_max}]  "
                f"loss={float(loss.detach().item()):.6f}  acc={st['acc']:.4f}  "
                f"dt={t_b1 - t_b0:.2f}s"
            )
            print(step_msg, flush=True)
            if (bi == 1) or (bi % max(1, int(log_every)) == 0) or (bi == nbatches):
                log_fn(step_msg)

            tr_loss_sum += float(loss.detach().item())
            tr_acc_sum  += float(st["acc"])
            n_batches   += 1

        # ---- evaluation (batched) ----
                # ---- evaluation (batched) ----
                # ---- evaluation (val は acc のみ) ----
        model.eval()

        if dl_va is not None:
            eval_loader, tag = dl_va, "val"
            ev = evaluate_val_accuracy_only(model, eval_loader, device=dev.type)   # ★ ここだけ acc-only
        else:
            eval_loader, tag = dl_tr, "train"
            # 学習データでのスナップショットを見るときは軽量で十分
            ev = evaluate_val_accuracy_only(model, eval_loader, device=dev.type)

        t_ep1 = time.time()

        # 表示は acc のみ（macroF1 は出さない）
        ep_msg = (f"[{tag}] epoch={ep} acc={ev['acc']:.4f} "
                  f"(train CE~{tr_loss_sum/max(1,n_batches):.4f}, "
                  f"acc~{tr_acc_sum/max(1,n_batches):.4f}, "
                  f"epoch_dt={t_ep1 - t_ep0:.1f}s)")
        print(ep_msg, flush=True); log_fn(ep_msg)

        # best 更新判定も acc のみ
        cur_acc = float(ev["acc"])
        if checkpoint_dir and (cur_acc > best_acc):
            best_acc = cur_acc
            path = os.path.join(checkpoint_dir, "best.pt")
            save_checkpoint_vectors(
                model, opt, epoch=ep, best_acc=best_acc, path=path,
                class_names=getattr(dl_tr.dataset, "classes", None)
            )
            best_msg = f"[ckpt] NEW BEST acc={best_acc:.4f} → {path}"
            print(best_msg, flush=True); log_fn(best_msg)


        # periodic checkpoint
        if checkpoint_dir and (ep % max(1, int(save_every)) == 0):
            path = os.path.join(checkpoint_dir, f"epoch_{ep:04d}.pt")
            save_checkpoint_vectors(
                model, opt, epoch=ep, best_acc=best_acc, path=path,
                class_names=getattr(dl_tr.dataset, "classes", None)
            )
            ckpt_msg = f"[ckpt] saved {path}"
            print(ckpt_msg, flush=True); log_fn(ckpt_msg)


        if dev.type == "cuda":
            torch.cuda.empty_cache()

    return model

@torch.no_grad()
def evaluate_val_accuracy_only(
    model: NPC_VECTORS_MC,
    dl: DataLoader,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    学習時の val 用：正解率のみを計算（最小限で高速・安定）
    - 混同行列や F1 は一切計算しない
    - バッチ進捗を print
    """
    import time

    dev = _pick_device(device)
    model.eval().to(dev)

    total = 0
    correct = 0

    total_batches = len(dl)
    print(f"[eval(val)] start (acc only): batches={total_batches}", flush=True)

    for bi, (x_bkqd, m_bk, y_b, _paths) in enumerate(dl, start=1):
        t0 = time.time()

        x = x_bkqd.to(dev, non_blocking=True)
        m = m_bk.to(dev, non_blocking=True)
        y = y_b.to(dev, non_blocking=True).long()

        pred, _prob, _ = model.predict_from_feats(x, m)

        c = (pred.view(-1) == y.view(-1)).sum().item()
        n = y.numel()
        correct += c
        total   += n

        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

        print(f"[eval(val)] batch {bi}/{total_batches}  acc_now={correct/max(1,total):.4f}  dt={dt:.2f}s",
              flush=True)

    acc = float(correct / max(1, total))
    print(f"[eval(val)] done: acc={acc:.4f}", flush=True)
    return {"acc": acc}





@torch.no_grad()
def evaluate_multiclass_vectors(
    model: NPC_VECTORS_MC,
    dl: DataLoader,
    device: Optional[str] = None
) -> Dict[str, float]:
    """
    マルチクラス評価（毎バッチ進捗 print / デバイス不一致修正 / 高速化）
    """
    import time

    dev = _pick_device(device)
    model.eval().to(dev)

    C = int(model.C)
    cm = torch.zeros(C, C, dtype=torch.int64, device=dev)  # ← モデルと同じデバイス

    total_samples = 0
    total_batches = len(dl)
    print(f"[eval] start evaluation: batches={total_batches}", flush=True)

    last_print_t = time.time()

    for bi, (x_bkqd, m_bk, y_b, _paths) in enumerate(dl, start=1):
        t0 = time.time()

        # データをデバイスへ
        x = x_bkqd.to(dev, non_blocking=True)
        m = m_bk.to(dev, non_blocking=True)
        y = y_b.to(dev, non_blocking=True).long()

        # 推論
        pred, _prob, _ = model.predict_from_feats(x, m)  # (B,)
        pred = pred.long()

        # 混同行列をベクトル化で更新
        idx = (y.view(-1) * C + pred.view(-1)).long()  # flat index: 0..C*C-1
        cm.view(-1).index_add_(
            0,
            idx,
            torch.ones_like(idx, dtype=torch.int64, device=dev),
        )
        total_samples += y.numel()

        # ★ 毎バッチ進捗を出す（バッチ時間も表示）
        t1 = time.time()
        print(f"[eval] batch {bi}/{total_batches}  dt={t1 - t0:.2f}s", flush=True)

    if total_samples == 0:
        return {"acc": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}

    # 集計は CPU で
    cm = cm.cpu()
    acc = float(cm.diag().sum().item() / max(1, cm.sum().item()))

    prec = rec = f1 = 0.0
    for c in range(C):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f = 2 * p * r / max(1e-12, p + r)
        prec += p; rec += r; f1 += f
    prec /= C; rec /= C; f1 /= C

    print(f"[eval] done: acc={acc:.4f}, macroF1={f1:.4f}", flush=True)
    return {"acc": acc, "macro_precision": float(prec), "macro_recall": float(rec), "macro_f1": float(f1)}




# =============================================================================
# Checkpoint I/O
# =============================================================================

def _hparams_from_model(model: NPC_VECTORS_MC) -> Dict[str, int | str]:
    grp = model.multi.groups["vec"]
    some_pc = grp.circuits[0].patch_pc
    return {
        "arch": "NPC_VECTORS_MC",
        "num_classes": model.C,
        "K": model.K, "q": model.q, "D": model.D,
        "pc_depth_patch": some_pc.depth,
        "pc_latents_patch": some_pc.L,
        "pc_repetitions_patch": some_pc.R,
        "pc_pieces_patch": some_pc.P,
    }

def save_checkpoint_vectors(model: NPC_VECTORS_MC,
                            optimizer: Optional[torch.optim.Optimizer],
                            epoch: int,
                            best_acc: float,
                            path: str,
                            class_names: Optional[List[str]] = None):
    ckpt = {
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "best_acc": float(best_acc),
        "hparams": _hparams_from_model(model),
        "class_names": class_names,
        "version": "vec_ckpt_v1",
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(ckpt, path)

def load_checkpoint_vectors(path: str, map_location: Optional[str] = None) -> Tuple[NPC_VECTORS_MC, Dict]:
    ckpt = torch.load(path, map_location=map_location or _pick_device().type)
    hp = ckpt.get("hparams", {})
    if hp.get("arch","") != "NPC_VECTORS_MC":
        raise RuntimeError("Checkpoint is not a VECTORS model checkpoint.")
    model = NPC_VECTORS_MC(
        num_classes=int(hp["num_classes"]),
        K=int(hp["K"]), q=int(hp["q"]), d=int(hp["D"]),
        mixture_M=4,
        learn_class_prior=True,
        pc_depth_patch=int(hp["pc_depth_patch"]),
        pc_latents_patch=int(hp["pc_latents_patch"]),
        pc_repetitions_patch=int(hp["pc_repetitions_patch"]),
        pc_pieces_patch=int(hp["pc_pieces_patch"]),
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    meta = {
        "epoch": ckpt.get("epoch", None),
        "best_acc": ckpt.get("best_acc", None),
        "optimizer_state": ckpt.get("optimizer_state", None),
        "hparams": hp,
        "class_names": ckpt.get("class_names", None),
        "version": ckpt.get("version", None),
    }
    return model, meta
