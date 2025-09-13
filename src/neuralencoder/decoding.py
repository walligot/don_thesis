import numpy as np
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
import plotly.io as pio
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import jax
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from copy import deepcopy

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.metrics import balanced_accuracy_score

# map: dataset_idx -> (dataset_name, subject, task)
from foundational_ssm.constants import DATASET_IDX_TO_GROUP


if 'google.colab' in sys.modules:
    pio.renderers.default = 'colab'
        
### Load Activations
def load_activation_dict(path):
    """
    Loads a nested activations dict saved with np.savez where:
      - Outer keys are dataset IDs (saved as strings)
      - Inner values are dicts with keys like 'ssm_pre_activation_0', etc.
    Returns:
      dict[int, dict[str, np.ndarray]]
    """
    loaded = np.load(path, allow_pickle=True)
    return {
        int(k): {ik: np.array(iv) for ik, iv in v.item().items()}
        for k, v in loaded.items()
    }

def load_activation_dict_rtt(path):
    """
    Loads a nested activations dict saved with np.savez where:
      - Outer keys are dataset IDs (saved as strings)
      - Inner values are dicts with keys like 'ssm_pre_activation_0', etc.
    Returns:
      dict[int, dict[str, np.ndarray]]
    """
    loaded = np.load(path, allow_pickle=True)
    return {
        55: {ik: np.array(iv) for ik, iv in v.item().items()}
        for k, v in loaded.items()
    }

def blend_and_normalise_np(model, d1, d2, norm_keys, insert_key="neural_input_raw", axis=1, eps=1e-8):
    """
    For each outer key in d1:
      - insert d2[key][insert_key] as `insert_key`
      - z-normalise each key in `norm_keys` along `axis`
    Returns a new blended dict (does not mutate d1/d2).
    """
    out = {}
    for k, inner1 in d1.items():
        inner = dict(inner1)  # shallow copy of inner dict
        inner[insert_key] = np.asarray(d2[k][insert_key])
        #print(model.encoders[k].shape)
        #print(inner['neural_input'].shape)
        encode_t = jax.vmap(model.encoders[k], in_axes=0)  # over time
        encode_bt = jax.vmap(encode_t, in_axes=0)
        inner['post_encoder_raw'] = encode_bt(np.asarray(d2[k]['neural_input_raw']))
        inner['post_encoder'] = encode_bt(inner['neural_input'])

        for nk in norm_keys:
            if nk in inner:
                x = np.asarray(inner[nk], dtype=np.float32)
                if x.ndim <= axis:
                    raise ValueError(f"{nk} for outer key {k} has ndim={x.ndim}, cannot normalise along axis={axis}.")
                m = x.mean(axis=axis, keepdims=True)
                s = x.std(axis=axis, ddof=0, keepdims=True)
                inner[nk] = (x - m) / np.maximum(s, eps)

        out[k] = inner
    return out

def blend_and_normalise_np_rtt(model, d1, norm_keys, insert_key="neural_input_raw", axis=1, eps=1e-8):
    """
    For each outer key in d1:
      - insert d2[key][insert_key] as `insert_key`
      - z-normalise each key in `norm_keys` along `axis`
    Returns a new blended dict (does not mutate d1/d2).
    """
    out = {}
    for k, inner1 in d1.items():
        inner = dict(inner1)  # shallow copy of inner dict
        #inner[insert_key] = np.asarray(d2[k][insert_key])
        #print(model.encoders[k].shape)
        #print(inner['neural_input'].shape)
        encode_t = jax.vmap(model.encoder, in_axes=0)  # over time
        encode_bt = jax.vmap(encode_t, in_axes=0)
        inner['post_encoder_raw'] = encode_bt(inner['neural_input_raw'])
        inner['post_encoder'] = encode_bt(inner['neural_input'])

        for nk in norm_keys:
            if nk in inner:
                x = np.asarray(inner[nk], dtype=np.float32)
                if x.ndim <= axis:
                    raise ValueError(f"{nk} for outer key {k} has ndim={x.ndim}, cannot normalise along axis={axis}.")
                m = x.mean(axis=axis, keepdims=True)
                s = x.std(axis=axis, ddof=0, keepdims=True)
                inner[nk] = (x - m) / np.maximum(s, eps)

        out[k] = inner
    return out


### Decoding
# -------------------- utilities --------------------

def _ensure_real_features(x3d: np.ndarray) -> np.ndarray:
    """
    Ensure real-valued features. If complex, split into real/imag along the last dim.
    (n_trials, T, D) -> (n_trials, T, D*2) when complex; unchanged if real.
    """
    x = np.asarray(x3d)
    if np.iscomplexobj(x):
        return np.concatenate([x.real, x.imag], axis=-1)
    return x

def _bootstrap_ci(vals: np.ndarray, n_boot=1000, ci=95.0, centre="median", seed=0):
    """Percentile bootstrap CI over 1D array; drops NaNs if present."""
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.RandomState(seed)
    stats = np.empty(n_boot, float)
    for i in range(n_boot):
        s = rng.choice(v, size=v.size, replace=True)
        stats[i] = np.median(s) if centre == "median" else np.mean(s)
    alpha = (100 - ci) / 2
    lo, hi = np.percentile(stats, [alpha, 100 - alpha])
    point = float(np.median(v) if centre == "median" else np.mean(v))
    return point, float(lo), float(hi)

def _fit_ridge_r2(Xtr, ytr, Xte, yte, alphas=(1e-3, 1e-2, 1e-1, 1.0)) -> float:
    dec = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        RidgeCV(alphas=alphas)
    )
    dec.fit(Xtr, ytr)
    return float(dec.score(Xte, yte))

def _fit_classifier_bac(Xtr, ytr_like, Xte, yte_like) -> float:
    """
    Balanced accuracy for subject/task classification.
    Robust to label dtype; returns NaN if only one class in train.
    """
    y_tr = np.asarray(ytr_like).ravel().astype(str)
    y_te = np.asarray(yte_like).ravel().astype(str)

    if np.unique(y_tr).size < 2:
        return np.nan

    enc = LabelEncoder().fit(y_tr)
    y_tr_enc = enc.transform(y_tr)

    # drop unseen test labels (rare, but safe)
    mask = np.isin(y_te, enc.classes_)
    if not np.all(mask):
        Xte = Xte[mask]; y_te = y_te[mask]
        if Xte.shape[0] == 0:
            return np.nan
    y_te_enc = enc.transform(y_te)

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=1000, class_weight="balanced")
    )
    clf.fit(Xtr, y_tr_enc)
    return float(balanced_accuracy_score(y_te_enc, clf.predict(Xte)))


# -------------------- splitter (per dataset, after flattening) --------------------

def split_flatten_per_group(
    hs_list: List[np.ndarray],
    beh_list: List[np.ndarray],
    group_ids: List[int],
    mask_list: Optional[List[np.ndarray]] = None,
    test_size: float = 0.2,
    random_state: int = 0,
    shuffle: bool = True,
):
    """
    For each dataset id:
      - ensure real features (split complex into real/imag),
      - (n_trials, T, D) -> flatten to (n_rows, D),
      - optionally apply a boolean/0-1 mask of shape (n_trials, T, 1),
      - split within that dataset after masking/flattening, then pool.

    Parameters
    ----------
    hs_list : list of arrays, each (n_trials, T, D)
    beh_list: list of arrays, each (n_trials, T[, B])
    group_ids: list of ints, same length as hs_list
    mask_list: optional list of arrays, each (n_trials, T, 1); truthy keeps row

    Returns
    -------
    Xtr, Xte : (N_tr, D), (N_te, D)
    ytr, yte : (N_tr, B), (N_te, B)
    group_tr/te   : int dataset id per row
    subject_tr/te : str subject per row
    task_tr/te    : str task per row
    """
    assert len(hs_list) == len(beh_list) == len(group_ids), "length mismatch"
    if mask_list is not None:
        assert len(mask_list) == len(hs_list), "mask_list length mismatch"

    rng = np.random.RandomState(random_state)

    Xtr_parts, Xte_parts = [], []
    ytr_parts, yte_parts = [], []
    gtr_parts, gte_parts = [], []
    str_parts_tr, str_parts_te = [], []
    tsk_parts_tr, tsk_parts_te = [], []

    D_seen = None
    B_seen = None

    for i, (hs_g, beh_g, gid) in enumerate(zip(hs_list, beh_list, group_ids)):
        # ensure real features for hs
        hs_g = _ensure_real_features(np.asarray(hs_g))
        beh_g = np.asarray(beh_g)

        if beh_g.ndim == 2:
            beh_g = beh_g[..., None]
        elif beh_g.ndim != 3:
            raise ValueError(f"behaviour for dataset {gid} must be 2D or 3D")

        if hs_g.ndim != 3:
            raise ValueError(f"features for dataset {gid} must be 3D (n_trials, T, D)")

        n_trials, T, D = hs_g.shape
        if beh_g.shape[0] != n_trials or beh_g.shape[1] != T:
            raise ValueError(f"time/trial dims mismatch between hs and beh for dataset {gid}")

        B = beh_g.shape[-1]

        # optional mask checks
        mask_g = None
        if mask_list is not None:
            mask_g = np.asarray(mask_list[i])
            if mask_g.ndim != 3 or mask_g.shape[0] != n_trials or mask_g.shape[1] != T or mask_g.shape[2] != 1:
                raise ValueError(f"mask for dataset {gid} must be (n_trials, T, 1)")
            # flatten later; allow non-bool but cast to bool via != 0

        if D_seen is None: D_seen = D
        if B_seen is None: B_seen = B
        if D_seen != D: raise ValueError(f"feature dim mismatch in dataset {gid}")
        if B_seen != B: raise ValueError(f"target dim mismatch in dataset {gid}")

        # flatten
        Xg = hs_g.reshape(-1, D)      # (n_trials*T, D)
        yg = beh_g.reshape(-1, B)     # (n_trials*T, B)

        # apply mask (after flattening)
        if mask_g is not None:
            mg = mask_g.reshape(-1).astype(bool)
            if mg.size != Xg.shape[0]:
                raise ValueError(f"flattened mask size mismatch in dataset {gid}")
            if mg.any():
                Xg = Xg[mg]
                yg = yg[mg]
            else:
                # nothing to keep; skip this dataset
                continue

        n = Xg.shape[0]
        if n < 2:
            # not enough samples to split; skip this dataset
            continue

        # per-dataset split (ceil for test; keep both sides non-empty)
        if isinstance(test_size, (int, np.integer)):
            n_te = int(test_size)
        else:
            n_te = int(np.ceil(float(test_size) * n))
        n_te = max(1, min(n_te, n - 1))

        order = rng.permutation(n) if shuffle else np.arange(n)
        te_idx = order[:n_te]
        tr_idx = order[n_te:]

        # labels from constant map
        _, subj, task = DATASET_IDX_TO_GROUP[int(gid)]
        subj = str(subj); task = str(task)

        # assemble
        Xtr_parts.append(Xg[tr_idx]);   Xte_parts.append(Xg[te_idx])
        ytr_parts.append(yg[tr_idx]);   yte_parts.append(yg[te_idx])

        # clean dtypes
        gtr_parts.append(np.full(tr_idx.size, int(gid), dtype=np.int64))
        gte_parts.append(np.full(te_idx.size, int(gid), dtype=np.int64))
        str_parts_tr.append(np.full(tr_idx.size, subj))
        str_parts_te.append(np.full(te_idx.size, subj))
        tsk_parts_tr.append(np.full(tr_idx.size, task))
        tsk_parts_te.append(np.full(te_idx.size, task))

    if not Xtr_parts or not Xte_parts:
        raise ValueError("no datasets had enough samples to split")

    Xtr = np.concatenate(Xtr_parts, axis=0)
    Xte = np.concatenate(Xte_parts, axis=0)
    ytr = np.concatenate(ytr_parts, axis=0)
    yte = np.concatenate(yte_parts, axis=0)

    group_tr = np.concatenate(gtr_parts, axis=0)
    group_te = np.concatenate(gte_parts, axis=0)
    subject_tr = np.concatenate(str_parts_tr, axis=0)
    subject_te = np.concatenate(str_parts_te, axis=0)
    task_tr = np.concatenate(tsk_parts_tr, axis=0)
    task_te = np.concatenate(tsk_parts_te, axis=0)

    return (Xtr, Xte, ytr, yte,
            group_tr, group_te,
            subject_tr, subject_te,
            task_tr, task_te)

def _concat_windows(x3d: np.ndarray, n_concat: int, mask3d: Optional[np.ndarray] = None) -> np.ndarray:
    """
    From (n_trials, T, D) → concat n_concat timesteps:
      -> (n_trials * groups, D * n_concat)
    Drops leftover timesteps if not divisible by n_concat.

    If mask3d is provided with shape (n_trials, T, 1), only masked
    timesteps (mask != 0) are kept before forming windows.
    """
    x = _ensure_real_features(np.asarray(x3d))  # handle complex
    if x.ndim != 3:
        raise ValueError("x3d must be 3D (n_trials, T, D)")
    n, T, D = x.shape

    # Fast path: no mask -> vectorised like original
    if mask3d is None:
        groups = T // n_concat
        if groups < 1:
            return np.empty((0, D * n_concat), dtype=x.real.dtype)
        x = x[:, :groups * n_concat, :]
        x = x.reshape(n, groups, n_concat, D)
        x = np.transpose(x, (0, 1, 3, 2))       # (n, g, D, n_concat)
        x = x.reshape(n * groups, D * n_concat) # (n*g, D*n_concat)
        return x

    # Masked path: window per trial after filtering by mask
    m = np.asarray(mask3d)
    if m.shape != (n, T, 1):
        raise ValueError("mask3d must have shape (n_trials, T, 1)")
    m = m.astype(bool)[..., 0]  # (n, T)

    rows = []
    for i in range(n):
        xi = x[i]            # (T, D)
        mi = m[i]            # (T,)
        if mi.any():
            x_keep = xi[mi]  # (Tk, D)
            g = x_keep.shape[0] // n_concat
            if g > 0:
                x_trim = x_keep[: g * n_concat, :].reshape(g, n_concat, D)
                x_trim = np.transpose(x_trim, (0, 2, 1))     # (g, D, n_concat)
                rows.append(x_trim.reshape(g, D * n_concat)) # (g, D*n_concat)
        # if no timesteps or no full window: skip trial

    if not rows:
        return np.empty((0, D * n_concat), dtype=x.real.dtype)
    return np.concatenate(rows, axis=0)

# -------------------- splitter for subject/task classification --------------------

def split_concat_per_group(
    hs_list: List[np.ndarray],
    group_ids: List[int],
    *,
    mask_list: Optional[List[np.ndarray]] = None,
    n_concat: int = 100,
    test_size: float = 0.2,
    random_state: int = 0,
    shuffle: bool = True,
):
    """
    For each dataset id:
      - ensure real features,
      - optionally apply (n_trials, T, 1) mask, keeping mask != 0,
      - concat n_concat timesteps per trial (drop leftovers),
      - flatten across trials×groups to rows,
      - split within that dataset after concat, then pool.

    Returns
    -------
    Xtr, Xte : (N_tr, D*n_concat), (N_te, D*n_concat)
    subject_tr/te : (N_tr,), (N_te,)
    task_tr/te    : (N_tr,), (N_te,)
    """
    assert len(hs_list) == len(group_ids), "length mismatch"
    if mask_list is not None:
        assert len(mask_list) == len(hs_list), "mask_list length mismatch"

    rng = np.random.RandomState(random_state)

    Xtr_parts, Xte_parts = [], []
    subj_tr_parts, subj_te_parts = [], []
    task_tr_parts, task_te_parts = [], []

    D_seen = None
    for i, (hs_g, gid) in enumerate(zip(hs_list, group_ids)):
        mask_g = None if mask_list is None else mask_list[i]
        Xg = _concat_windows(hs_g, n_concat=n_concat, mask3d=mask_g)  # (n_rows, D*n_concat)
        if Xg.size == 0:
            continue

        if D_seen is None:
            D_seen = Xg.shape[1]
        if D_seen != Xg.shape[1]:
            raise ValueError(f"feature dim mismatch after concat in dataset {gid}")

        n = Xg.shape[0]
        if n < 2:
            continue

        # per-dataset split (ceil for test; keep both sides non-empty)
        if isinstance(test_size, (int, np.integer)):
            n_te = int(test_size)
        else:
            n_te = int(np.ceil(float(test_size) * n))
        n_te = max(1, min(n_te, n - 1))

        order = rng.permutation(n) if shuffle else np.arange(n)
        te_idx = order[:n_te]
        tr_idx = order[n_te:]

        # labels from constant map
        _, subj, task = DATASET_IDX_TO_GROUP[int(gid)]
        subj = str(subj); task = str(task)

        Xtr_parts.append(Xg[tr_idx]); Xte_parts.append(Xg[te_idx])
        subj_tr_parts.append(np.full(tr_idx.size, subj, dtype=object))
        subj_te_parts.append(np.full(te_idx.size, subj, dtype=object))
        task_tr_parts.append(np.full(tr_idx.size, task, dtype=object))
        task_te_parts.append(np.full(te_idx.size, task, dtype=object))

    if not Xtr_parts or not Xte_parts:
        raise ValueError("no datasets had enough samples to split (classification).")

    Xtr = np.concatenate(Xtr_parts, axis=0)
    Xte = np.concatenate(Xte_parts, axis=0)
    subj_tr = np.concatenate(subj_tr_parts, axis=0)
    subj_te = np.concatenate(subj_te_parts, axis=0)
    task_tr = np.concatenate(task_tr_parts, axis=0)
    task_te = np.concatenate(task_te_parts, axis=0)

    return Xtr, Xte, subj_tr, subj_te, task_tr, task_te


# -------------------- data prep for a feature key --------------------

def _prep_lists_for_xkey(
    datasets: Dict[int, Dict[str, np.ndarray]],
    x_key: str,
    behaviour_key: str
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """Build hs_list, beh_list, dataset ids (sorted for determinism)."""
    hs_list, beh_list, ids, mask_list = [], [], [], []
    for ds_id in sorted(datasets.keys()):
        inner = datasets[ds_id]
        hs_list.append(np.asarray(inner[x_key]))          # may be complex
        beh_list.append(np.asarray(inner[behaviour_key])) # real, 2D/3D
        mask_list.append(np.asarray(inner["mask"]))       # real, 3D
        ids.append(int(ds_id))
    return hs_list, beh_list, ids, mask_list


# -------------------- main pipeline (no 'group' classifier) --------------------

def evaluate_feature_sets_pipeline(
    datasets: Dict[int, Dict[str, np.ndarray]],
    x_keys: List[str],
    behaviour_key: str,
    random_states: List[int],
    test_size: float = 0.2,
    alphas=(1e-3, 1e-2, 1e-1, 1.0),
    n_concat=50,
    n_boot: int = 5000,
    ci: float = 95.0,
    centre: str = "median",
    make_plots: bool = True
):
    """
    For each feature key (tqdm):
      - per-dataset split after flattening,
      - behaviour decoder (RidgeCV → R²),
      - subject & task classifiers (balanced accuracy),
      - repeat across random_states; bootstrap CIs; summary in results.
      - also compute per-dataset behaviour curves.
    """
    global_results = {}
    per_dataset = {ds_id: {} for ds_id in sorted(datasets.keys())}

    # ---- global evaluation across all datasets ----
    for xk in tqdm(x_keys, desc="Evaluating feature sets"):
        hs_list, beh_list, group_ids, mask_list = _prep_lists_for_xkey(datasets, xk, behaviour_key)

        r2_scores = []
        bac_subject = []
        bac_task = []

        for seed in random_states:
            (Xtr, Xte, ytr, yte,
             group_tr, group_te,
             subj_tr, subj_te,
             task_tr, task_te) = split_flatten_per_group(
                hs_list, beh_list, group_ids,
                #mask_list=mask_list,
                test_size=test_size, random_state=seed, shuffle=True
            )

            r2_scores.append(_fit_ridge_r2(Xtr, ytr, Xte, yte, alphas=alphas))


            # ---- subject/task classification: new concat splitter ----
            (Xtr_cls, Xte_cls,
             subj_tr_cls, subj_te_cls,
             task_tr_cls, task_te_cls) = split_concat_per_group(
                hs_list, group_ids,
                #mask_list=mask_list,
                n_concat=n_concat,  # change if you want a different window length
                test_size=test_size, random_state=seed, shuffle=True
            )
            bac_subject.append(_fit_classifier_bac(Xtr_cls, subj_tr_cls, Xte_cls, subj_te_cls))
            bac_task.append(_fit_classifier_bac(Xtr_cls, task_tr_cls, Xte_cls, task_te_cls))

        r2_scores = np.asarray(r2_scores)
        bac_subject = np.asarray(bac_subject)
        bac_task = np.asarray(bac_task)

        r2_pt, r2_lo, r2_hi = _bootstrap_ci(r2_scores, n_boot=n_boot, ci=ci, centre=centre, seed=0)
        s_pt, s_lo, s_hi = _bootstrap_ci(bac_subject, n_boot=n_boot, ci=ci, centre=centre, seed=0)
        t_pt, t_lo, t_hi = _bootstrap_ci(bac_task, n_boot=n_boot, ci=ci, centre=centre, seed=0)

        global_results[xk] = {
            "behaviour_scores": r2_scores, "behaviour_point": r2_pt, "behaviour_ci": (r2_lo, r2_hi),
            "subject_scores": bac_subject, "subject_point": s_pt,    "subject_ci": (s_lo, s_hi),
            "task_scores": bac_task,       "task_point": t_pt,       "task_ci": (t_lo, t_hi),
        }

    # ---- per-dataset behaviour curves ----
    for ds_id in sorted(datasets.keys()):
        for xk in x_keys:
            hs_list = [np.asarray(datasets[ds_id][xk])]
            beh_list = [np.asarray(datasets[ds_id][behaviour_key])]
            group_ids = [int(ds_id)]
            r2s = []
            for seed in random_states:
                Xtr, Xte, ytr, yte, *_ = split_flatten_per_group(
                    hs_list, beh_list, group_ids,
                    test_size=test_size, random_state=seed, shuffle=True
                )
                r2s.append(_fit_ridge_r2(Xtr, ytr, Xte, yte, alphas=alphas))
            r2s = np.asarray(r2s)
            pt, lo, hi = _bootstrap_ci(r2s, n_boot=n_boot, ci=ci, centre=centre, seed=0)
            per_dataset[ds_id][xk] = {"scores": r2s, "point": pt, "ci": (lo, hi)}

    results = {
        "global": global_results,
        "per_dataset": per_dataset,
        "x_keys": list(x_keys),
        "dataset_ids": list(sorted(datasets.keys())),
    }

    if make_plots:
        plot_results_grid(results, subplot_labels=x_keys)  # default labels

    return results


# -------------------- plotting (2×2 grid) --------------------

def plot_results_grid(results: Dict, subplot_labels: List[str],
                      baseline_subject: float, baseline_task: float):
    """
    Build a 2×2 grid:
      (0,0) global behaviour (red CI band)
      (0,1) per-dataset behaviour (task → colour family; datasets within task → shades)
      (1,0) task classification  (+ dotted baseline)
      (1,1) subject classification (+ dotted baseline)

    subplot_labels: x tick labels for ALL subplots (order must match results['x_keys'])
    """
    import re
    from collections import defaultdict

    x_keys = results["x_keys"]
    assert len(subplot_labels) == len(x_keys), "subplot_labels must match length of x_keys"
    xs = np.arange(len(x_keys))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)

    # ---- TL: global behaviour ----
    ax = axes[0, 0]
    beh_pts = [results["global"][k]["behaviour_point"] for k in x_keys]
    beh_lo  = [results["global"][k]["behaviour_ci"][0] for k in x_keys]
    beh_hi  = [results["global"][k]["behaviour_ci"][1] for k in x_keys]
    ax.fill_between(xs, beh_lo, beh_hi, alpha=0.25, color="red")
    ax.plot(xs, beh_pts, marker="o", color="red")
    ax.set_xticks(xs, subplot_labels, rotation=45, ha="right")
    ax.set_ylabel("Behaviour R²")
    ax.set_title("Behaviour decoding (global)")

    # ---- TR: per-dataset behaviour (task-coloured, shaded per dataset) ----
    ax = axes[0, 1]

    # helper: extract leading numeric id if present
    def base_int(ds_id):
        m = re.match(r'^(\d+)', str(ds_id))
        return int(m.group(1)) if m else None

    # map each dataset to (subject, task) via DATASET_IDX_TO_GROUP[base_id]
    ds_info = []  # (orig_key, subject, task)
    for ds_id in results["dataset_ids"]:
        bi = base_int(ds_id)
        if bi is not None and bi in DATASET_IDX_TO_GROUP:
            subj = DATASET_IDX_TO_GROUP[bi][1]
            task = DATASET_IDX_TO_GROUP[bi][-1]  # last element as task
        else:
            subj = str(ds_id)
            task = "unknown"
        ds_info.append((ds_id, subj, task))

    # group dataset ids by task
    by_task = defaultdict(list)
    for ds_id, subj, task in ds_info:
        by_task[task].append((ds_id, subj))

    # assign a distinct cmap per task (cycle if more tasks than cmaps)
    task_names = list(by_task.keys())
    task_cmaps = [
        "Blues", "Greens", "Oranges", "Purples", "Reds", "Greys",
        "BuGn", "GnBu", "PuBu", "YlGn"
    ]
    cmap_for_task = {
        t: plt.get_cmap(task_cmaps[i % len(task_cmaps)])
        for i, t in enumerate(task_names)
    }

    # plot each dataset with a shade from its task's colour map
    for task in task_names:
        cmap = cmap_for_task[task]
        group = by_task[task]  # list of (ds_id, subj)
        n = len(group)
        shades = np.linspace(0.35, 0.85, num=max(2, n))  # mid-range for visibility
        for (shade, (ds_id, subj)) in zip(shades, group):
            col = cmap(shade)
            pts = [results["per_dataset"][ds_id][k]["point"] for k in x_keys]
            los = [results["per_dataset"][ds_id][k]["ci"][0] for k in x_keys]
            his = [results["per_dataset"][ds_id][k]["ci"][1] for k in x_keys]
            ax.fill_between(xs, los, his, alpha=0.18, color=col)
            ax.plot(xs, pts, marker="o", color=col, label=f"{subj} | {task}")

    ax.set_xticks(xs, subplot_labels, rotation=45, ha="right")
    ax.set_ylabel("Behaviour R²")
    ax.set_title("Behaviour decoding per dataset")
    ax.legend(ncol=2, fontsize=9)

    # ---- BL: task classification ----
    ax = axes[1, 0]
    tsk_pts = [results["global"][k]["task_point"] for k in x_keys]
    tsk_lo  = [results["global"][k]["task_ci"][0] for k in x_keys]
    tsk_hi  = [results["global"][k]["task_ci"][1] for k in x_keys]
    ax.fill_between(xs, tsk_lo, tsk_hi, alpha=0.2)
    ax.plot(xs, tsk_pts, marker="o")
    ax.axhline(baseline_task, linestyle="--", linewidth=1.0, label="Baseline - Random Guess")
    ax.set_xticks(xs, subplot_labels, rotation=45, ha="right")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Task classification (global)")
    ax.legend(loc="lower right", fontsize=9)

    # ---- BR: subject classification ----
    ax = axes[1, 1]
    sub_pts = [results["global"][k]["subject_point"] for k in x_keys]
    sub_lo  = [results["global"][k]["subject_ci"][0] for k in x_keys]
    sub_hi  = [results["global"][k]["subject_ci"][1] for k in x_keys]
    ax.fill_between(xs, sub_lo, sub_hi, alpha=0.2)
    ax.plot(xs, sub_pts, marker="o")
    ax.axhline(baseline_subject, linestyle="--", linewidth=1.0, label="Baseline - Random Guess")
    ax.set_xticks(xs, subplot_labels, rotation=45, ha="right")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Subject classification (global)")
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    return fig, axes

def plot_results_grid_behaviour(results: Dict, subplot_labels: List[str], save_path=None):
    """
    Build a 1×2 grid:
      (0,0) global behaviour (red CI band)
      (0,1) per-dataset behaviour (task → colour family; datasets within task → shades)

    Legend labels match the original: "{subject} | {task}"

    subplot_labels: x tick labels for ALL subplots (order must match results['x_keys'])
    """
    x_keys = results["x_keys"]
    assert len(subplot_labels) == len(x_keys), "subplot_labels must match length of x_keys"
    xs = np.arange(len(x_keys))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharex=False)

    # ---- (0,0) global behaviour ----
    ax = axes[0]
    beh_pts = [results["global"][k]["behaviour_point"] for k in x_keys]
    beh_lo  = [results["global"][k]["behaviour_ci"][0] for k in x_keys]
    beh_hi  = [results["global"][k]["behaviour_ci"][1] for k in x_keys]
    ax.fill_between(xs, beh_lo, beh_hi, alpha=0.25, color="red")
    ax.plot(xs, beh_pts, marker="o", color="red")
    ax.set_xticks(xs, subplot_labels, rotation=45, ha="right")
    ax.set_ylabel("Behaviour R²")
    ax.set_title("Behaviour decoding (global)")

    # ---- (0,1) per-dataset behaviour (task-coloured, shaded per dataset) ----
    ax = axes[1]

    # helper: extract leading numeric id if present
    def base_int(ds_id):
        m = re.match(r'^(\d+)', str(ds_id))
        return int(m.group(1)) if m else None

    # map each dataset to (subject, task) via DATASET_IDX_TO_GROUP[base_id]
    ds_info = []  # list of tuples: (orig_key, subject, task)
    for ds_id in results["dataset_ids"]:
        bi = base_int(ds_id)
        if bi is not None and bi in DATASET_IDX_TO_GROUP:
            subj = DATASET_IDX_TO_GROUP[bi][1]
            task = DATASET_IDX_TO_GROUP[bi][-1]  # "take the last element per mapped id as task"
        else:
            subj = str(ds_id)
            task = "unknown"
        ds_info.append((ds_id, subj, task))

    # group dataset ids by task
    from collections import defaultdict
    by_task = defaultdict(list)
    for ds_id, subj, task in ds_info:
        by_task[task].append((ds_id, subj))

    # assign a distinct cmap per task (cycle if more tasks than cmaps)
    task_names = list(by_task.keys())
    task_cmaps = [
        "Blues", "Greens", "Oranges", "Purples", "Reds", "Greys",
        "BuGn", "GnBu", "PuBu", "YlGn"  # extras if needed
    ]
    cmap_for_task = {
        t: plt.get_cmap(task_cmaps[i % len(task_cmaps)])
        for i, t in enumerate(task_names)
    }

    # for each task, give each dataset a distinct shade from its task cmap
    for task in task_names:
        cmap = cmap_for_task[task]
        group = by_task[task]  # list of (ds_id, subj)
        n = len(group)
        # choose mid-range shades to keep visibility; light→dark across datasets
        shades = np.linspace(0.35, 0.85, num=max(2, n))
        for (shade, (ds_id, subj)) in zip(shades, group):
            col = cmap(shade)
            pts = [results["per_dataset"][ds_id][k]["point"] for k in x_keys]
            los = [results["per_dataset"][ds_id][k]["ci"][0] for k in x_keys]
            his = [results["per_dataset"][ds_id][k]["ci"][1] for k in x_keys]
            ax.fill_between(xs, los, his, alpha=0.18, color=col)
            ax.plot(xs, pts, marker="o", color=col, label=f"{subj} | {task}")

    ax.set_xticks(xs, subplot_labels, rotation=45, ha="right")
    ax.set_ylabel("Behaviour R²")
    ax.set_title("Behaviour decoding per dataset")
    ax.legend(ncol=2, fontsize=9)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    return fig, axes


def merge_results(res_a: dict, res_b: dict, *,
              allow_overlap: bool = False,
              order: str = "a_then_b"):
    """
    Merge two pipeline results with disjoint (or overlapping) x_keys.

    - Requires identical dataset_ids (same data pool).
    - If x_keys overlap and allow_overlap=False, raises.
    - order: "a_then_b" or "b_then_a" controls x_keys order.

    Returns merged results dict compatible with plot_results_grid(...).
    """
    # 1) sanity: same datasets
    a_ids = list(res_a["dataset_ids"])
    b_ids = list(res_b["dataset_ids"])
    if a_ids != b_ids:
        raise ValueError("dataset_ids differ between runs; merge only if they match.")

    # 2) x_keys handling
    a_keys = list(res_a["x_keys"])
    b_keys = list(res_b["x_keys"])
    overlap = set(a_keys).intersection(b_keys)
    if overlap and not allow_overlap:
        raise ValueError(f"x_keys overlap: {sorted(overlap)}. Set allow_overlap=True to overwrite.")

    merged = deepcopy(res_a)
    # global + per_dataset are dicts keyed by x_key; just update with b’s entries
    merged["global"].update(res_b["global"])
    for ds_id in merged["dataset_ids"]:
        merged["per_dataset"][ds_id].update(res_b["per_dataset"][ds_id])

    # 3) x_keys order
    if order == "a_then_b":
        merged_keys = a_keys + [k for k in b_keys if k not in a_keys]
    elif order == "b_then_a":
        merged_keys = b_keys + [k for k in a_keys if k not in b_keys]
    else:
        raise ValueError("order must be 'a_then_b' or 'b_then_a'.")

    merged["x_keys"] = merged_keys
    return merged

import gzip, pickle, numpy as np

_FORMAT_TAG = "ssm_eval_results_v1"

def save_decoding_results(results: dict, path: str) -> None:
    """
    Save results dict to a gzipped pickle. Example: save_results(res, "results.pkl.gz")
    """
    blob = {
        "__format__": _FORMAT_TAG,
        "__numpy__": np.__version__,
        "payload": results,
    }
    with gzip.open(path, "wb") as f:
        pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_decoding_results(path: str) -> dict:
    """
    Load results dict saved by save_results.
    """
    with gzip.open(path, "rb") as f:
        blob = pickle.load(f)
    if not isinstance(blob, dict) or blob.get("__format__") != _FORMAT_TAG:
        raise ValueError("File does not look like a saved results blob.")
    return blob["payload"]


### RTT decoding

# ---------- utils ----------

def _ensure_real_features_rtt(x3d: np.ndarray) -> np.ndarray:
    """
    Ensure real-valued features. If complex, split into real/imag along last dim.
    (n_trials, T, D) -> (n_trials, T, D*2) when complex; unchanged if real.
    """
    x = np.asarray(x3d)
    return np.concatenate([x.real, x.imag], axis=-1) if np.iscomplexobj(x) else x


def _bootstrap_ci_rtt(vals: np.ndarray, n_boot=5000, ci=95.0, centre="median", seed=0):
    """
    Percentile bootstrap CI over a 1D array; drops NaNs if present.
    Returns (point_estimate, lo, hi) where point_estimate is mean/median.
    """
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.RandomState(seed)
    stats = np.empty(n_boot, float)
    for i in range(n_boot):
        s = rng.choice(v, size=v.size, replace=True)
        stats[i] = np.median(s) if centre == "median" else np.mean(s)
    alpha = (100.0 - ci) / 2.0
    lo, hi = np.percentile(stats, [alpha, 100.0 - alpha])
    point = float(np.median(v) if centre == "median" else np.mean(v))
    return point, float(lo), float(hi)


def _fit_ridge_r2_rtt(Xtr, ytr, Xte, yte, alphas=(1e-3, 1e-2, 1e-1, 1.0)) -> float:
    """
    StandardScaler + RidgeCV; returns R^2 on the test split.
    """
    pipe = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                         RidgeCV(alphas=alphas))
    pipe.fit(Xtr, ytr)
    return float(pipe.score(Xte, yte))


def _row_split_indices_rtt(n_rows: int, test_size=0.2, seed=0, shuffle=True):
    """
    Make train/test row indices with ceil test size (like sklearn).
    Ensures both sides non-empty.
    """
    if isinstance(test_size, (int, np.integer)):
        n_te = int(test_size)
    else:
        n_te = int(np.ceil(float(test_size) * n_rows))
    n_te = max(1, min(n_te, n_rows - 1))
    rng = np.random.RandomState(seed)
    order = rng.permutation(n_rows) if shuffle else np.arange(n_rows)
    te = order[:n_te]
    tr = order[n_te:]
    return tr, te


# ---------- main pipeline (single-group) ----------

def evaluate_single_group_pipeline_rtt(
    datasets: dict,
    x_keys: list,
    behaviour_key: str,
    random_states: list,
    test_size: float = 0.2,
    alphas=(1e-3, 1e-2, 1e-1, 1.0),
    n_boot: int = 5000,
    ci: float = 95.0,
    centre: str = "median",
    lock_splits_across_xkeys: bool = True,
):
    """
    Single-group variant: `datasets` has ONE outer key.
    For each x_key:
      - ensure real features, flatten trials×time to rows,
      - for each seed: row-wise split, fit RidgeCV, record R^2,
      - bootstrap CI across seeds.
    Returns a compact results dict.
    """
    if len(datasets) != 1:
        raise ValueError("Expected exactly one outer key in `datasets`.")

    # grab the only inner dict
    (outer_key,) = list(datasets.keys())
    inner = datasets[outer_key]

    # prepare y once (shape -> (N*T, B))
    beh = np.asarray(inner[behaviour_key])
    if beh.ndim == 2:
        beh = beh[..., None]
    if beh.ndim != 3:
        raise ValueError("behaviour array must be 2D or 3D.")
    N, T, B = beh.shape
    y_all = beh.reshape(-1, B)
    n_rows_ref = y_all.shape[0]

    # optionally precompute per-seed splits once (reused for all x_keys)
    precomp_splits = {}
    if lock_splits_across_xkeys:
        for seed in random_states:
            precomp_splits[seed] = _row_split_indices_rtt(n_rows_ref, test_size=test_size, seed=seed, shuffle=True)

    results = {"x_keys": list(x_keys), "global": {}}

    for xk in tqdm(x_keys):
        X = _ensure_real_features_rtt(np.asarray(inner[xk]))  # (N,T,D or 2D* when complex)
        if X.ndim != 3:
            raise ValueError(f"features '{xk}' must be 3D (n_trials, T, D).")
        if X.shape[:2] != (N, T):
            raise ValueError(f"features '{xk}' first two dims must match behaviour (N,T).")
        D = X.shape[-1]
        X_all = X.reshape(-1, D)

        r2_scores = []
        for seed in random_states:
            if lock_splits_across_xkeys:
                tr_idx, te_idx = precomp_splits[seed]
            else:
                tr_idx, te_idx = _row_split_indices_rtt(X_all.shape[0], test_size=test_size, seed=seed, shuffle=True)

            Xtr, Xte = X_all[tr_idx], X_all[te_idx]
            ytr, yte = y_all[tr_idx], y_all[te_idx]

            r2_scores.append(_fit_ridge_r2_rtt(Xtr, ytr, Xte, yte, alphas=alphas))

        r2_scores = np.asarray(r2_scores, dtype=float)
        point, lo, hi = _bootstrap_ci_rtt(r2_scores, n_boot=n_boot, ci=ci, centre=centre, seed=0)

        results["global"][xk] = {
            "behaviour_scores": r2_scores,
            "behaviour_point": point,
            "behaviour_ci": (lo, hi),
        }

    return results


# ---------- plotting ----------

def plot_single_group_decoding_rtt(results: dict, labels: list):
    """
    Single plot: points across x_keys with a red CI band.
    `labels` must align with results['x_keys'] for x-ticks.
    """
    x_keys = results["x_keys"]
    if len(labels) != len(x_keys):
        raise ValueError("`labels` must have same length as results['x_keys'].")

    xs = np.arange(len(x_keys))
    pts = [results["global"][k]["behaviour_point"] for k in x_keys]
    lo  = [results["global"][k]["behaviour_ci"][0] for k in x_keys]
    hi  = [results["global"][k]["behaviour_ci"][1] for k in x_keys]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(xs, lo, hi, alpha=0.25, color="red")
    ax.plot(xs, pts, marker="o")
    ax.set_xticks(xs, labels, rotation=45, ha="right")
    ax.set_ylabel("Behaviour R²")
    ax.set_title("Behaviour decoding")
    fig.tight_layout()
    return fig, ax