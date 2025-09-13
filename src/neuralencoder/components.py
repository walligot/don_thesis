import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import jax
import equinox as eqx
import jax.numpy as jnp
from foundational_ssm.models.s5 import discretize_zoh


# GLU

# ---- small, NumPy-only GELU to avoid jax dependency ----
def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * (x**3))))

def compute_gate_masks_by_context(model, by_context, block_idx=0, gate_path="w2",
                                  x_key="pre_activation_0",
                                  open_rule=("thr", 0.8), closed_rule=("thr", 0.2),
                                  normalise: bool = False, ctx_ids=None):
    """
    by_context[cid] may include:
      - x_key: list of (T, H) arrays
      - 'mask': either (n_trials, T, 1) array or list of (T, 1)/(T,) arrays
    """
    def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

    blk = model.ssm_blocks[block_idx]; ssm = blk.ssm
    lam_c = np.asarray(ssm.Lambda_re) + 1j*np.asarray(ssm.Lambda_im)
    dt = np.exp(np.asarray(ssm.log_step).reshape(-1))
    lam_d = np.exp(dt * lam_c)

    Wg_raw = np.asarray(getattr(blk.glu, gate_path).weight)
    bg_raw = getattr(blk.glu, gate_path).bias
    bg = None if bg_raw is None else np.asarray(bg_raw)

    H = int(ssm.H)
    Wg = Wg_raw if Wg_raw.shape[0] == H else Wg_raw.T
    if Wg.shape[0] != H:
        raise ValueError(f"W_gate incompatible with H={H}: got {Wg.shape}")

    if ctx_ids is None:
        ctx_ids = list(by_context.keys())
    meanG, stdG, openM, closedM = [], [], [], []

    for cid in ctx_ids:
        Xs = [np.asarray(x) for x in by_context[cid][x_key]]  # list of (T, H)
        if not Xs:
            # keep shapes consistent
            meanG.append(np.full(H, np.nan)); stdG.append(np.full(H, np.nan))
            openM.append(np.zeros(H, bool)); closedM.append(np.zeros(H, bool))
            continue
        if Xs[0].shape[1] != H:
            raise ValueError(f"H mismatch in context {cid}: {Xs[0].shape[1]} vs {H}")

        # optional masks
        Ms = [None] * len(Xs)
        if "mask" in by_context[cid] and by_context[cid]["mask"] is not None:
            Mraw = by_context[cid]["mask"]
            if isinstance(Mraw, (list, tuple)):
                Ms = [np.asarray(m).reshape(-1).astype(bool) for m in Mraw]
            else:
                Mraw = np.asarray(Mraw)
                if Mraw.ndim == 3 and Mraw.shape[2] == 1 and Mraw.shape[0] == len(Xs):
                    Ms = [Mraw[i, :, 0].astype(bool) for i in range(Mraw.shape[0])]
                elif Mraw.ndim == 2 and Mraw.shape[0] == len(Xs):
                    Ms = [Mraw[i, :].astype(bool) for i in range(Mraw.shape[0])]
                else:
                    raise ValueError(f"mask shape mismatch in context {cid}")

        T_min = min(x.shape[0] for x in Xs)

        # warm-up (same as before)
        mod = np.clip(np.abs(lam_d), 1e-8, 1-1e-8)
        tau = -1.0 / np.log(mod)
        T_warm = min(int(np.ceil(5*np.max(tau))), int(0.01*T_min), T_min-1)

        # GELU on SSM outputs, then gate logits (apply mask per trial)
        Z_trials = []
        for Xpre, mi in zip(Xs, Ms):
            Xseg = Xpre[T_warm:]                  # (T', H)
            if mi is not None:
                mi_seg = mi[T_warm:].astype(bool)
                if mi_seg.shape[0] < Xseg.shape[0]:
                    # pad false if mask shorter (defensive, minimal)
                    mi_seg = np.pad(mi_seg, (0, Xseg.shape[0]-mi_seg.shape[0]), constant_values=False)
                elif mi_seg.shape[0] > Xseg.shape[0]:
                    mi_seg = mi_seg[:Xseg.shape[0]]
                Xseg = Xseg[mi_seg]
            if Xseg.shape[0] == 0:
                continue
            Xg = gelu(Xseg)
            Z  = Xg @ Wg
            if bg is not None:
                Z = Z + bg
            Z_trials.append(Z)                    # (t_i, H)

        if not Z_trials:
            # no usable timesteps after warm-up+mask
            m = np.full(H, np.nan)
            s = np.full(H, np.nan)
            meanG.append(m); stdG.append(s)
            openM.append(np.zeros(H, bool)); closedM.append(np.zeros(H, bool))
            continue

        if normalise:
            Z_all = np.concatenate(Z_trials, axis=0)  # (sum_t, H)
            mu = Z_all.mean(axis=0, keepdims=True)
            sd = Z_all.std(axis=0, keepdims=True) + 1e-8
            Z_trials = [(Z - mu) / sd for Z in Z_trials]

        g_means = [sigmoid(Z).mean(axis=0) for Z in Z_trials]  # list of (H,)
        G = np.stack(g_means, 0)                                # (n_trials', H)
        m = G.mean(axis=0); s = G.std(axis=0, ddof=0)
        meanG.append(m); stdG.append(s)

        # open/closed masks based on mean
        if open_rule[0] == "thr":
            open_mask = m >= float(open_rule[1])
        else:
            open_mask = m >= np.quantile(m, float(open_rule[1]))
        if closed_rule[0] == "thr":
            closed_mask = m <= float(closed_rule[1])
        else:
            closed_mask = m <= np.quantile(m, float(closed_rule[1]))
        closed_mask = np.logical_and(closed_mask, ~open_mask)

        openM.append(open_mask.astype(bool))
        closedM.append(closed_mask.astype(bool))

    return ctx_ids, np.stack(meanG, 0), np.stack(stdG, 0), np.stack(openM, 0), np.stack(closedM, 0)


def plot_gate_states_heatmap(ctx_ids, meanG, stdG, openM, closedM,
                             top_cols=128, ctx_labels=None,
                             figsize=(12,4), title=None,
                             show_numbers=False, colour="ternary",
                             num_fmt="{:.2f}", std_fmt="{:.2f}"):
    """
    If show_numbers=True, each cell shows:
        <mean>
        ± <std>
    """
    n_ctx, H = meanG.shape
    labels = ctx_labels if ctx_labels is not None else [str(c) for c in ctx_ids]

    # pick informative columns
    col_var = meanG.var(axis=0)
    cols = np.argsort(col_var)[::-1][:min(top_cols, H)]

    # ternary: -1 closed, 0 middle, +1 open
    cat = np.zeros((n_ctx, H), dtype=int)
    cat[closedM] = -1
    cat[openM]   =  1
    Mshow_ternary = cat[:, cols]
    Mshow_values  = meanG[:, cols]
    Mshow_std     = stdG[:, cols]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if colour == "ternary":
        cmap = ListedColormap(["#6a3d9a", "#bdbdbd", "#ffd92f"])  # closed / middle / open
        im = ax.imshow(Mshow_ternary, aspect='auto', cmap=cmap, vmin=-1.5, vmax=1.5)
    elif colour == "heatmap":
        # colour by meanG values
        vmin, vmax = np.min(Mshow_values), np.max(Mshow_values)
        im = ax.imshow(Mshow_values, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    else:
        raise ValueError("colour must be 'ternary' or 'heatmap'")

    ax.set_title(title)
    ax.set_xlabel(f"Channels (top {Mshow_values.shape[1]} by variance)")
    ax.set_ylabel("Context")
    ax.set_yticks(range(n_ctx))
    ax.set_yticklabels(labels)

    if show_numbers:
        for i in range(Mshow_values.shape[0]):
            for j in range(Mshow_values.shape[1]):
                ax.text(j, i,
                        f"{num_fmt.format(Mshow_values[i, j])}\n± {std_fmt.format(Mshow_std[i, j])}",
                        ha='center', va='center', fontsize=6, color='black', linespacing=0.9)

    fig.tight_layout()
    return fig, ax

def plot_gate_states_heatmap_stacked(ctx_ids_list, meanG_list, stdG_list, openM_list, closedM_list,
                             top_cols=128, ctx_labels=None,
                             figsize=None, title=None, subplot_titles=None,
                             show_numbers=False, colour="ternary",
                             num_fmt="{:.2f}", std_fmt="{:.2f}"):
    """
    Stacked heatmaps. The first five arguments are lists of length n:
      - ctx_ids_list[i]: iterable of context identifiers for plot i
      - meanG_list[i]: (n_ctx_i, H) array of means
      - stdG_list[i]:  (n_ctx_i, H) array of stds
      - openM_list[i], closedM_list[i]: boolean masks or index arrays broadcastable to (n_ctx_i, H)

    If show_numbers=True, each cell shows:
        <mean>
        ± <std>
    """
    # Back-compat: allow single (non-list) inputs
    def _ensure_list(x):
        return x if isinstance(x, (list, tuple)) else [x]

    ctx_ids_list  = _ensure_list(ctx_ids_list)
    meanG_list    = _ensure_list(meanG_list)
    stdG_list     = _ensure_list(stdG_list)
    openM_list    = _ensure_list(openM_list)
    closedM_list  = _ensure_list(closedM_list)

    n = len(meanG_list)
    assert all(len(lst) == n for lst in [ctx_ids_list, stdG_list, openM_list, closedM_list]), \
        "First five arguments must all be lists of the same length."

    # ctx_labels can be:
    # - None -> derive from ctx_ids for each subplot
    # - list of lists (len n)
    # - single list to be reused for all (if shapes match)
    if ctx_labels is None:
        ctx_labels_list = [None] * n
    elif isinstance(ctx_labels, list) and len(ctx_labels) == n and any(isinstance(x, list) for x in ctx_labels):
        ctx_labels_list = ctx_labels
    else:
        # single list applied to all
        ctx_labels_list = [ctx_labels for _ in range(n)]

    # subplot titles
    if subplot_titles is None:
        subplot_titles = [None] * n
    else:
        assert len(subplot_titles) == n, "subplot_titles must be a list of length n."

    # Figure size: scale by number of subplots if not provided
    if figsize is None:
        figsize = (12, max(3.5 * n, 3.5))

    fig, axes = plt.subplots(n, 1, figsize=figsize, squeeze=False, sharex=False)
    axes = axes.ravel()

    for idx in range(n):
        ctx_ids = ctx_ids_list[idx]
        meanG   = np.asarray(meanG_list[idx])
        stdG    = np.asarray(stdG_list[idx])
        openM   = openM_list[idx]
        closedM = closedM_list[idx]

        n_ctx, H = meanG.shape
        labels = ctx_labels_list[idx]
        if labels is None:
            labels = [str(c) for c in ctx_ids]

        # pick informative columns per-plot
        col_var = meanG.var(axis=0)
        cols = np.argsort(col_var)[::-1][:min(top_cols, H)]

        # ternary: -1 closed, 0 middle, +1 open
        cat = np.zeros((n_ctx, H), dtype=int)
        cat[closedM] = -1
        cat[openM]   =  1
        Mshow_ternary = cat[:, cols]
        Mshow_values  = meanG[:, cols]
        Mshow_std     = stdG[:, cols]

        ax = axes[idx]

        if colour == "ternary":
            cmap = ListedColormap(["#6a3d9a", "#bdbdbd", "#ffd92f"])  # closed / middle / open
            im = ax.imshow(Mshow_ternary, aspect='auto', cmap=cmap, vmin=-1.5, vmax=1.5)
        elif colour == "heatmap":
            vmin, vmax = np.min(Mshow_values), np.max(Mshow_values)
            im = ax.imshow(Mshow_values, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        else:
            raise ValueError("colour must be 'ternary' or 'heatmap'")

        if subplot_titles[idx]:
            ax.set_title(subplot_titles[idx])

        ax.set_ylabel("Context")
        ax.set_yticks(range(n_ctx))
        ax.set_yticklabels(labels)

        ax.set_xlabel(f"Channels (top {Mshow_values.shape[1]} by variance)")

        if show_numbers:
            for i in range(Mshow_values.shape[0]):
                for j in range(Mshow_values.shape[1]):
                    ax.text(j, i,
                            f"{num_fmt.format(Mshow_values[i, j])}\n± {std_fmt.format(Mshow_std[i, j])}",
                            ha='center', va='center', fontsize=6, color='black', linespacing=0.9)

        ax.grid(False)

        # add a colourbar per subplot for clarity
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if colour == "ternary":
            cbar.set_ticks([-1, 0, 1])
            cbar.set_ticklabels(["Closed", "Middle", "Open"])
        else:
            cbar.set_label("Mean")

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()

    return fig, axes

def plot_gate_cosine(ctx_ids, meanG, ctx_labels=None,
                     title="Cosine similarity of mean gates", figsize=(6, 5),
                     annotate=True, fmt="{:.2f}", hide_diag=False, eps=1e-12):
    """
    Cosine-similarity heatmap with numeric values in cells.

    meanG: array (n_ctx, H) of per-context mean gate values.
    ctx_labels: None, list[str] of length n_ctx, or dict[int]->str.
    """
    M = np.asarray(meanG, dtype=float)                 # (n_ctx, H)
    n_ctx = M.shape[0]

    # labels
    if ctx_labels is None:
        labels = [str(c) for c in ctx_ids]
    elif isinstance(ctx_labels, dict):
        labels = [ctx_labels.get(c, str(c)) for c in ctx_ids]
    else:
        labels = list(ctx_labels)

    # cosine similarity: normalise rows then dot
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    U = M / norms
    S = U @ U.T
    S = np.clip(S, 0.0, 1.0)  # gates are nonnegative; keep in [0,1] for display

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(S, vmin=0, vmax=1, aspect='equal')
    ax.set_title(title)
    ax.set_xticks(range(n_ctx)); ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(n_ctx)); ax.set_yticklabels(labels)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine")

    if annotate:
        thresh = 0.6
        for i in range(n_ctx):
            for j in range(n_ctx):
                if hide_diag and i == j:
                    continue
                val = S[i, j]
                colour = "white" if val >= thresh else "black"
                ax.text(j, i, fmt.format(val), ha='center', va='center',
                        color=colour, fontsize=9)

    fig.tight_layout()
    return fig, ax

def plot_gate_cosine_stacked(ctx_ids_list, meanG_list, ctx_labels=None,
                     title="Cosine similarity of mean gates",
                     subplot_titles=None, figsize=None,
                     annotate=True, fmt="{:.2f}", hide_diag=False, eps=1e-12):
    """
    Cosine-similarity heatmaps in a row.

    The first two args are lists of length n:
      - ctx_ids_list[i]: iterable of context identifiers
      - meanG_list[i]: (n_ctx, H) array of per-context mean gate values

    ctx_labels:
      - None: labels from ctx_ids
      - list of length n: each entry can be list[str] or dict[int]->str for that subplot
      - single list[str] applied to all subplots

    subplot_titles: list of strings for each subplot
    """
    def _ensure_list(x):
        return x if isinstance(x, (list, tuple)) else [x]

    ctx_ids_list = _ensure_list(ctx_ids_list)
    meanG_list   = _ensure_list(meanG_list)
    n = len(meanG_list)
    assert len(ctx_ids_list) == n, "ctx_ids_list and meanG_list must have the same length."

    # ctx_labels handling
    if ctx_labels is None:
        ctx_labels_list = [None] * n
    elif isinstance(ctx_labels, list) and len(ctx_labels) == n and any(isinstance(x, (list, dict)) for x in ctx_labels):
        ctx_labels_list = ctx_labels
    else:
        # single labels list/dict reused for all
        ctx_labels_list = [ctx_labels for _ in range(n)]

    # subplot titles
    if subplot_titles is None:
        subplot_titles = [None] * n
    else:
        assert len(subplot_titles) == n, "subplot_titles must be a list of length n."

    # default figsize: scale by number of subplots
    if figsize is None:
        figsize = (5 * n, 5)

    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False, sharey=False)
    axes = axes.ravel()

    for idx in range(n):
        ctx_ids = ctx_ids_list[idx]
        M = np.asarray(meanG_list[idx], dtype=float)
        n_ctx = M.shape[0]

        # labels
        if ctx_labels_list[idx] is None:
            labels = [str(c) for c in ctx_ids]
        elif isinstance(ctx_labels_list[idx], dict):
            labels = [ctx_labels_list[idx].get(c, str(c)) for c in ctx_ids]
        else:
            labels = list(ctx_labels_list[idx])

        # cosine similarity
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        U = M / norms
        S = U @ U.T
        S = np.clip(S, 0.0, 1.0)

        ax = axes[idx]
        im = ax.imshow(S, vmin=0, vmax=1, aspect='equal')
        if subplot_titles[idx]:
            ax.set_title(subplot_titles[idx])
        ax.set_xticks(range(n_ctx)); ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticks(range(n_ctx)); ax.set_yticklabels(labels)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine")

        if annotate:
            thresh = 0.6
            for i in range(n_ctx):
                for j in range(n_ctx):
                    if hide_diag and i == j:
                        continue
                    val = S[i, j]
                    colour = "white" if val >= thresh else "black"
                    ax.text(j, i, fmt.format(val), ha='center', va='center',
                            color=colour, fontsize=9)

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()

    return fig, axes


### Eigenvalues

def plot_eig_spectra_grid(untrained, trained, row_labels, point_size=12):
    """
    Plot eigenvalue spectra (complex plane) for paired lists of weight matrices.

    Parameters
    ----------
    untrained : list of array-like
        List of square weight matrices (e.g., numpy arrays or torch tensors).
    trained : list of array-like
        List of square weight matrices aligned with `untrained`.
    row_labels : list of str
        Row labels, same length as the lists.
    point_size : int, optional
        Marker size for eigenvalue scatter points.

    Returns
    -------
    fig, axes : matplotlib Figure and Axes array
    """
    if len(untrained) != len(trained):
        raise ValueError("`untrained` and `trained` must be the same length.")
    if len(row_labels) != len(untrained):
        raise ValueError("`row_labels` length must match the lists.")

    def _to_numpy(a):
        try:
            import torch
            if isinstance(a, torch.Tensor):
                a = a.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(a)

    n_rows = len(untrained)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(10, 4 * n_rows), constrained_layout=True
    )
    if n_rows == 1:
        axes = np.array([axes])

    axes[0, 0].set_title("Untrained")
    axes[0, 1].set_title("Trained")

    theta = np.linspace(0, 2 * np.pi, 400)
    unit_circle = np.exp(1j * theta)

    for i, (U, T) in enumerate(zip(untrained, trained)):
        for j, M in enumerate([U, T]):
            ax = axes[i, j]
            M = _to_numpy(M)
            vals = np.linalg.eigvals(M)
            ax.scatter(vals.real, vals.imag, s=point_size)
            ax.plot(unit_circle.real, unit_circle.imag, 'k--', lw=1)
            ax.set_aspect("equal", adjustable="box")
            ax.axhline(0, lw=0.8, alpha=0.5)
            ax.axvline(0, lw=0.8, alpha=0.5)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            if i == n_rows - 1:
                ax.set_xlabel("Real")
            if j == 0:
                ax.set_ylabel("Imag")

        axes[i, 0].set_ylabel(f"{row_labels[i]}\nImag", rotation=90, labelpad=12)

    return fig, axes

def get_A_matrix(model, block_id):
  lambdas = model.ssm_blocks[block_id].ssm.Lambda_re + 1j *model.ssm_blocks[block_id].ssm.Lambda_im
  return np.diag(np.array(lambdas))

def get_discretized_ssm_parameters(ssm):
    Lambda = ssm.Lambda_re + 1j * ssm.Lambda_im
    B_tilde = ssm.B[..., 0] + 1j * ssm.B[..., 1]
    C_tilde = ssm.C[..., 0] + 1j * ssm.C[..., 1]

    Delta = ssm.step_rescale * jnp.exp(ssm.log_step[:, 0])
    Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, Delta)
    Lambda_bar = jnp.concat([Lambda_bar, jnp.conj(Lambda_bar)])
    # print(B_tilde, B_bar)

    return Lambda_bar, B_bar, C_tilde, Delta

def get_discrete_A_matrix(model, block_id):
    """
    Discretise block.ssm via ZOH, returning (A_bar, B_bar).
    A_bar: diag of real(Lambda_bar) shape (P, P)
    B_bar: real part of B_bar shape (P, H)
    """
    ssm = model.ssm_blocks[block_id].ssm
    Lambda_bar, B_bar, C_tilde, Delta = get_discretized_ssm_parameters(ssm)
    A_bar = jnp.diag(Lambda_bar)
    return A_bar

### Encoders Similarity

def compare_encoders_cosine_similarity(
    model, top_k=10, scaled=True, title="", context_at_end=True,
    include_context=False, encoder_labels=None, encoder_idx=None
):
    """
    For each (optionally selected) encoder in model.encoders, compute sim1:
      (encoder · B₀, left singular vectors) vs (A₀, right singular vectors).

    Context handling:
      - Uses model.context_embedding.embedding_size (or .weight.shape[1]) as ctx_dim.
      - Trims exactly ctx_dim columns from B₀ (end by default; set context_at_end=False to trim from start).
      - If include_context=True, also adds bias from context columns.

    Plotting:
      - Two columns of heatmaps.
      - Square aspect for cells.
      - Shared colour scale.
      - Optional custom encoder_labels, which will also be filtered if encoder_idx is set.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from math import ceil

    EPS = 1e-12

    def get_A_and_B(ssm):
        Lambda = np.array(ssm.Lambda_re) + 1j * np.array(ssm.Lambda_im)
        A = np.diag(Lambda)
        B = np.array(ssm.B[..., 0]) + 1j * np.array(ssm.B[..., 1])
        return A, B

    def compute_cosine_similarity(M1, M2, left_vectors_M1=True, left_vectors_M2=False, scaled=True):
        U1, s1, Vh1 = np.linalg.svd(M1, full_matrices=False)
        U2, s2, Vh2 = np.linalg.svd(M2, full_matrices=False)

        vecs1 = U1[:, :top_k] if left_vectors_M1 else Vh1.conj().T[:, :top_k]
        vecs2 = U2[:, :top_k] if left_vectors_M2 else Vh2.conj().T[:, :top_k]

        vecs1 = vecs1 / (np.linalg.norm(vecs1, axis=0, keepdims=True) + EPS)
        vecs2 = vecs2 / (np.linalg.norm(vecs2, axis=0, keepdims=True) + EPS)

        cs = np.abs(vecs1.conj().T @ vecs2)

        if scaled:
            s1n = s1[:top_k] / (np.max(s1[:top_k]) + EPS)
            s2n = s2[:top_k] / (np.max(s2[:top_k]) + EPS)
            sim = cs * np.outer(s1n, s2n)
            sim = sim / (np.max(sim) + EPS)
        else:
            sim = cs
        return sim

    # Pick encoder indices
    all_indices = list(range(len(model.encoders)))
    if encoder_idx is None:
        selected_idx = all_indices
    else:
        selected_idx = list(encoder_idx)
        for i in selected_idx:
            if i not in all_indices:
                raise ValueError(f"Invalid encoder index: {i}")

    # Filter encoder_labels if provided
    if encoder_labels is not None:
        if len(encoder_labels) != len(model.encoders):
            raise ValueError(
                f"encoder_labels length {len(encoder_labels)} does not match number of encoders {len(model.encoders)}."
            )
        encoder_labels = [encoder_labels[i] for i in selected_idx]

    # Block 0
    A0, B0 = get_A_and_B(model.ssm_blocks[0].ssm)
    B0_real, B0_imag = B0.real, B0.imag
    B_width = B0_real.shape[1]

    # Read ctx_dim
    try:
        ctx_dim = int(getattr(model.context_embedding, "embedding_size"))
    except Exception:
        try:
            ctx_dim = int(model.context_embedding.weight.shape[1])
        except Exception:
            raise ValueError("Could not determine context_embedding size (embedding_size / weight.shape[1]).")

    if ctx_dim < 0 or ctx_dim > B_width:
        raise ValueError(f"Invalid context size: ctx_dim={ctx_dim}, B_width={B_width}")

    # Trim context columns from B₀
    if context_at_end:
        b_slice = slice(0, B_width - ctx_dim)
        ctx_slice = slice(B_width - ctx_dim, B_width)
    else:
        b_slice = slice(ctx_dim, B_width)
        ctx_slice = slice(0, ctx_dim)

    B0_input = (B0_real[:, b_slice] + 1j * B0_imag[:, b_slice])
    in_features_expected = B0_input.shape[1]

    similarities, enc_names = [], []

    for idx_pos, idx in enumerate(selected_idx):
        enc = model.encoders[idx]
        W = np.array(getattr(enc, "weight"))

        if in_features_expected == W.shape[0]:
            W_use = W
        elif in_features_expected == W.shape[1]:
            W_use = W.T
        else:
            raise ValueError(
                f"Encoder {idx}: weight shape {W.shape} incompatible with B0_input width {in_features_expected}."
            )

        B0_encoded = B0_input @ W_use

        if include_context:
            B_ctx = B0_real[:, ctx_slice] + 1j * B0_imag[:, ctx_slice]  # (64, ctx_dim)
            context_vec = np.array(model.context_embedding.weight[idx])  # (ctx_dim,)
            bias_term = (B_ctx @ context_vec).reshape(-1, 1)  # (64, 1)
            B0_encoded = B0_encoded + bias_term

        sim = compute_cosine_similarity(B0_encoded, A0, left_vectors_M1=True, left_vectors_M2=False, scaled=scaled)
        similarities.append(sim)

        if encoder_labels is not None:
            name = encoder_labels[idx_pos]
        else:
            name = getattr(enc, "name", None) or enc.__class__.__name__
            name = f"Encoder {idx}: {name}"

        enc_names.append(name)

    # Plot in two columns
    n_enc = len(similarities)
    n_cols = 2
    n_rows = ceil(n_enc / n_cols)

    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize=(top_k * 0.6 * n_cols + 2, n_rows * (top_k * 0.6 + 1)),
        constrained_layout=True
    )
    axs = np.atleast_2d(axs)

    vmin, vmax = 0.0, 1.0
    for i, (sim, label) in enumerate(zip(similarities, enc_names)):
        r, c = divmod(i, n_cols)
        ax = axs[r, c]
        im = ax.imshow(sim, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel("Right singular vectors (A₀)")
        ax.set_ylabel("Left singular vectors (E·B₀)")
        ax.set_xticks(range(top_k))
        ax.set_yticks(range(top_k))
        ax.set_xticklabels([f"RSV {j+1}" for j in range(top_k)], rotation=45)
        ax.set_yticklabels([f"LSV {j+1}" for j in range(top_k)])

    # Hide unused subplots
    for j in range(len(similarities), n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axs[r, c].axis("off")

    # Shared colourbar
    cbar = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
    cbar.set_label("Cosine similarity" + (" (scaled)" if scaled else ""))

    if title:
        fig.suptitle(title)
    plt.show()

    return similarities




def compare_encoders_cosine_similarity(
    model, untrained_model=None, top_k=10, scaled=True, title="",
    context_at_end=True, include_context=False,
    encoder_labels=None, encoder_idx=None
):
    """
    For each (optionally selected) encoder in `model.encoders`, compute sim:
      (encoder · B₀, left singular vectors) vs (A₀, right singular vectors).

    If `untrained_model` is provided, the first encoder from it is also
    included and plotted first as "Untrained Baseline".

    Context handling:
      - Uses model.context_embedding.embedding_size (or .weight.shape[1]) as ctx_dim.
      - Trims exactly ctx_dim columns from B₀ (end by default; set context_at_end=False to trim from start).
      - If include_context=True, also adds bias from context columns.

    Plotting:
      - Heatmaps with square aspect.
      - Shared colour scale.
      - "Untrained Baseline" always appears first if untrained_model is given.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from math import ceil

    EPS = 1e-12

    def get_A_and_B(ssm):
        Lambda = np.array(ssm.Lambda_re) + 1j * np.array(ssm.Lambda_im)
        A = np.diag(Lambda)
        B = np.array(ssm.B[..., 0]) + 1j * np.array(ssm.B[..., 1])
        return A, B

    def compute_cosine_similarity(M1, M2, left_vectors_M1=True, left_vectors_M2=False, scaled=True):
        U1, s1, Vh1 = np.linalg.svd(M1, full_matrices=False)
        U2, s2, Vh2 = np.linalg.svd(M2, full_matrices=False)

        vecs1 = U1[:, :top_k] if left_vectors_M1 else Vh1.conj().T[:, :top_k]
        vecs2 = U2[:, :top_k] if left_vectors_M2 else Vh2.conj().T[:, :top_k]

        vecs1 = vecs1 / (np.linalg.norm(vecs1, axis=0, keepdims=True) + EPS)
        vecs2 = vecs2 / (np.linalg.norm(vecs2, axis=0, keepdims=True) + EPS)

        cs = np.abs(vecs1.conj().T @ vecs2)
        #cs = np.abs(vecs2.conj().T @ vecs1)

        if scaled:
            # normalise SVs by each matrix's max SV (top singular value)
            s1n = s1[:top_k] / (np.max(s1) + EPS)
            s2n = s2[:top_k] / (np.max(s2) + EPS)
            sim = cs * np.outer(s1n, s2n)
        else:
            sim = cs

        return sim

    def prepare(model, encoder_idx):
        # Block 0
        A0, B0 = get_A_and_B(model.ssm_blocks[0].ssm)
        B0_real, B0_imag = B0.real, B0.imag
        B_width = B0_real.shape[1]

        # Read ctx_dim
        try:
            ctx_dim = int(getattr(model.context_embedding, "embedding_size"))
        except Exception:
            try:
                ctx_dim = int(model.context_embedding.weight.shape[1])
            except Exception:
                #raise ValueError("Could not determine context_embedding size.")
                ctx_dim=0

        if ctx_dim < 0 or ctx_dim > B_width:
            raise ValueError(f"Invalid context size: ctx_dim={ctx_dim}, B_width={B_width}")

        # Trim context columns from B₀
        if context_at_end:
            b_slice = slice(0, B_width - ctx_dim)
            ctx_slice = slice(B_width - ctx_dim, B_width)
        else:
            b_slice = slice(ctx_dim, B_width)
            ctx_slice = slice(0, ctx_dim)

        B0_input = (B0_real[:, b_slice] + 1j * B0_imag[:, b_slice])
        return A0, B0_input, B0_real, B0_imag, ctx_slice, ctx_dim

    # Select encoders for trained model
    all_indices = list(range(len(model.encoders)))
    if encoder_idx is None:
        selected_idx = all_indices
    else:
        selected_idx = list(encoder_idx)

    if encoder_labels is not None:
        if len(encoder_labels) != len(model.encoders):
            raise ValueError("encoder_labels length mismatch.")
        encoder_labels = [encoder_labels[i] for i in selected_idx]

    # Prepare trained model
    A0, B0_input, B0_real, B0_imag, ctx_slice, ctx_dim = prepare(model, selected_idx)
    in_features_expected = B0_input.shape[1]

    similarities, enc_names = [], []

    # Optionally add untrained baseline first
    if untrained_model is not None:
        A0_u, B0_input_u, B0_real_u, B0_imag_u, ctx_slice_u, ctx_dim_u = prepare(untrained_model, [0])
        in_features_expected_u = B0_input_u.shape[1]
        enc_u = untrained_model.encoders[0]
        W_u = np.array(getattr(enc_u, "weight"))
        if in_features_expected_u == W_u.shape[0]:
            W_use_u = W_u
        elif in_features_expected_u == W_u.shape[1]:
            W_use_u = W_u.T
        else:
            raise ValueError("Untrained encoder weight incompatible.")
        B0_encoded_u = B0_input_u @ W_use_u
        if include_context:
            B_ctx_u = B0_real_u[:, ctx_slice_u] + 1j * B0_imag_u[:, ctx_slice_u]
            context_vec_u = np.array(untrained_model.context_embedding.weight[0])
            B0_encoded_u = B0_encoded_u + (B_ctx_u @ context_vec_u).reshape(-1, 1)
        sim_u = compute_cosine_similarity(B0_encoded_u, A0_u, scaled=scaled)
        similarities.append(sim_u)
        enc_names.append("Untrained Baseline")

    # Trained encoders
    for idx_pos, idx in enumerate(selected_idx):
        enc = model.encoders[idx]
        W = np.array(getattr(enc, "weight"))
        if in_features_expected == W.shape[0]:
            W_use = W
        elif in_features_expected == W.shape[1]:
            W_use = W.T
        else:
            raise ValueError("Encoder weight incompatible with B0_input.")
        B0_encoded = B0_input @ W_use
        if include_context:
            B_ctx = B0_real[:, ctx_slice] + 1j * B0_imag[:, ctx_slice]
            context_vec = np.array(model.context_embedding.weight[idx])
            B0_encoded = B0_encoded + (B_ctx @ context_vec).reshape(-1, 1)
        sim = compute_cosine_similarity(B0_encoded, A0, scaled=scaled)
        similarities.append(sim)
        if encoder_labels is not None:
            name = encoder_labels[idx_pos]
        else:
            name = getattr(enc, "name", None) or enc.__class__.__name__
            name = f"Encoder {idx}: {name}"
        enc_names.append(name)

    # Plot
    n_enc = len(similarities)
    n_cols = 2
    n_rows = ceil(n_enc / n_cols)
    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize=(top_k * 0.6 * n_cols + 2, n_rows * (top_k * 0.6 + 1)),
        constrained_layout=True
    )
    axs = np.atleast_2d(axs)
    vmin, vmax = 0.0, 1.0

    for i, (sim, label) in enumerate(zip(similarities, enc_names)):
        r, c = divmod(i, n_cols)
        ax = axs[r, c]
        im = ax.imshow(sim, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel("Right singular vectors (A)")
        ax.set_ylabel("Left singular vectors (E·B)")
        ax.set_xticks(range(top_k))
        ax.set_yticks(range(top_k))
        ax.set_xticklabels([f"RSV {j+1}" for j in range(top_k)], rotation=45)
        ax.set_yticklabels([f"LSV {j+1}" for j in range(top_k)])

    # Hide unused subplots
    for j in range(len(similarities), n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axs[r, c].axis("off")

    # Shared colourbar
    cbar = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
    cbar.set_label("Cosine similarity" + (" (scaled)" if scaled else ""))

    if title:
        fig.suptitle(title)
    plt.show()

    return similarities

def plot_similarity_of_similarities(similarities, labels=None, title="Similarity of Similarities"):
    """
    Given a list of similarity matrices (e.g. from compare_encoders_cosine_similarity),
    compute pairwise cosine similarities between their flattened entries and show as a heatmap.

    Args:
        similarities: list of (k x k) numpy arrays
        labels: optional list of labels, length = len(similarities)
        title: figure title
    """
    import numpy as np
    import matplotlib.pyplot as plt

    sims = []
    for M in similarities:
        v = M.flatten().astype(float)
        v = v / (np.linalg.norm(v) + 1e-12)
        sims.append(v)
    sims = np.stack(sims)

    # Cosine similarity matrix
    cosmat = sims @ sims.T

    n = cosmat.shape[0]
    if labels is None:
        labels = [f"Enc {i}" for i in range(n)]

    fig, ax = plt.subplots(figsize=(0.6*n+2, 0.6*n+2))
    im = ax.imshow(cosmat, cmap="viridis", vmin=0, vmax=1)

    # Tick labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Annotate with values
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cosmat[i,j]:.2f}", ha="center", va="center",
                    color="white" if cosmat[i,j] < 0.5 else "black", fontsize=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity between similarity matrices")

    plt.title(title)
    plt.tight_layout()
    plt.show()

    return cosmat


