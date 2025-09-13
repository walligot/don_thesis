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
import sys
import os
import jax
from tqdm import tqdm
import gzip
import pickle

# map: dataset_idx -> (dataset_name, subject, task)
from foundational_ssm.constants import DATASET_IDX_TO_GROUP, DATASET_IDX_TO_GROUP_SHORT

if 'google.colab' in sys.modules:
    pio.renderers.default = 'colab'


### Procrustes

def _pad_last_dim(X, target_dim):
    pad_width = target_dim - X.shape[2]
    if pad_width < 0:
        return X[..., :target_dim]
    return np.pad(X, ((0, 0), (0, 0), (0, pad_width)))

def _pad_to_match_dims(A, B):
    target_dim = max(A.shape[2], B.shape[2])
    return _pad_last_dim(A, target_dim), _pad_last_dim(B, target_dim)

def _shuffle_timesteps(X, inds):
    return np.concatenate([x[inds] for x in X], axis=0)

def _shuffle_trials(X, inds):
    return X[inds].reshape(-1, X.shape[2])

def _shuffle_all(X, inds):
    X_flat = X.reshape(-1, X.shape[2])
    return X_flat[inds]

def _prepare_concat_single(X, mode, shuffled_inds):
    if mode == 'shuffle_timesteps':
        return _shuffle_timesteps(X, shuffled_inds)
    elif mode == 'shuffle_trials':
        return _shuffle_trials(X, shuffled_inds)
    elif mode == 'shuffle_all':
        return _shuffle_all(X, shuffled_inds)
    elif mode == 'concat':
        return X.reshape(-1, X.shape[2])
    else:
        raise ValueError(f"Unknown mode: {mode}")

def _apply_pca_shared(A_concat, B_concat, pca_dim):
    A_pca = PCA(n_components=pca_dim).fit_transform(A_concat)
    B_pca = PCA(n_components=pca_dim).fit_transform(B_concat)
    return A_pca, B_pca

def _apply_pca_trialwise(A, B, pca_dim):
    A_pca = np.stack([PCA(n_components=pca_dim).fit_transform(a) for a in A])
    B_pca = np.stack([PCA(n_components=pca_dim).fit_transform(b) for b in B])
    return A_pca, B_pca

def batched_procrustes_distance(X, Y, normalize=True, mode='concat', pca_dim=None, shuffled_inds=None):
    X_list = X if isinstance(X, list) else [X]
    Y_list = Y if isinstance(Y, list) else [Y]

    # Ensure all inputs are 3D
    X_list = [x[None, ...] if x.ndim == 2 else x for x in X_list]
    Y_list = [y[None, ...] if y.ndim == 2 else y for y in Y_list]

    # Check n_trials and n_timesteps match
    shape_ref = X_list[0].shape[:2] if X_list else Y_list[0].shape[:2]
    for tensor in X_list + Y_list:
        if tensor.shape[:2] != shape_ref:
            raise ValueError("Mismatch in n_trials or n_timesteps")

    # Decide whether PCA is needed
    all_dims = [x.shape[2] for x in X_list + Y_list]
    dims_match = all(d == all_dims[0] for d in all_dims)
    apply_pca = pca_dim is not None or not dims_match

    if apply_pca:
        if pca_dim is None:
            pca_dim = int(0.8 * min(all_dims))
            print(f"[pca] Auto PCA: projecting to {pca_dim} dims (hidden dims differ)")
        if pca_dim <= 0:
            raise ValueError("pca_dim must be positive")

    # Generate shuffle indices if needed
    if shuffled_inds is None and mode in {'shuffle_timesteps', 'shuffle_trials', 'shuffle_all'}:
        n_trials, n_timesteps = shape_ref
        if mode == 'shuffle_timesteps':
            shuffled_inds = np.random.permutation(n_timesteps)
        elif mode == 'shuffle_trials':
            shuffled_inds = np.random.permutation(n_trials)
        elif mode == 'shuffle_all':
            shuffled_inds = np.random.permutation(n_trials * n_timesteps)

    results = []
    for A in X_list:
        row = []
        for B in Y_list:
            # Pad if PCA required
            #A_pad, B_pad = (A, B) if not apply_pca else _pad_to_match_dims(A, B)
            A_pad, B_pad = (A, B)

            if mode == 'mean':
                if apply_pca:
                    A_proc, B_proc = _apply_pca_trialwise(A_pad, B_pad, pca_dim)
                else:
                    A_proc, B_proc = A_pad, B_pad

                score = np.mean([
                    #procrustes_distance(a, b, normalize=normalize)
                    procrustes(a, b)[2]
                    for a, b in zip(A_proc, B_proc)
                ])
                del A_proc, B_proc
            else:
                A_concat = _prepare_concat_single(A_pad, mode, shuffled_inds)
                B_concat = _prepare_concat_single(B_pad, mode, shuffled_inds)

                if apply_pca:
                    A_proc, B_proc = _apply_pca_shared(A_concat, B_concat, pca_dim)
                else:
                    A_proc, B_proc = A_concat, B_concat

                #score = procrustes_distance(A_proc, B_proc, normalize=normalize)
                _, _, score = procrustes(A_proc, B_proc)
                del A_concat, B_concat, A_proc, B_proc

            del A_pad, B_pad
            row.append(score)
        results.append(row)

    return np.array(results)

def align_trajectories(hidden_states):
    """
    Aligns the trajectories within each numpy array in hidden_states to the first trajectory using Procrustes analysis.

    Args:
    - hidden_states (list of np.ndarray): List of numpy arrays where each array has shape (n_timesteps, n_hidden_units).

    Returns:
    - aligned_trajectories (list of np.ndarray): List of numpy arrays containing the aligned trajectories for each model/condition.
    """
    aligned_trajectories = []  # To store the aligned arrays

    # Step 1: Extract the first trajectory's data (baseline for alignment)
    baseline_trajectory = hidden_states[0]  # Shape: (n_timesteps, n_hidden_units)

    # Step 2: Iterate over each array in the list (representing different models/conditions)
    for trajectory in hidden_states:
        # Apply Procrustes analysis to align each trajectory to the baseline
        m1, m2, disparity = procrustes(baseline_trajectory, trajectory)

        # Append the aligned trajectory (m1 is the aligned trajectory) to the list
        aligned_trajectories.append(m2)  # m1 is the aligned trajectory

    return aligned_trajectories


def _ensure_mask3d(m, n_trials, T):
    m = np.asarray(m)
    if m.ndim == 2 and m.shape == (n_trials, T): m = m[..., None]
    if m.ndim != 3 or m.shape != (n_trials, T, 1):
        raise ValueError(f"mask must be (n_trials, T) or (n_trials, T, 1); got {m.shape}")
    return m.astype(bool)

def _angle_bins_from_targets(targets, mask=None, n_bins=8):
    v = np.asarray(targets, dtype=float)  # (n_trials, T, 2)
    if v.ndim != 3 or v.shape[-1] != 2:
        raise ValueError("targets must be (n_trials, n_timesteps, 2)")
    n_trials, T, _ = v.shape
    if mask is not None:
        m = _ensure_mask3d(mask, n_trials, T)
        v = v * m
    pos = np.cumsum(v, axis=1)
    end = pos[:, -1, :]                                  # (n_trials, 2)
    ang = np.degrees(np.arctan2(end[:, 1], end[:, 0]))   # [-180,180]
    shifted = (ang + 180.0) % 360.0
    bin_w = 360.0 / n_bins
    bins = -180.0 + (shifted // bin_w).astype(int) * bin_w
    return bins.astype(float)                             # (n_trials,)

def compute_bin_averages_and_check(
    data,
    components=('post_encoder','ssm_post_glu_0','ssm_post_glu_1'),
    N=100,
    n_bins=8,
    verbose=True,
):
    """
    For each dataset and component:
      - recompute angle_bins from targets+mask,
      - group trials by bin,
      - filter timesteps where mask=False,
      - take first N valid timesteps per trial,
      - average across trials (shape (N,D)),
      - report bins that are missing / insufficient.
    """

    bin_w = 360.0 / n_bins
    bin_order = (-180.0 + bin_w * np.arange(n_bins)).astype(float)

    def angle_bins_from_targets(targets, mask, n_bins):
        v = np.asarray(targets, dtype=float)       # (n_trials,T,2)
        n_trials, T, _ = v.shape
        m = np.asarray(mask, dtype=bool)
        if m.ndim == 2: m = m[..., None]
        v = v * m
        pos = np.cumsum(v, axis=1)
        end = pos[:, -1, :]
        ang = np.degrees(np.arctan2(end[:, 1], end[:, 0]))
        shifted = (ang + 180.0) % 360.0
        return -180.0 + (shifted // bin_w).astype(int) * bin_w

    averages, status = {}, {}
    for k, entry in data.items():
        # shapes
        H0 = np.asarray(entry[components[0]])
        n_trials, T, _ = H0.shape
        M = np.asarray(entry['mask'], dtype=bool)
        if M.ndim == 2: M = M[..., None]
        if M.shape != (n_trials, T, 1):
            raise ValueError(f"{k}: mask shape {M.shape} mismatch with {H0.shape}")

        # recompute bins
        angle_bins = angle_bins_from_targets(entry['targets'], M, n_bins)

        averages[k], missing_bins, invalid_bins = {}, [], []
        for comp in components:
            H = np.asarray(entry[comp])
            if H.shape[:2] != (n_trials, T):
                raise ValueError(f"{k}:{comp} shape mismatch {H.shape}")
            D = H.shape[-1]
            averages[k][comp] = {}

            for b in bin_order:
                idx = np.where(angle_bins == b)[0]
                if idx.size == 0:
                    missing_bins.append(b); continue

                valid_trials = []
                for t in idx:
                    valid_idx = np.where(M[t, :, 0])[0]
                    if valid_idx.size >= N:
                        keep = valid_idx[-N:]
                        valid_trials.append(H[t, keep, :])  # (N,D)

                if not valid_trials:
                    invalid_bins.append(b); continue

                Hb = np.stack(valid_trials, axis=0)   # (n_valid, N, D)
                avg = Hb.mean(axis=0)                 # (N,D)
                averages[k][comp][b] = avg

        status[k] = dict(
            missing_bins=sorted(set(missing_bins)),
            invalid_bins=sorted(set(invalid_bins)),
            all_good=not missing_bins and not invalid_bins,
        )

    if verbose:
        for k, st in status.items():
            if st['missing_bins']:
                txt = ", ".join(f"{int(b)}°" for b in st['missing_bins'])
                print(f"[{k}] ERROR: missing bins {txt}")
            if st['invalid_bins']:
                txt = ", ".join(f"{int(b)}°" for b in st['invalid_bins'])
                print(f"[{k}] ERROR: bins lacking {N} valid timesteps: {txt}")
            if st['all_good']:
                print(f"[{k}] OK: all bins present with ≥{N} valid timesteps.")

    return averages, status

def compute_pairwise_procrustes(
    data,
    components=('post_encoder','ssm_post_glu_0','ssm_post_glu_1'),
    N=100,
    n_bins=8,
    data_keys=None,
):
    """
    Compute pairwise Procrustes similarity matrices for multiple datasets.

    Parameters
    ----------
    data : dict
        Datasets keyed by name. Each entry must contain:
          - components (n_trials, T, D)
          - 'targets' (n_trials, T, 2)
          - 'mask' (n_trials, T, 1) or (n_trials, T)
    components : tuple of str
        Components to process.
    N : int
        Number of timesteps to keep from start (mask applied before).
    n_bins : int
        Number of angle bins.
    data_keys : list of str or None
        If provided, restrict to these dataset keys. Otherwise, use all.

    Returns
    -------
    results : dict
        results[component] = (similarity_matrix, dataset_keys)
    status : dict
        From compute_bin_averages_and_check
    """
    # --- Step 1: compute bin averages per dataset ---
    averages, status = compute_bin_averages_and_check(
        data, components=components, N=N, n_bins=n_bins, verbose=False
    )

    ds_keys = list(data.keys()) if data_keys is None else list(data_keys)
    n_datasets = len(ds_keys)

    results = {}
    for comp in tqdm(components, desc="Components"):
        mat = np.full((n_datasets, n_datasets), np.nan, dtype=float)
        for i, ki in enumerate(ds_keys):
            for j, kj in enumerate(ds_keys):
                if j < i:
                    mat[i, j] = mat[j, i]
                    continue

                bins_i = set(averages[ki][comp].keys())
                bins_j = set(averages[kj][comp].keys())
                common_bins = sorted(bins_i & bins_j)

                if not common_bins:
                    continue  # no overlap → leave NaN

                # build list of (n_bins_total, N, D)
                vecs = []
                for ds in (averages[ki], averages[kj]):
                    parts = [ds[comp][b] for b in common_bins]  # each (N, D)
                    arr = np.stack(parts, axis=0)               # (n_bins_common, N, D)
                    vecs.append(arr)

                dmat = batched_procrustes_distance(vecs, vecs)
                score = 1 - dmat[0, 1]
                mat[i, j] = mat[j, i] = score

        results[comp] = mat

    return results, status




### Plotly Graphs

def compute_angle_bin(row, n_bins=8):
    pos = row['target_pos']
    idx = row['active_target']
    if isinstance(pos, np.ndarray) and pos.ndim == 2 and 0 <= idx < len(pos):
        x, y = pos[idx]
        angle = np.arctan2(y, x)
        bin_id = int(((angle + np.pi) / (2 * np.pi)) * n_bins) % n_bins
        bin_size = 360 / n_bins
        start = int(bin_id * bin_size)
        end = int(start + bin_size)
        return f"{start}–{end}"
    return np.nan

def plot_trajectories_plotly(hidden_states, title, trial_types=None,
                              pca_dim=2, smooth=False, sigma=1, downsample=None,
                              save_path=None, mean_traj=False, align_traj=False):
    hidden_np = hidden_states.cpu().numpy()
    n_traj = hidden_np.shape[0]

    if hidden_np.ndim != 3:
        raise ValueError("Expected 3D tensor for hidden_states")

    all_h_pca = []
    for i in range(n_traj):
        h = hidden_np[i]
        if smooth:
            h = gaussian_filter1d(h, sigma=sigma, axis=0)
        if downsample is not None:
            h = h[::downsample]
        all_h_pca.append(h)

    # Downsample trial_types if it's 2D and mismatched in time dimension
    if (downsample is not None and trial_types is not None and isinstance(trial_types, (list, np.ndarray))):
        trial_types = np.array(trial_types)
        if trial_types.ndim == 2 and trial_types.shape[1] != all_h_pca[0].shape[0]:
            trial_types = trial_types[:, ::downsample]

    # Fit global PCA
    h_concat = np.concatenate(all_h_pca, axis=0)
    pca = PCA(n_components=pca_dim)
    pca.fit(h_concat)
    #all_h_pca = [pca.transform(h) for h in all_h_pca]

    if mean_traj:
        trial_types = np.array(trial_types)
        unique_types = sorted(set(trial_types))
        min_len = min(h.shape[0] for h in all_h_pca)

        mean_trajs = [
            np.mean([hidden_np[i][:min_len] for i in range(len(trial_types)) if trial_types[i] == label], axis=0)
            for label in unique_types
        ]

        #print(mean_trajs[0][:2])

        if align_traj:
            mean_trajs = align_trajectories(mean_trajs)

        #print(mean_trajs[0][:2])
        #print(mean_trajs[-1].shape)
        #print(mean_trajs[-1][:10])

        all_h_pca = [pca.transform(h) for h in mean_trajs]
        #print(all_h_pca[-1][:10])
        trial_types = unique_types
        opacity = 1.0

    elif align_traj:
        trial_types = np.array(trial_types)
        unique_types = sorted(set(trial_types))
        min_len = min(h.shape[0] for h in all_h_pca)
        aligned = [
            np.array([hidden_np[i][:min_len] for i in range(len(trial_types)) if trial_types[i] == label][0])
            for label in unique_types
        ]
        aligned = align_trajectories(aligned)
        all_h_pca = [pca.transform(h) for h in aligned]
        trial_types = unique_types
        opacity = 1.0

    else:
        opacity = 0.3
        all_h_pca = [pca.transform(h) for h in all_h_pca]

    explained = np.sum(pca.explained_variance_ratio_[:pca_dim])
    print(f"PCA explained {explained * 100:.2f}% of variance")

    # Colour setup
    cmap = get_cmap('tab20')
    #cmap = get_cmap('Greens')
    colour_map = {}
    colours = []
    is_per_timestep = False

    if trial_types is not None:
        trial_types = np.array(trial_types)
        if trial_types.ndim == 2:
            is_per_timestep = True
            unique_types = sorted(set(trial_types.ravel()))
        else:
            unique_types = sorted(set(trial_types))

        colour_map = {tt: cmap(i % cmap.N) for i, tt in enumerate(unique_types)}

        if not is_per_timestep:
            colours = [
                f'rgba({int(r*255)},{int(g*255)},{int(b*255)},{a})'
                for tt in trial_types
                for r, g, b, a in [colour_map[tt]]
            ]

    fig = go.Figure()
    seen_labels = set()

    for idx, h_pca in enumerate(all_h_pca):
        if is_per_timestep:
            tt_series = trial_types[idx]
            colour_array = [colour_map[tt] for tt in tt_series]
            rgba_list = [
                f'rgba({int(r*255)},{int(g*255)},{int(b*255)},{a})'
                for r, g, b, a in colour_array
            ]

            marker_dict = dict(size=2, color=rgba_list)

            if pca_dim == 3:
                fig.add_trace(go.Scatter3d(
                    x=h_pca[:, 0], y=h_pca[:, 1], z=h_pca[:, 2],
                    mode='markers',
                    marker=marker_dict,
                    name=f"Traj {idx + 1}",
                    showlegend=False
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=h_pca[:, 0], y=h_pca[:, 1],
                    mode='markers',
                    marker=marker_dict,
                    name=f"Traj {idx + 1}",
                    showlegend=False
                ))
        else:
            line_colour = colours[idx] if colours else None
            trace_label = str(trial_types[idx]) if trial_types is not None else f"Traj {idx + 1}"
            show_legend = trace_label not in seen_labels
            seen_labels.add(trace_label)

            line_opts = dict(width=2)
            if line_colour:
                line_opts["color"] = line_colour

            if pca_dim == 3:
                fig.add_trace(go.Scatter3d(
                    x=h_pca[:, 0], y=h_pca[:, 1], z=h_pca[:, 2],
                    mode='lines',
                    line=line_opts,
                    name=trace_label,
                    opacity=opacity,
                    showlegend=show_legend
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=h_pca[:, 0], y=h_pca[:, 1],
                    mode='lines',
                    line=line_opts,
                    name=trace_label,
                    opacity=opacity,
                    showlegend=show_legend
                ))

    # Median trajectory
    min_len = min(len(h) for h in all_h_pca)
    stacked = np.array([h[:min_len] for h in all_h_pca])
    mean_line = np.mean(stacked, axis=0)
    #if pca_dim == 3:
    #    fig.add_trace(go.Scatter3d(
    #        x=mean_line[:, 0], y=mean_line[:, 1], z=mean_line[:, 2],
    #        mode='lines',
    #        line=dict(color='black', width=4),
            #opacity=0,
    #        name='Mean Trajectory'
    #    ))
    #else:
    #    fig.add_trace(go.Scatter(
    #        x=mean_line[:, 0], y=mean_line[:, 1],
    #        mode='lines',
    #        line=dict(color='black', width=4),
    #        name='Mean Trajectory'
    #    ))

    # Legend setup
    if is_per_timestep:
        for label in unique_types:
            r, g, b, a = colour_map[label]
            rgba = f'rgba({int(r*255)},{int(g*255)},{int(b*255)},{a})'
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=6, color=rgba),
                name=str(label),
                showlegend=True
            ))
        fig.update_layout(showlegend=True, legend=dict(traceorder='normal'))

    elif trial_types is not None:
        labels_set = {trace.name for trace in fig.data}
        sorted_labels = sorted(labels_set, key=lambda x: int(x.split('–')[0]) if '–' in x else float('inf'))
        fig.data = tuple(sorted(fig.data, key=lambda t: sorted_labels.index(t.name) if t.name in sorted_labels else -1))
        fig.update_layout(showlegend=True, legend=dict(traceorder='normal'))

    else:
        fig.update_layout(showlegend=False)

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3' if pca_dim == 3 else ''
        ) if pca_dim == 3 else dict(
            xaxis_title='PC1',
            yaxis_title='PC2'
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=200, t=50, b=0),
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved to {save_path}")
    else:
        fig.show()


def plot_multiple_trajectories_plotly(hidden_states, overall_title,
                                      burn_in=0, labels=None,
                                      add_noise=False, pca_dim=2,
                                      smooth=False, sigma=1, downsample=None,
                                      trial_types=None, mean_traj = False, align_traj=False, save_path=None):
    n = len(hidden_states)
    titles = labels if labels else [f"Trajectory {i + 1}" for i in range(n)]

    print(overall_title)
    for i in range(n):
        this_save = None
        if save_path:
            base, ext = save_path.split("*")
            this_save = f"{base}{i}{ext}"
        plot_trajectories_plotly(hidden_states[i], titles[i],
                                 trial_types=trial_types,
                                 pca_dim=pca_dim,
                                 smooth=smooth,
                                 sigma=sigma,
                                 downsample=downsample,
                                 mean_traj=mean_traj,
                                 align_traj=align_traj,
                                 save_path=this_save)

def plot_trajectories_plotly_grid(hidden_states, title, trial_types=None,
                                  pca_dim=2, smooth=False, sigma=1, downsample=None,
                                  save_path=None, mean_traj=False, align_traj=False,
                                  fig=None, row=None, col=None, show=True,
                                  fixed_points=None, colour_map=None):
    """
    Draws a single subplot of trajectories. Legends are suppressed here; the parent
    grid adds a single shared legend. Accepts an optional shared `colour_map`.
    """
    import numpy as np
    import plotly.graph_objs as go
    from sklearn.decomposition import PCA
    from scipy.ndimage import gaussian_filter1d
    from matplotlib.cm import get_cmap

    def mpl_rgba_to_plotly_rgba(rgba):
        r, g, b, a = rgba
        return f'rgba({int(r*255)},{int(g*255)},{int(b*255)},{a})'

    hidden_np = hidden_states.cpu().numpy()
    n_traj = hidden_np.shape[0]

    if hidden_np.ndim != 3:
        raise ValueError("Expected 3D tensor for hidden_states")

    # Preprocess each trajectory
    all_h = []
    for i in range(n_traj):
        h = hidden_np[i]
        if smooth:
            h = gaussian_filter1d(h, sigma=sigma, axis=0)
        if downsample is not None:
            h = h[::downsample]
        all_h.append(h)

    # If trial_types is 2D per-timestep and downsampled, match lengths
    if (downsample is not None and trial_types is not None and isinstance(trial_types, (list, np.ndarray))):
        trial_types = np.array(trial_types, dtype=object)
        if trial_types.ndim == 2 and trial_types.shape[1] != all_h[0].shape[0]:
            trial_types = trial_types[:, ::downsample]

    # Global PCA across trajectories in this subplot
    h_concat = np.concatenate(all_h, axis=0)
    pca = PCA(n_components=pca_dim).fit(h_concat)

    if mean_traj:
        trial_types = np.array(trial_types, dtype=object)
        unique_types = sorted(set(trial_types))
        min_len = min(h.shape[0] for h in all_h)
        mean_trajs = [
            np.mean([hidden_np[i][:min_len] for i in range(len(trial_types)) if trial_types[i] == label], axis=0)
            for label in unique_types
        ]
        if align_traj:
            mean_trajs = align_trajectories(mean_trajs)  # assumed defined by caller
        all_h_pca = [pca.transform(h) for h in mean_trajs]
        trial_types = unique_types
        opacity = 1.0
    else:
        opacity = 0.3
        all_h_pca = [pca.transform(h) for h in all_h]

    explained = np.sum(pca.explained_variance_ratio_[:pca_dim])
    print(f"PCA explained {explained * 100:.2f}% of variance")

    # Colours
    colours = []
    is_per_timestep = False
    if trial_types is not None:
        trial_types = np.array(trial_types, dtype=object)

        if colour_map is None:
            cmap = get_cmap('tab20')
            if trial_types.ndim == 2:
                unique_types = sorted(set(trial_types.ravel()))
            else:
                unique_types = sorted(set(trial_types))
            colour_map = {tt: mpl_rgba_to_plotly_rgba(cmap(i % cmap.N)) for i, tt in enumerate(unique_types)}

        if trial_types.ndim == 2:
            is_per_timestep = True
        else:
            colours = [colour_map[tt] for tt in trial_types]

    created_local_fig = False
    if fig is None:
        fig = go.Figure()
        created_local_fig = True

    add_kwargs = {}
    if row is not None and col is not None:
        add_kwargs = dict(row=row, col=col)

    # Plot traces (no legends here)
    for idx, h_pca in enumerate(all_h_pca):
        if is_per_timestep:
            tt_series = trial_types[idx]
            rgba_list = [colour_map[tt] for tt in tt_series]
            marker_dict = dict(size=2, color=rgba_list)

            if pca_dim == 3:
                fig.add_trace(go.Scatter3d(
                    x=h_pca[:, 0], y=h_pca[:, 1], z=h_pca[:, 2],
                    mode='markers', marker=marker_dict,
                    name=f"Traj {idx + 1}", showlegend=False
                ), **add_kwargs)
            else:
                fig.add_trace(go.Scatter(
                    x=h_pca[:, 0], y=h_pca[:, 1],
                    mode='markers', marker=marker_dict,
                    name=f"Traj {idx + 1}", showlegend=False
                ), **add_kwargs)
        else:
            line_colour = colours[idx] if colours else None
            trace_label = str(trial_types[idx]) if trial_types is not None else f"Traj {idx + 1}"
            line_opts = dict(width=2)
            if line_colour:
                line_opts["color"] = line_colour

            if pca_dim == 3:
                fig.add_trace(go.Scatter3d(
                    x=h_pca[:, 0], y=h_pca[:, 1], z=h_pca[:, 2],
                    mode='lines', line=line_opts,
                    name=trace_label, opacity=opacity,
                    showlegend=False
                ), **add_kwargs)
            else:
                fig.add_trace(go.Scatter(
                    x=h_pca[:, 0], y=h_pca[:, 1],
                    mode='lines', line=line_opts,
                    name=trace_label, opacity=opacity,
                    showlegend=False
                ), **add_kwargs)

    # Fixed points (no legend)
    if fixed_points is not None and len(fixed_points) > 0:
        fp_proj = pca.transform(fixed_points)
        if pca_dim == 3:
            fig.add_trace(go.Scatter3d(
                x=fp_proj[:, 0], y=fp_proj[:, 1], z=fp_proj[:, 2],
                mode='markers', marker=dict(size=6, color='black', symbol='x'),
                name="Fixed Points", showlegend=False
            ), **add_kwargs)
        else:
            fig.add_trace(go.Scatter(
                x=fp_proj[:, 0], y=fp_proj[:, 1],
                mode='markers', marker=dict(size=8, color='black', symbol='x'),
                name="Fixed Points", showlegend=False
            ), **add_kwargs)

    if created_local_fig:
        fig.update_layout(
            title=title,
            width=800, height=600,
            margin=dict(l=0, r=200, t=50, b=0),
        )
        if save_path:
            fig.write_html(save_path)
            print(f"Saved to {save_path}")
        elif show:
            fig.show()

    return fig

def plot_multiple_trajectories_plotly_grid(hidden_states, overall_title,
                                           burn_in=0, labels=None,
                                           add_noise=False, pca_dim=2,
                                           smooth=False, sigma=1, downsample=None,
                                           trial_types=None, mean_traj=False,
                                           align_traj=False, save_path=None,
                                           fixed_points=None):
    """
    Two columns, rows expand as needed. Subplots are square.
    Single, shared legend at right-centre titled 'Reaching Angle'.
    Subplot titles are centred & bold inside each panel.
    """
    import numpy as np
    import math
    from plotly.subplots import make_subplots
    import plotly.graph_objs as go
    from matplotlib.cm import get_cmap

    n = len(hidden_states)
    cols = 2
    rows = math.ceil(n / cols)

    subplot_type = 'scene' if pca_dim == 3 else 'xy'
    specs = [[{'type': subplot_type} for _ in range(cols)] for _ in range(rows)]

    # Shared colour map (if trial_types provided)
    colour_map = None
    if trial_types is not None:
        trial_arr = np.array(trial_types, dtype=object)
        unique_types = sorted(set(trial_arr.ravel())) if trial_arr.ndim == 2 else sorted(set(trial_arr))
        cmap = get_cmap('tab20')

        def mpl_rgba_to_plotly_rgba(rgba):
            r, g, b, a = rgba
            return f'rgba({int(r*255)},{int(g*255)},{int(b*255)},{a})'

        colour_map = {tt: mpl_rgba_to_plotly_rgba(cmap(i % cmap.N)) for i, tt in enumerate(unique_types)}

    # No subplot_titles (saves vertical space); zero vertical spacing
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=specs,
        vertical_spacing=0.0,
        horizontal_spacing=0.08
    )

    # Add each subplot (no legends inside)
    for i, hs in enumerate(hidden_states):
        r = 1 + (i // cols)
        c = 1 + (i % cols)
        fp = None if fixed_points is None else fixed_points[i]
        title_txt = labels[i] if labels else f"Trajectory {i+1}"

        plot_trajectories_plotly_grid(
            hs, title_txt, trial_types=trial_types,
            pca_dim=pca_dim, smooth=smooth, sigma=sigma,
            downsample=downsample, mean_traj=mean_traj,
            align_traj=align_traj, fig=fig, row=r, col=c,
            show=False, fixed_points=fp, colour_map=colour_map
        )

    # Figure sizing to keep squares; add right margin for legend
    subplot_size = 500  # px per subplot side
    fig_width  = cols * subplot_size + 300   # legend space on right
    fig_height = rows * subplot_size + 110   # top/bottom margins only

    fig.update_layout(
        title_text=overall_title,
        width=fig_width, height=fig_height,
        margin=dict(l=40, r=240, t=70, b=40),
        legend=dict(
            title=dict(text="Reaching Angle", font=dict(size=18)),  # bigger title
            font=dict(size=14),                                     # bigger items
            x=1.02, y=0.5, xanchor="left", yanchor="middle",
            bgcolor="rgba(255,255,255,0.6)"
        )
    )

    # Enforce square aspect for each subplot
    if pca_dim == 2:
        for i_ax in range(1, n + 1):
            yaxis_name = f"yaxis{i_ax}" if i_ax > 1 else "yaxis"
            scaleanchor_val = f"x{i_ax}" if i_ax > 1 else "x"
            fig.update_layout({yaxis_name: dict(scaleanchor=scaleanchor_val)})
    else:
        for i_ax in range(1, n + 1):
            scene_name = f"scene{i_ax}" if i_ax > 1 else "scene"
            fig.update_layout({scene_name: dict(aspectmode="cube")})

    # Centre, bold in-panel titles
    for i in range(n):
        r = 1 + (i // cols); c = 1 + (i % cols)
        title_txt = labels[i] if labels else f"Trajectory {i+1}"
        axnum = i + 1
        xref = f"x{axnum} domain" if axnum > 1 else "x domain"
        yref = f"y{axnum} domain" if axnum > 1 else "y domain"
        fig.add_annotation(
            text=f"<b>{title_txt}</b>",
            showarrow=False,
            xref=xref, yref=yref,
            x=0.5, y=0.98,                 # centred at top
            xanchor="center", yanchor="top",
            align="center",
            font=dict(size=14, color="#222")
        )

    # Single legend entries (dummy traces) to match shared colours
    if colour_map is not None:
        for tt, col in colour_map.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                line=dict(color=col, width=2),
                name=str(tt), showlegend=True
            ))

    if save_path:
        fig.write_html(save_path)
        print(f"Saved to {save_path}")
    else:
        fig.show()

### Similarity Matrix
def plot_similarity_matrices_blocks(sim_2_block, sim_4_block,
                                        labels=None, save_path=None, cmap="Greens"):
    """
    3×2 grid of heatmaps:
        Rows = ['Encoded Neural Input', 'First Block Post GLU', 'Last Block Post GLU']
        Cols = ['Foundational 2 Block Model', 'Foundational 4 Block Model']

    - x-tick labels only on bottom row
    - y-tick labels only on left column
    - single shared colourbar, placed outside on the right, vertically centred
    """
    row_titles = ['Encoded Neural Input', 'First Block Post GLU', 'Last Block Post GLU']
    col_titles = ['Foundational 2 Block Model', 'Foundational 4 Block Model']

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    mappables = []

    for i, row_title in enumerate(row_titles):
        for j, col_title in enumerate(col_titles):
            ax = axs[i, j]
            # pick matrix
            mat = sim_2_block[i] if j == 0 else sim_4_block[i]

            hm = sns.heatmap(mat, xticklabels=labels, yticklabels=labels,
                             cmap=cmap, cbar=False, ax=ax)
            mappables.append(hm.collections[0])

            # row labels
            if j == 0:
                ax.set_ylabel(row_title, fontsize=11)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            # col labels
            if i == 0:
                ax.set_title(col_title, fontsize=12)

            # ticks
            if i == len(row_titles) - 1:  # bottom row
                ax.set_xticklabels(labels if labels is not None else ax.get_xticklabels(),
                                   rotation=90, fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_yticklabels(labels if labels is not None else ax.get_yticklabels(),
                                   fontsize=8)
            else:
                ax.set_yticklabels([])

    # shared colourbar — outside right centre
    cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
    fig.colorbar(mappables[0], cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # reserve space for cbar

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()



### Participation Ratio

# --- core PR ---
def participation_ratio_fast(X):
    """Return absolute PR and the normalising dim (min(units, samples))."""
    s = np.linalg.svd(X, compute_uv=False)
    s2 = s**2
    pr_abs = (s2.sum()**2) / (s2**2).sum()
    return float(pr_abs), min(X.shape)

# --- block subsampling along time axis ---
def _block_indices(T, target_len, block_len, rng):
    """Circular block bootstrap indices for a single draw."""
    n_blocks = int(np.ceil(target_len / block_len))
    idxs = []
    for _ in range(n_blocks):
        start = rng.integers(0, T)
        end = start + block_len
        if end <= T:
            idxs.append(np.arange(start, end))
        else:
            idxs.append(np.r_[np.arange(start, T), np.arange(0, end - T)])
    return np.concatenate(idxs)[:target_len]

def pr_subsamples(H, n_sub=10, sample_frac=0.6, block_len=None, ratio=False, rng=None):
    """
    H: (T, units) — time × features (last dim is units).
    Returns: np.array of length n_sub with ABSOLUTE PR values from block subsamples.

    Note: 'ratio' is ignored (kept for API compatibility).
    """
    rng = np.random.default_rng(rng)
    T, _ = H.shape
    if T < 2:
        raise ValueError("Need at least 2 time points.")
    if block_len is None:
        block_len = max(1, int(np.sqrt(T)))
    target_len = max(block_len, int(sample_frac * T))

    vals = np.empty(n_sub, dtype=float)
    for i in range(n_sub):
        idx = _block_indices(T, target_len, block_len, rng)
        Hb = H[idx]
        pr, _ = participation_ratio_fast(Hb.T)  # absolute PR only
        vals[i] = pr
    return vals

# --- bootstrap over the n_sub PR values ---
def bootstrap_stat(vals, n_boot=1000, stat="median", ci=(2.5, 97.5), rng=None):
    """
    vals: 1D array of PR values (e.g., length 10).
    Returns dict with point estimate, CI, and bootstrap mean/std of the bootstrap distribution.
    """
    rng = np.random.default_rng(rng)
    vals = np.asarray(vals, float)
    n = vals.size
    if n < 2:
        raise ValueError("Need at least 2 values for bootstrap.")

    if stat == "median":
        f = np.median
    elif stat == "mean":
        f = np.mean
    else:
        raise ValueError("stat must be 'median' or 'mean'.")

    boots = np.empty(n_boot, float)
    idx_choices = np.arange(n)
    for b in range(n_boot):
        sample = vals[rng.integers(0, n, size=n)]
        boots[b] = f(sample)

    est = f(vals)
    lo, hi = np.percentile(boots, ci)
    return {"estimate": float(est),
            "ci": (float(lo), float(hi)),
            "boot_mean": float(boots.mean()),
            "boot_std": float(boots.std(ddof=1)),
            "n_vals": int(n),
            "n_boot": int(n_boot),
            "stat": stat}

# --- wrapper for your dict -> per key and concat ---
def pr_bootstrap(data, components,
                 n_sub=10, n_boot=1000,
                 sample_frac=0.6, block_len=None,
                 ratio=False, stat="median", rng=None, rtt=False):
    """
    data: {key: {component: array(..., units)}}; last dim is units.
    components: list of component names.
    Returns:
      {
        key: {
          component: {
            "vals": array (absolute PRs),
            "summary": dict (bootstrap over those PRs),
            "dim": int  # min(units, full T for this key-component)
          }, ...
        },
        "concat": {component: {...}}
      }

    Note: 'ratio' is ignored (kept for API compatibility). Ratios are handled in plotting.
    """
    keys = list(data.keys())
    out = {k: {} for k in keys}
    if not rtt:
        out["concat"] = {}

    for comp in tqdm(components, total=len(components)):
        H_all = []

        for k in keys:
            if comp not in data[k]:
                continue
            A = data[k][comp]
            H = A.reshape(-1, A.shape[-1])  # (T_k, units)


            # --- minimal mask application ---
            m = data[k].get("mask", None)
            if m is not None:
                m = np.asarray(m).reshape(-1).astype(bool)
                if m.size != H.shape[0]:
                    raise ValueError(f"mask length mismatch for key '{k}', component '{comp}'")
                H = H[m]
                if H.shape[0] == 0:
                    continue

            vals = pr_subsamples(H, n_sub=n_sub, sample_frac=sample_frac,
                                 block_len=block_len, ratio=False, rng=rng)  # absolute PRs
            summary = bootstrap_stat(vals, n_boot=n_boot, stat=stat, rng=rng)
            dim_full = min(H.shape[1], H.shape[0])
            out[k][comp] = {"vals": vals, "summary": summary, "dim": int(dim_full)}
            H_all.append(H)

        if not rtt:
            if H_all:
                Hc = np.vstack(H_all)
                vals_c = pr_subsamples(Hc, n_sub=n_sub, sample_frac=sample_frac,
                                      block_len=block_len, ratio=False, rng=rng)
                summary_c = bootstrap_stat(vals_c, n_boot=n_boot, stat=stat, rng=rng)
                dim_c = min(Hc.shape[1], Hc.shape[0])
                out["concat"][comp] = {"vals": vals_c, "summary": summary_c, "dim": int(dim_c)}

    return out

def plot_pr_results_with_ci(results, components, component_labels,
                            ratio=False, concat_last=True, ci_alpha=0.15):
    """
    Plot median (or point estimate) with CI shading per key across components.

    Expects each cell as:
      results[key][component] == {"summary": {"estimate": float, "ci": (lo, hi)}, "dim": int}

    If ratio=True, divides estimate and CI bounds by 'dim' at plot time.
    """
    x = np.arange(len(components))
    fig, ax = plt.subplots(figsize=(10, 5))

    # order keys, putting concat/concatenated last if requested
    keys = list(results.keys())
    tail = [k for k in ("concat", "concatenated") if k in keys]
    if concat_last and tail:
        keys = [k for k in keys if k not in tail] + tail

    for key in keys:
        comp_dict = results.get(key, {})
        if not comp_dict:
            continue

        xs, y_med, y_lo, y_hi = [], [], [], []
        for i, comp in enumerate(components):
            if comp not in comp_dict:
                continue
            cell = comp_dict[comp]
            if not isinstance(cell, dict) or "summary" not in cell or "dim" not in cell:
                continue  # skip malformed entries

            est = float(cell["summary"].get("estimate", np.nan))
            lo, hi = cell["summary"].get("ci", (np.nan, np.nan))
            d = cell.get("dim", None)

            if ratio and (d is not None and d > 0):
                est = est / d
                if np.isfinite(lo): lo = lo / d
                if np.isfinite(hi): hi = hi / d

            xs.append(i)
            y_med.append(est)
            y_lo.append(lo)
            y_hi.append(hi)

        if not xs:
            continue

        xs = np.asarray(xs)
        y_med = np.asarray(y_med, dtype=float)
        y_lo = np.asarray(y_lo, dtype=float)
        y_hi = np.asarray(y_hi, dtype=float)

        lw = 2.2 if key in ("concat", "concatenated") else 1.6
        (line,) = ax.plot(xs, y_med, marker="o", linewidth=lw, label=str(key))
        finite = np.isfinite(y_lo) & np.isfinite(y_hi)
        if finite.any():
            ax.fill_between(xs[finite], y_lo[finite], y_hi[finite],
                            alpha=ci_alpha, facecolor=line.get_color())

    ax.set_xticks(x, component_labels, rotation=45, ha="right")
    ax.set_ylabel("Participation Ratio" if ratio else "Effective Dimension")
    #ax.set_title("Participation Ratio across Components")
    ax.legend(title="Key", frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()

def plot_pr_results_with_ci_stacked(results, components, component_labels,
                                    concat_last=True, ci_alpha=0.15):
    """
    Plot median (or point estimate) with CI shading per key across components.

    Produces two subplots:
      - Top: effective dimension (ratio=False)
      - Bottom: participation ratio (ratio=True)

    Expects each cell as:
      results[key][component] == {"summary": {"estimate": float, "ci": (lo, hi)}, "dim": int}
    """
    import re
    from collections import defaultdict

    def base_int(x):
        m = re.match(r'^(\d+)', str(x))
        return int(m.group(1)) if m else None

    # ---- build legend label & task for each key (to colour by task) ----
    # Falls back to the raw key and task='unknown' if not in DATASET_IDX_TO_GROUP
    key_meta = {}  # key -> dict(label, task)
    for key in results.keys():
        if str(key) in ("concat", "concatenated"):
            key_meta[key] = {"label": str(key), "task": "concat_special"}
            continue
        bi = base_int(key)
        if bi is not None and bi in DATASET_IDX_TO_GROUP:
            subj = DATASET_IDX_TO_GROUP[bi][1]
            task = DATASET_IDX_TO_GROUP[bi][-1]
            label = f"{subj} | {task}"
            key_meta[key] = {"label": label, "task": task}
        else:
            key_meta[key] = {"label": str(key), "task": "unknown"}

    # ---- colour assignment: per-task cmap; concat fixed yellow ----
    task_names = sorted({m["task"] for m in key_meta.values() if m["task"] not in ("concat_special",)})
    task_cmaps = [
        "Blues", "Greens", "Oranges", "Purples", "Reds", "Greys",
        "BuGn", "GnBu", "PuBu", "YlGn"
    ]
    cmap_for_task = {
        t: plt.get_cmap(task_cmaps[i % len(task_cmaps)])
        for i, t in enumerate(task_names)
    }

    # group keys by task to allocate distinct shades within each task
    by_task = defaultdict(list)  # task -> [key,...]
    for k, m in key_meta.items():
        by_task[m["task"]].append(k)

    # pick colours
    key_colour = {}
    for task, keys_in_task in by_task.items():
        if task == "concat_special":
            for k in keys_in_task:
                key_colour[k] = {"line": "#f1c40f", "fill": "#f1c40f"}  # yellow
            continue
        cmap = cmap_for_task.get(task, plt.get_cmap("tab10"))
        n = max(2, len(keys_in_task))
        shades = np.linspace(0.35, 0.85, num=n)  # mid-range for visibility
        for shade, k in zip(shades, keys_in_task):
            col = cmap(shade)
            key_colour[k] = {"line": col, "fill": col}

    x = np.arange(len(components))
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # order keys, putting concat/concatenated last if requested
    keys = list(results.keys())
    tail = [k for k in ("concat", "concatenated") if k in keys]
    if concat_last and tail:
        keys = [k for k in keys if k not in tail] + tail

    for ax, ratio in zip(axes, [False, True]):
        for key in keys:
            comp_dict = results.get(key, {})
            if not comp_dict:
                continue

            xs, y_med, y_lo, y_hi = [], [], [], []
            for i, comp in enumerate(components):
                if comp not in comp_dict:
                    continue
                cell = comp_dict[comp]
                if not isinstance(cell, dict) or "summary" not in cell or "dim" not in cell:
                    continue  # skip malformed entries

                est = float(cell["summary"].get("estimate", np.nan))
                lo, hi = cell["summary"].get("ci", (np.nan, np.nan))
                d = cell.get("dim", None)

                if ratio and (d is not None and d > 0):
                    est = est / d
                    if np.isfinite(lo): lo = lo / d
                    if np.isfinite(hi): hi = hi / d

                xs.append(i)
                y_med.append(est)
                y_lo.append(lo)
                y_hi.append(hi)

            if not xs:
                continue

            xs = np.asarray(xs)
            y_med = np.asarray(y_med, dtype=float)
            y_lo = np.asarray(y_lo, dtype=float)
            y_hi = np.asarray(y_hi, dtype=float)

            meta = key_meta.get(key, {"label": str(key), "task": "unknown"})
            colours = key_colour.get(key, {"line": None, "fill": None})
            lw = 2.2 if str(key) in ("concat", "concatenated") else 1.6

            (line,) = ax.plot(xs, y_med, marker="o", linewidth=lw,
                              label=meta["label"], color=colours["line"])
            finite = np.isfinite(y_lo) & np.isfinite(y_hi)
            if finite.any():
                ax.fill_between(xs[finite], y_lo[finite], y_hi[finite],
                                alpha=ci_alpha, facecolor=colours["fill"] or line.get_color())

        ax.set_ylabel("Participation Ratio" if ratio else "Effective Dimension")
        ax.legend(ncol=1, fontsize=9, frameon=True)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xticks(x, component_labels, rotation=45, ha="right")
    fig.tight_layout()
    plt.show()