"""
nbit.analysis

Auto-generated from nbit_analysis.ipynb.
- Exports analysis/helper functions found in the notebook.
- Re-exports training API from nbit.training when used.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
import numpy as np
import copy
import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
import plotly.io as pio
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import math
from scipy.spatial import procrustes
from nbit.training import ContinuousRNN, generate_batch, eval_rnn, eval_rnn_multiple, load_rnn, plot_training_curves, randomise_rnn, save_rnn_model, train_rnn

if 'google.colab' in sys.modules:
    pio.renderers.default = 'colab'


def get_perimeter_input(transitions=None, T_per=300, pulse_width=5, pulse_offset=50, T_buffer=100, n_bits=2, device=device):
    assert n_bits == 2, 'Only 2-bit states supported.'
    if transitions is None:
        transitions = [[+1, +1], [-1, +1], [-1, -1], [+1, -1], [+1, +1]]
    num_transitions = len(transitions) - 1
    T_total = T_buffer + T_per * (num_transitions + 1)
    inputs = torch.zeros(1, T_total, n_bits, device=device)
    targets = torch.zeros(1, T_total, n_bits, device=device)
    init_indices = []
    t_start = T_buffer
    t_pulse = t_start + pulse_offset
    inputs[0, t_pulse:t_pulse + pulse_width, :] = +1
    targets[0, t_start:t_start + T_per, :] = torch.tensor([+1, +1], device=device)
    init_indices.extend([t_pulse - 1, t_pulse + 1, t_start + T_per - 1])
    for i in range(num_transitions):
        curr = torch.tensor(transitions[i], device=device)
        next_ = torch.tensor(transitions[i + 1], device=device)
        flip_bit = (curr != next_).nonzero(as_tuple=True)[0].item()
        t_start = T_buffer + T_per * (i + 1)
        t_pulse = t_start + pulse_offset
        t_late = t_start + T_per - 1
        inputs[0, t_pulse:t_pulse + pulse_width, flip_bit] = -curr[flip_bit]
        targets[0, t_start:t_start + T_per, :] = next_
        init_indices.extend([t_pulse - 1, t_pulse + 1, t_late])
    return (torch.cat([inputs], dim=0), init_indices, targets)

### Fixed points

def get_hidden_states(rnn, input_list, burn_in=0, add_noise=False, pca_dim=None, smooth=False, sigma=1, squeeze=True, downsample=None):
    """
    Gets hidden states from an RNN and optionally applies PCA and smoothing.

    Parameters
    ----------
    rnn : torch.nn.Module
        RNN model with return_h=True support.
    input_list : torch.Tensor or list of tensors
        Input sequence(s) to the RNN.
    add_noise : bool, default=False
        Whether to enable noise during the forward pass.
    pca_dim : int or None, default=None
        If not None, apply PCA with this number of components.
    smooth : bool, default=False
        If True, apply Gaussian smoothing to the PCA output.
    sigma : float, default=1
        Standard deviation for the Gaussian filter.
    squeeze : bool, default=True
        If True, remove the batch dimension from the output.

    Returns
    -------
    hidden_states : np.ndarray
        The raw hidden states if pca_dim is None, or PCA-projected (and optionally smoothed) data.
    """

    with torch.no_grad():
        _, hs = rnn(input_list, return_h=True, add_noise=add_noise)
        orig_device = hs.device
        orig_dtype = hs.dtype
        if squeeze:
            hs = hs.squeeze(0)
        hs = hs.detach().cpu().numpy()  # shape: (T, N)
        if squeeze:
          hs = hs[burn_in:]
        else:
          hs = hs[:, burn_in:, :]

        if downsample is not None:
            if squeeze:
              hs = hs[::downsample]
            else:
              hs = hs[:, ::downsample, :]


    if pca_dim is not None:
        if squeeze:
          pca = PCA(n_components=pca_dim)
          hs = pca.fit_transform(hs)
          if smooth:
              hs = gaussian_filter1d(hs, sigma=sigma, axis=0)
        else:
          B, T, N = hs.shape
          pca = PCA(n_components=pca_dim)
          hs_2d = hs.reshape(B*T, N)            # flatten across batch+time
          hs_pca = pca.fit_transform(hs_2d)     # fit once on all data
          hs = hs_pca.reshape(B, T, pca_dim)    # back to [B, T, pca_dim]
          if smooth:
              hs = gaussian_filter1d(hs, sigma=sigma, axis=1)  # smooth over time


    hs = torch.tensor(hs, dtype=orig_dtype, device=orig_device)

    return hs

def pca_with_smoothing(hidden_states, pca_dim=2, smooth=False, sigma=1):
    orig_device = hidden_states.device
    orig_dtype = hidden_states.dtype
    hidden_states = hidden_states.cpu().numpy()

    pca = PCA(n_components=pca_dim)
    h_pca = pca.fit_transform(hidden_states)
    if smooth:
        h_pca = gaussian_filter(h_pca, sigma=sigma)

    h_pca = torch.tensor(h_pca, dtype=orig_dtype, device=orig_device)

    return h_pca

def find_fixed_points(rnn, hidden_states, init_indices, T=1000, pulse_t=150, k=3, n_bits=3):
    # --- PARAMETERS ---
    #T = 1000                # sequence length
    #pulse_t = 150      # time of the bit‐flip pulse
    #k = 3                  # number of samples per trajectory: before, after, end
    #n_bits = 3
    device = next(rnn.parameters()).device

    # --- 2) Sample k=3 points from each trajectory ---
    #init_points = []   # list of torch.Tensor shape (1, N)
    #for traj in flip_trajectories:
    #for hs in hidden_states:
        #hs = traj
        #times = [pulse_t - 1, pulse_t + 1, T - 1]  # just before, just after, end
    #    for t in init_indices:
    #        init_points.append(hs[t].view(1, -1))
    init_points = hidden_states[init_indices]

    def find_fixed_point(rnn, h_init, input_t=None, n_iters=500, lr=1e-2):
        h = h_init.clone().detach().to(rnn.input_weights.device).requires_grad_(True)
        optimizer = torch.optim.Adam([h], lr=lr)

        for _ in range(n_iters):
            optimizer.zero_grad()
            input_effective = torch.zeros(1, rnn.n_bits, device=h.device) if input_t is None else input_t
            dh = (-h + torch.tanh(h @ rnn.recurrent_weights.T + input_effective @ rnn.input_weights.T)) * dt
            loss = torch.norm(dh)
            loss.backward()
            optimizer.step()

        return h.detach()

    fixed_points = []
    for h_init in tqdm(init_points):
        #fp = find_fixed_point(rnn, h_init.unsqueeze(0))  # add batch dim
        fp = find_fixed_point(rnn, h_init)
        fixed_points.append(fp.cpu().numpy())


    fixed_points = np.vstack(fixed_points)
    #fixed_points = fixed_points.squeeze(1)

    #return fixed_points, flip_trajectories
    return fixed_points

def classify_fixed_points_dynamics(rnn, fixed_points, dt=dt):
    """
    Classify each fixed point as an attractor, repeller, or saddle
    based on the Jacobian eigenvalues of the autonomous dynamics.
    """
    if not isinstance(fixed_points, torch.Tensor):
        fixed_points = torch.tensor(
            fixed_points, dtype=torch.float32, device=rnn.output_weights.device
        )
    classifications = []
    for h_fp in fixed_points:
        h_fp = h_fp.detach().requires_grad_(True)
        input_t = torch.zeros(rnn.n_bits, device=h_fp.device)
        def dynamics(h):
            return (-h + torch.tanh(
                        h @ rnn.recurrent_weights.T +
                        input_t @ rnn.input_weights.T
                   )) * dt
        J = torch.autograd.functional.jacobian(dynamics, h_fp)
        eigvals = torch.linalg.eigvals(J).cpu().numpy()
        real_parts = eigvals.real
        n_pos = (real_parts > 0).sum()
        n_neg = (real_parts < 0).sum()
        if n_pos == 0:
            classifications.append("attractor")
        elif n_neg == 0:
            classifications.append("repeller")
        else:
            classifications.append("saddle")
    return classifications

def classify_memory_states(fixed_points, rnn, tol=0.2):
    """
    Determine the ±1/0 memory pattern at each fixed point by thresholding.

    tol: any readout with |z|<tol will be labeled 0, otherwise ±1.
    """
    # get outputs [n_fp, n_bits]
    if not isinstance(fixed_points, torch.Tensor):
        fixed_points = torch.tensor(
            fixed_points, dtype=torch.float32, device=rnn.output_weights.device
        )
    else:
        fixed_points = fixed_points.to(rnn.output_weights.device)

    with torch.no_grad():
        z = fixed_points @ rnn.output_weights.T             # shape (n_fp, n_bits)
        z = z.cpu().numpy()

    # threshold into -1, 0, +1
    labels = np.zeros_like(z, dtype=int)
    labels[z >  tol] = +1
    labels[z < -tol] = -1
    # labels where |z|<=tol remain 0

    return labels

def memory_pattern_to_string(memory_patterns):
    """
    Convert each ±1 memory pattern into a human-readable string.
    """
    return [' '.join(f"{int(b):+d}" for b in row) for row in memory_patterns]

### Eigenvalue
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
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            if i == n_rows - 1:
                ax.set_xlabel("Real")
            if j == 0:
                ax.set_ylabel("Imag")

        axes[i, 0].set_ylabel(f"{row_labels[i]}\nImag", rotation=90, labelpad=12)

    return fig, axes

### Matplotlib Trajectory
def view_from_vector(ax, v):
    """
    Given a direction vector v = [vx,vy,vz] in data space (PCA coords),
    set ax.view_init so the camera looks along v.
    """
    # normalize
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v)
    # elevation is angle above the xy-plane:
    elev = np.degrees(np.arcsin(v[2]))
    # azimuth is angle in the xy-plane from the x-axis:
    azim = np.degrees(np.arctan2(v[1], v[0]))
    ax.view_init(elev=elev, azim=azim)

def plot_trajectories(rnn, hidden_states, fixed_points, title, pca_dim=2, ax=None,
                      show_legend=True, smooth=False, sigma=1, downsample=None):

    pca = PCA(n_components=pca_dim)
    h_pca = pca.fit_transform(hidden_states.cpu().numpy())
    if smooth:
        h_pca = gaussian_filter1d(h_pca, sigma=sigma, axis=0)
    explained = np.sum(pca.explained_variance_ratio_[:2])
    print(f"PCA explained {explained * 100:.2f}% of variance")

    if fixed_points is not None:
        fp_pca = pca.transform(fixed_points)
        memory_states = classify_memory_states(fixed_points, rnn)
        memory_labels = memory_pattern_to_string(memory_states)
        dynamics_labels = classify_fixed_points_dynamics(rnn, fixed_points)
        unique_memories, color_indices = np.unique(memory_labels, return_inverse=True)
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i) for i in color_indices]
        shape_map = {"attractor": "o", "saddle": "s", "repeller": "X"}

    if downsample is not None:
        h_pca = h_pca[::downsample]

    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d' if pca_dim == 3 else None)

    # Plot trajectory
    if pca_dim == 3:
        ax.plot(h_pca[:, 0], h_pca[:, 1], h_pca[:, 2], color='lightsteelblue', lw=1, label='Trajectory')
    else:
        ax.plot(h_pca[:, 0], h_pca[:, 1], color='lightsteelblue', lw=1, label='Trajectory')

    scatter_handles = []
    scatter_labels = []
    if fixed_points is not None:
        plotted_labels = set()
        for i, coords in enumerate(fp_pca):
            x, y = coords[:2]
            z = coords[2] if pca_dim == 3 else None
            full_label = f"{memory_labels[i]} ({dynamics_labels[i]})"
            label = None if full_label in plotted_labels else full_label
            if label:
                plotted_labels.add(full_label)
            sc = ax.scatter(*coords[:3] if pca_dim == 3 else coords[:2],
                            color=colors[i],
                            marker=shape_map[dynamics_labels[i]],
                            s=100, label=label)
            if label:
                scatter_handles.append(sc)
                scatter_labels.append(label)

    # Quiver arrows (trajectory direction)
    step = max(1, len(h_pca) // 50)
    points = h_pca[::step]
    next_points = h_pca[step::step]
    if len(points) > len(next_points):
        points = points[:-1]
    dirs = next_points - points

    if pca_dim == 3:
        ax.quiver(points[:, 0], points[:, 1], points[:, 2],
                  dirs[:, 0], dirs[:, 1], dirs[:, 2],
                  length=0.1, normalize=True,
                  color='navy', linewidth=0.5, arrow_length_ratio=0.5)
    else:
        ax.quiver(points[:, 0], points[:, 1],
                  dirs[:, 0], dirs[:, 1],
                  angles='xy', scale_units='xy', scale=1,
                  color='navy', width=0.003,
                  headwidth=5, headlength=7, headaxislength=6)

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    if show_legend and (scatter_handles or True):  # always show trajectory label
        handles, labels = ax.get_legend_handles_labels()
        line_handles = [h for h in handles if h.get_label() == 'Trajectory']
        final_handles = line_handles + scatter_handles
        final_labels = ['Trajectory'] + scatter_labels
        ax.legend(final_handles, final_labels, loc='best', fontsize='small', frameon=True)

    if pca_dim == 3:
        view_from_vector(ax, [1, 0, 0.3])
        ax.azim += 10

    if ax is None:
        plt.tight_layout()
        plt.show()

    return ax, memory_labels, dynamics_labels

#plot_trajectories(threebitrnn, flip_trajectories, fixed_points, 3)

def plot_traj_fixed_points(rnn, input_transitions, title, n_bits=2, ax=None, burn_in=0, add_noise=False, pca_dim=None, smooth=False, sigma=1, downsample=None):
    input_list, init_indices, _ = get_perimeter_input(transitions=input_transitions, pulse_width=3, n_bits=2)
    if n_bits == 3:
        zeros = torch.zeros_like(input_list[..., :1])  # shape (1,500,1)
        input_list = torch.cat([input_list, zeros], dim=-1)  # shape (1,500,3)
        input_list[0, 0:10, 2] = 1.0
    hidden_states = get_hidden_states(rnn, input_list, burn_in=burn_in, add_noise=add_noise)
    init_indices = [i - burn_in for i in init_indices if i >= burn_in]
    fixed_points = find_fixed_points(rnn, hidden_states, init_indices, n_bits)
    ax, memory_labels, dynamics_labels = plot_trajectories(rnn, hidden_states, fixed_points, title, pca_dim=pca_dim, ax=ax, smooth=smooth, sigma=sigma, downsample=downsample)

    # Changed: now returns only fixed_points
    return fixed_points, memory_labels, dynamics_labels

def plot_multiple_trajectories(model_col_0, models_col_1, col_1_titles, input_transitions_list, overall_title, burn_in=0, add_noise=False, pca_dim=2, smooth=False, sigma=1, downsample=None):

    n = len(models_col_1)

    fig = plt.figure(figsize=(12, 4 * n))
    axs = []

    for i in range(n):
        # Left axis: 3D if pca_dim==3 else 2D
        if pca_dim == 3:
            ax_left = fig.add_subplot(n, 2, 2 * i + 1, projection='3d')
            ax_right = fig.add_subplot(n, 2, 2 * i + 2, projection='3d')
        else:
            ax_left = fig.add_subplot(n, 2, 2 * i + 1)
            ax_right = fig.add_subplot(n, 2, 2 * i + 2)

        axs.append([ax_left, ax_right])

    axs = np.array(axs)
    fig.suptitle(overall_title, fontsize=16)


    for i in range(n):
        # Left column: fixed plot (reuse the same data/parameters)
        plot_traj_fixed_points(model_col_0 , None, "Two Bit RNN (Baseline)", n_bits=2, ax=axs[i, 0], burn_in=burn_in, add_noise=add_noise, pca_dim=pca_dim, smooth=smooth, sigma=sigma, downsample=downsample)

        # Right column: ith plot from the list
        plot_traj_fixed_points(models_col_1[i] , input_transitions_list[i], col_1_titles[i], n_bits=models_col_1[i].n_bits, ax=axs[i, 1], burn_in=burn_in, add_noise=add_noise, pca_dim=pca_dim, smooth=smooth, sigma=sigma, downsample=downsample)

    plt.tight_layout()
    #fig.savefig("/home/davio/output/combined_trajectories.pdf", dpi=300, bbox_inches='tight')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # only needed if your env requires explicit import

def plot_multiple_trajectories_grid(
    models,
    titles,
    input_transitions_list=None,
    overall_title="",
    burn_in=0,
    add_noise=False,
    pca_dim=2,
    smooth=False,
    sigma=1,
    downsample=None,
):
    """
    Plot models in a 2 x N grid (N = ceil(len(models)/2)).
    - models: list of model objects
    - titles: list of strings, same length as models
    - input_transitions_list: optional list matching models (or None for all)
    """

    m = len(models)
    if len(titles) != m:
        raise ValueError("`titles` must match the number of `models`.")
    if input_transitions_list is not None and len(input_transitions_list) != m:
        raise ValueError("`input_transitions_list` must be None or match the number of `models`.")

    n_cols = int(np.ceil(m / 2))
    n_rows = 2
    fig_height = 5 * n_rows
    fig_width = 5 * n_cols
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.suptitle(overall_title, fontsize=16)

    # Create axes
    axes = []
    for r in range(2):
        row_axes = []
        for c in range(n_cols):
            idx = r * n_cols + c
            if pca_dim == 3:
                ax = fig.add_subplot(2, n_cols, idx + 1, projection='3d')
            else:
                ax = fig.add_subplot(2, n_cols, idx + 1)
            row_axes.append(ax)
        axes.append(row_axes)
    axes = np.array(axes)

    # Changed: accumulate fixed points per model
    fixed_points_list = []
    memory_labels_list = []
    dynamics_labels_list = []

    # Plot each model
    for i, model in enumerate(models):
        row = i // n_cols
        col = i % n_cols
        transitions = None if input_transitions_list is None else input_transitions_list[i]
        title = titles[i]
        n_bits = getattr(model, "n_bits", 2)

        fps, memory_labels, dynamics_labels = plot_traj_fixed_points(
            model,
            transitions,
            title,
            n_bits=n_bits,
            ax=axes[row, col],
            burn_in=burn_in,
            add_noise=add_noise,
            pca_dim=pca_dim,
            smooth=smooth,
            sigma=sigma,
            downsample=downsample,
        )
        fixed_points_list.append(fps)
        memory_labels_list.append(memory_labels)
        dynamics_labels_list.append(dynamics_labels)

    # Hide any unused axes (if odd number of models)
    last_idx = m
    total_slots = 2 * n_cols
    for idx in range(last_idx, total_slots):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].axis('off')

    plt.tight_layout()
    plt.show()

    # Changed: return list of fixed_points, same length as models
    return fixed_points_list, memory_labels_list, dynamics_labels_list


### Plotly Trajectory
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
    else:
        opacity = 0.3
        all_h_pca = [pca.transform(h) for h in all_h_pca]

    explained = np.sum(pca.explained_variance_ratio_[:pca_dim])
    print(f"PCA explained {explained * 100:.2f}% of variance")

    # Colour setup
    cmap = get_cmap('tab20')
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
        #fig.add_trace(go.Scatter3d(
        #    x=mean_line[:, 0], y=mean_line[:, 1], z=mean_line[:, 2],
        #    mode='lines',
        #    line=dict(color='black', width=4),
            #opacity=0,
        #    name='Mean Trajectory'
        #))
    #else:
        #fig.add_trace(go.Scatter(
        #    x=mean_line[:, 0], y=mean_line[:, 1],
        #    mode='lines',
        #    line=dict(color='black', width=4),
        #    name='Mean Trajectory'
        #))

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
                                  fixed_points=None):
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
    pca = PCA(n_components=pca_dim).fit(h_concat)

    if mean_traj:
        trial_types = np.array(trial_types)
        unique_types = sorted(set(trial_types))
        min_len = min(h.shape[0] for h in all_h_pca)

        mean_trajs = [
            np.mean([hidden_np[i][:min_len] for i in range(len(trial_types)) if trial_types[i] == label], axis=0)
            for label in unique_types
        ]

        if align_traj:
            mean_trajs = align_trajectories(mean_trajs)

        all_h_pca = [pca.transform(h) for h in mean_trajs]
        trial_types = unique_types
        opacity = 1.0
    else:
        opacity = 0.3
        all_h_pca = [pca.transform(h) for h in all_h_pca]

    explained = np.sum(pca.explained_variance_ratio_[:pca_dim])
    print(f"PCA explained {explained * 100:.2f}% of variance")

    # Prepare colours
    cmap = get_cmap('tab20')
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

    created_local_fig = False
    if fig is None:
        fig = go.Figure()
        created_local_fig = True

    add_kwargs = {}
    if row is not None and col is not None:
        add_kwargs = dict(row=row, col=col)

    for idx, h_pca in enumerate(all_h_pca):
        if is_per_timestep:
            tt_series = trial_types[idx]
            colour_array = [colour_map[tt] for tt in tt_series]
            rgba_list = [f'rgba({int(r*255)},{int(g*255)},{int(b*255)},{a})'
                         for r, g, b, a in colour_array]

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
            show_legend = False

            line_opts = dict(width=2)
            if line_colour:
                line_opts["color"] = line_colour

            if pca_dim == 3:
                fig.add_trace(go.Scatter3d(
                    x=h_pca[:, 0], y=h_pca[:, 1], z=h_pca[:, 2],
                    mode='lines', line=line_opts,
                    name=trace_label, opacity=opacity,
                    showlegend=show_legend
                ), **add_kwargs)
            else:
                fig.add_trace(go.Scatter(
                    x=h_pca[:, 0], y=h_pca[:, 1],
                    mode='lines', line=line_opts,
                    name=trace_label, opacity=opacity,
                    showlegend=show_legend
                ), **add_kwargs)

    # --- overlay all fixed points once on top of everything ---
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
    n = len(hidden_states)
    cols = math.ceil(n / 2)
    rows = 1 if n == 1 else 2

    subplot_type = 'scene' if pca_dim == 3 else 'xy'
    specs = [[{'type': subplot_type} for _ in range(cols)] for _ in range(rows)]

    titles = labels if labels else [f"Trajectory {i + 1}" for i in range(n)]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        specs=specs, vertical_spacing=0.08, horizontal_spacing=0.05)

    for i, hs in enumerate(hidden_states):
        r = 1 + (i // cols)
        c = 1 + (i % cols)
        plot_trajectories_plotly_grid(
            hs, titles[i], trial_types=trial_types,
            pca_dim=pca_dim, smooth=smooth, sigma=sigma,
            downsample=downsample, mean_traj=mean_traj,
            align_traj=align_traj, fig=fig, row=r, col=c,
            show=False, fixed_points=fixed_points[i]
        )

    fig.update_layout(title_text=overall_title, height=700, width=1200)

    if save_path:
        fig.write_html(save_path)
        print(f"Saved to {save_path}")
    else:
        fig.show()

    # --- UPDATED inner function: now accepts memory_labels/dynamics_labels and optional colour_map ---
def plot_trajectories_plotly_grid(
    hidden_states, title, trial_types=None,
    pca_dim=2, smooth=False, sigma=1, downsample=None,
    save_path=None, mean_traj=False, align_traj=False,
    fig=None, row=None, col=None, show=True,
    fixed_points=None,
    memory_labels=None,           # NEW
    dynamics_labels=None,         # NEW
    colour_map=None               # NEW (for consistent colours across a grid)
):
    hidden_np = hidden_states.cpu().numpy()
    n_traj = hidden_np.shape[0]

    if hidden_np.ndim != 3:
        raise ValueError("Expected 3D tensor for hidden_states")

    all_h_pca = []
    for i in range(n_traj):
        h = hidden_np[i]
        if smooth:
            from scipy.ndimage import gaussian_filter1d
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
    from sklearn.decomposition import PCA
    pca = PCA(n_components=pca_dim).fit(h_concat)

    if mean_traj:
        trial_types = np.array(trial_types)
        unique_types = sorted(set(trial_types))
        min_len = min(h.shape[0] for h in all_h_pca)

        mean_trajs = [
            np.mean([hidden_np[i][:min_len] for i in range(len(trial_types)) if trial_types[i] == label], axis=0)
            for label in unique_types
        ]

        if align_traj:
            # assumes you have this util
            mean_trajs = align_trajectories(mean_trajs)

        all_h_pca = [pca.transform(h) for h in mean_trajs]
        trial_types = unique_types
        opacity = 1.0
    else:
        opacity = 0.3
        all_h_pca = [pca.transform(h) for h in all_h_pca]

    explained = np.sum(pca.explained_variance_ratio_[:pca_dim])
    print(f"PCA explained {explained * 100:.2f}% of variance")

    import plotly.graph_objects as go
    from matplotlib.cm import get_cmap

    # Prepare colours for trajectories (unchanged)
    cmap = get_cmap('tab20')
    colour_map_trials = {}
    colours = []
    is_per_timestep = False
    if trial_types is not None:
        trial_types = np.array(trial_types)
        if trial_types.ndim == 2:
            is_per_timestep = True
            unique_types = sorted(set(trial_types.ravel()))
        else:
            unique_types = sorted(set(trial_types))
        colour_map_trials = {tt: cmap(i % cmap.N) for i, tt in enumerate(unique_types)}
        if not is_per_timestep:
            colours = [
                f'rgba({int(r*255)},{int(g*255)},{int(b*255)},{a})'
                for tt in trial_types
                for r, g, b, a in [colour_map_trials[tt]]
            ]

    created_local_fig = False
    if fig is None:
        fig = go.Figure()
        created_local_fig = True

    add_kwargs = {}
    if row is not None and col is not None:
        add_kwargs = dict(row=row, col=col)

    # Plot trajectories
    for idx, h_pca in enumerate(all_h_pca):
        if is_per_timestep:
            tt_series = trial_types[idx]
            colour_array = [colour_map_trials[tt] for tt in tt_series]
            rgba_list = [f'rgba({int(r*255)},{int(g*255)},{int(b*255)},{a})'
                         for r, g, b, a in colour_array]

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
            show_legend = False

            line_opts = dict(width=2)
            if line_colour:
                line_opts["color"] = line_colour

            if pca_dim == 3:
                fig.add_trace(go.Scatter3d(
                    x=h_pca[:, 0], y=h_pca[:, 1], z=h_pca[:, 2],
                    mode='lines', line=line_opts,
                    name=trace_label, opacity=opacity,
                    showlegend=show_legend
                ), **add_kwargs)
            else:
                fig.add_trace(go.Scatter(
                    x=h_pca[:, 0], y=h_pca[:, 1],
                    mode='lines', line=line_opts,
                    name=trace_label, opacity=opacity,
                    showlegend=show_legend
                ), **add_kwargs)

    # --- FIXED POINTS: colour by memory_labels, shape by dynamics_labels, one legend entry per unique combined label ---
    if fixed_points is not None and len(fixed_points) > 0:
        fp_proj = pca.transform(fixed_points)

        # Build (or receive) colour map keyed by memory label
        if memory_labels is None or dynamics_labels is None:
            # fall back to a single black 'x' if labels not provided
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
        else:
            memory_labels = list(memory_labels)
            dynamics_labels = list(dynamics_labels)
            if len(memory_labels) != len(fp_proj) or len(dynamics_labels) != len(fp_proj):
                raise ValueError("memory_labels and dynamics_labels must match number of fixed_points.")

            # default shapes matching your matplotlib map
            shape_map = {"attractor": "circle", "saddle": "square", "repeller": "x"}

            # global colour map (consistent across grid) or local if not provided
            if colour_map is None:
                unique_mem = sorted(set(memory_labels))
                cmap_fp = get_cmap('tab20')
                colour_map = {
                    m: f'rgba({int(cmap_fp(i % cmap_fp.N)[0]*255)},'
                       f'{int(cmap_fp(i % cmap_fp.N)[1]*255)},'
                       f'{int(cmap_fp(i % cmap_fp.N)[2]*255)},'
                       f'{cmap_fp(i % cmap_fp.N)[3]})'
                    for i, m in enumerate(unique_mem)
                }

            # registry on the figure so legend entries are only added once across subplots
            if not hasattr(fig, "_shown_labels"):
                fig._shown_labels = set()

            for i, coords in enumerate(fp_proj):
                mem = memory_labels[i]
                dyn = dynamics_labels[i]
                combined = f"{mem} ({dyn})"
                colour = colour_map.get(mem, 'black')
                symbol = shape_map.get(dyn, "circle")

                show_this = combined not in fig._shown_labels
                if show_this:
                    fig._shown_labels.add(combined)

                marker_dict = dict(size=8 if pca_dim != 3 else 6, color=colour, symbol=symbol)

                if pca_dim == 3:
                    fig.add_trace(go.Scatter3d(
                        x=[coords[0]], y=[coords[1]], z=[coords[2]],
                        mode='markers', marker=marker_dict,
                        name=combined, showlegend=show_this,
                        legendgroup=combined
                    ), **add_kwargs)
                else:
                    fig.add_trace(go.Scatter(
                        x=[coords[0]], y=[coords[1]],
                        mode='markers', marker=marker_dict,
                        name=combined, showlegend=show_this,
                        legendgroup=combined
                    ), **add_kwargs)

    if created_local_fig:
        fig.update_layout(
            title=title,
            width=800, height=600,
            margin=dict(l=0, r=220, t=50, b=0),
            legend=dict(x=1.02, y=0.5, yanchor='middle', xanchor='left')  # right & centred
        )
        if save_path:
            fig.write_html(save_path)
            print(f"Saved to {save_path}")
        elif show:
            fig.show()

    return fig


def plot_multiple_trajectories_plotly_grid(
    hidden_states, overall_title,
    burn_in=0, labels=None,
    add_noise=False, pca_dim=2,
    smooth=False, sigma=1, downsample=None,
    trial_types=None, mean_traj=False,
    align_traj=False, save_path=None,
    fixed_points=None,
    memory_labels_list=None,          # NEW: list aligned with hidden_states
    dynamics_labels_list=None         # NEW: list aligned with hidden_states
):
    import math
    from plotly.subplots import make_subplots

    n = len(hidden_states)
    cols = math.ceil(n / 2)
    rows = 1 if n == 1 else 2

    subplot_type = 'scene' if pca_dim == 3 else 'xy'
    specs = [[{'type': subplot_type} for _ in range(cols)] for _ in range(rows)]

    titles = labels if labels else [f"Trajectory {i + 1}" for i in range(n)]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        specs=specs, vertical_spacing=0.08, horizontal_spacing=0.05)

    # Build a global colour map for memory labels so colours are consistent across the grid
    global_colour_map = None
    if memory_labels_list is not None:
        all_mems = []
        for ml in memory_labels_list:
            if ml is not None:
                all_mems.extend(list(ml))
        unique_mems = sorted(set(all_mems))
        cmap = get_cmap('tab20')
        global_colour_map = {
            m: f'rgba({int(cmap(i % cmap.N)[0]*255)},'
               f'{int(cmap(i % cmap.N)[1]*255)},'
               f'{int(cmap(i % cmap.N)[2]*255)},'
               f'{cmap(i % cmap.N)[3]})'
            for i, m in enumerate(unique_mems)
        }

    for i, hs in enumerate(hidden_states):
        r = 1 + (i // cols)
        c = 1 + (i % cols)

        fp_i = None if fixed_points is None else fixed_points[i]
        ml_i = None if memory_labels_list is None else memory_labels_list[i]
        dl_i = None if dynamics_labels_list is None else dynamics_labels_list[i]

        plot_trajectories_plotly_grid(
            hs, titles[i], trial_types=trial_types,
            pca_dim=pca_dim, smooth=smooth, sigma=sigma,
            downsample=downsample, mean_traj=mean_traj,
            align_traj=align_traj, fig=fig, row=r, col=c,
            show=False, fixed_points=fp_i,
            memory_labels=ml_i, dynamics_labels=dl_i,
            colour_map=global_colour_map
        )

    fig.update_layout(
        title_text=overall_title,
        height=700, width=1200,
        legend=dict(x=1.02, y=0.5, yanchor='middle', xanchor='left')  # global legend position
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved to {save_path}")
    else:
        fig.show()

    return fig

### DSA
def get_hs_for_dsa(rnn, input_transitions, n_bits=2, add_noise=False, burn_in=0, pca_dim=None, smooth=False, sigma=1, downsample=None):
    input_list, _, _ = get_perimeter_input(transitions=input_transitions, pulse_width=3, n_bits=2)
    if n_bits == 3:
        zeros   = torch.zeros_like(input_list[..., :1])       # shape (1,500,1)
        input_list = torch.cat([input_list, zeros], dim=-1)       # shape (1,500,3)
        input_list[0, 0:10, 2]   = 1.0
    hidden_states = get_hidden_states(rnn, input_list, add_noise=add_noise, burn_in=burn_in, pca_dim=pca_dim, smooth=smooth, sigma=sigma, downsample=downsample)
    return hidden_states

def get_hs_for_dsa_random(rnn, T, batch_size, n_bits=2, pulse_prob=pulse_prob, add_noise=False, burn_in=0, pca_dim=None, smooth=False, sigma=1, downsample=None, squeeze=True):
    input_list, _ = generate_batch(batch_size, T, 2, pulse_prob, device=rnn.device)
    if n_bits == 3:
        zeros   = torch.zeros_like(input_list[..., :1])       # shape (1,500,1)
        input_list = torch.cat([input_list, zeros], dim=-1)       # shape (1,500,3)
        input_list[0, 0:10, 2]   = 1.0
    hidden_states = get_hidden_states(rnn, input_list, add_noise=add_noise, burn_in=burn_in, pca_dim=pca_dim, smooth=smooth, sigma=sigma, downsample=downsample, squeeze=squeeze)
    return hidden_states

def normalise_invert_dsa_angular(dsa_scores):
    """Normalise angles in [0, π] to [0, 1] and invert."""
    scores = np.array(dsa_scores, dtype=float)
    return 1 - scores / np.pi

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

### Similarity Plotting
def plot_grouped_bars(list1, list2, list3, labels, title="Grouped Bar Chart", log_scale=False):
    n = len(labels)
    ind = np.arange(n)  # x locations for groups
    width = 0.25  # bar width

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(ind - width, list1, width, label='No Noise', color='C0')
    ax.bar(ind, list2, width, label='Same Noise', color='C1')
    ax.bar(ind + width, list3, width, label='Different Noise', color='C2')

    ax.set_xticks(ind)
    ax.set_xticklabels(labels, rotation=-30, ha='left')
    ax.legend()
    ax.set_ylabel('Score')
    ax.set_title(title)
    if log_scale:
        ax.set_yscale('log')
    plt.tight_layout()
    plt.show()

def plot_dsa_procrustes(dsa_scores, procrustes_scores, labels, title="Grouped Bar Chart", log_scale=False):
    n = len(labels)
    ind = np.arange(n)
    width = 0.35  # wider bars for two groups

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(ind - width/2, dsa_scores, width, label='DSA', color='C0')
    ax.bar(ind + width/2, procrustes_scores, width, label='Procrustes', color='C1')

    ax.set_xticks(ind)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylabel('Score')
    ax.set_title(title)
    if log_scale:
        ax.set_yscale('log')
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

def plot_similarity_matrices(dsa_raw, procrustes_raw,
                             dsa_smooth, procrustes_smooth,
                             labels=None, save_path=None):
    """
    Make a 2x2 grid of heatmaps:
      columns: DSA, Procrustes
      rows: Raw, Smooth
    """

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    cmap = "Greens"
    # Top row: Raw
    sns.heatmap(dsa_raw, xticklabels=False, yticklabels=labels,
                cmap=cmap, ax=axs[0, 0])
    axs[0, 0].set_title("DSA Similarity (Raw)")

    sns.heatmap(procrustes_raw, xticklabels=False, yticklabels=False,
                cmap=cmap, ax=axs[0, 1])
    axs[0, 1].set_title("Procrustes Similarity (Raw)")

    # Bottom row: Smooth
    sns.heatmap(dsa_smooth, xticklabels=labels, yticklabels=labels,
                cmap=cmap, ax=axs[1, 0])
    axs[1, 0].set_title("DSA Similarity (Smooth)")

    sns.heatmap(procrustes_smooth, xticklabels=labels, yticklabels=False,
                cmap=cmap, ax=axs[1, 1])
    axs[1, 1].set_title("Procrustes Similarity (Smooth)")

    # Tick formatting only for the axes that have labels
    for ax in [axs[1,0], axs[1,1]]:
        ax.tick_params(axis='x', labelsize=8, rotation=90)
    for ax in [axs[0,0], axs[1,0]]:
        ax.tick_params(axis='y', labelsize=8)

    #plt.suptitle("Similarity Analysis", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def plot_similarity_matrices_raw(dsa_raw, procrustes_raw,
                                 labels=None, save_path=None, cmap="Greens"):
    """
    1x2 grid of similarity heatmaps (raw only):
      [ DSA | Procrustes ]
    - y-tick labels only on the left plot
    - x-tick labels along the bottom (only row)
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Left: DSA (show y labels)
    sns.heatmap(dsa_raw, xticklabels=labels, yticklabels=labels,
                cmap=cmap, ax=axs[0])
    axs[0].set_title("DSA Similarity")

    # Right: Procrustes (hide y labels)
    sns.heatmap(procrustes_raw, xticklabels=labels, yticklabels=False,
                cmap=cmap, ax=axs[1])
    axs[1].set_title("Procrustes Similarity")

    # Ticks
    for ax in axs:
        ax.tick_params(axis='x', labelsize=8, rotation=90)
    axs[0].tick_params(axis='y', labelsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


