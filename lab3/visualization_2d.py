"""
visualization_2d.py — Графічне відображення:
1) Кореляційне поле (Scatter Plot) + еліпси ліній рівня щільності f(x, y) (п. 1.3, 1.5).
2) 2D-гістограма (теплова карта частот) (п. 1.4).
3) 3D-стовпчикова просторова гістограма (п. 1.4).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statistics_2d import compute_point_estimates, compute_2d_grouping


def plot_scatter_and_ellipses(x: np.ndarray, y: np.ndarray, ax=None):
    """Побудова кореляційного поля та еліпсів розсіювання Q(x, y) / (1 - r^2) = c^2."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    ax.clear()
    pts = compute_point_estimates(x, y)
    ax.scatter(x, y, color='royalblue', alpha=0.6, edgecolors='k', s=25, label=f'Вибірка (N={len(x)})')
    ax.plot(pts["x_bar"], pts["y_bar"], marker='+', color='red', markersize=10, markeredgewidth=2, label='Центр (x̄, ȳ)')

    # Побудова ліній рівня відтвореної щільності f(x, y)
    margin_x = 3.5 * pts["s_x"]
    margin_y = 3.5 * pts["s_y"]
    grid_x = np.linspace(pts["x_bar"] - margin_x, pts["x_bar"] + margin_x, 200)
    grid_y = np.linspace(pts["y_bar"] - margin_y, pts["y_bar"] + margin_y, 200)
    gx, gy = np.meshgrid(grid_x, grid_y)

    # Нормована квадратична форма Махаланобіса: Q(x, y) / (1 - r^2)
    r = pts["r_xy"]
    q_norm = (
        ((gx - pts["x_bar"]) / pts["s_x"]) ** 2
        - 2.0 * r * ((gx - pts["x_bar"]) / pts["s_x"]) * ((gy - pts["y_bar"]) / pts["s_y"])
        + ((gy - pts["y_bar"]) / pts["s_y"]) ** 2
    ) / max(1e-12, (1.0 - r ** 2))

    # Рівні c^2 = [1, 4, 9] відповідають 1-sigma (39.3%), 2-sigma (86.5%), 3-sigma (98.9%)
    levels = [1.0, 4.0, 9.0]
    cs = ax.contour(gx, gy, q_norm, levels=levels, colors=['green', 'orange', 'red'], linewidths=[1.8, 1.8, 2.0])
    labels = {1.0: '1σ (39.3%)', 4.0: '2σ (86.5%)', 9.0: '3σ (98.9%)'}
    ax.clabel(cs, inline=True, fmt=lambda val: labels.get(val, f'{val:.1f}'), fontsize=8)

    ax.set_title(f"Кореляційне поле та еліпси розсіювання (r = {pts['r_xy']:.4f})", fontsize=10, fontweight='bold')
    ax.set_xlabel("Ознака X", fontsize=9)
    ax.set_ylabel("Ознака Y", fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8, loc='upper left')
    return ax


def plot_2d_heatmap(x: np.ndarray, y: np.ndarray, k: int = 7, ax=None):
    """Побудова 2D-гістограми (теплової карти частот n_ij та p_ij*) із числовими анотаціями."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    ax.clear()
    grp = compute_2d_grouping(x, y, k=k)
    counts = grp["counts"].T      # транспонуємо: вісь X горизонтально, вісь Y вертикально
    rel_freqs = grp["rel_freqs"].T
    xe = grp["x_edges"]
    ye = grp["y_edges"]

    mesh = ax.pcolormesh(xe, ye, counts, cmap='YlGnBu', edgecolors='gray', alpha=0.88)

    # Анотація частоти n_ij та відносної частоти p_ij* у кожній непорожній комірці
    xc = grp["x_centers"]
    yc = grp["y_centers"]
    max_c = np.max(counts)
    for i in range(k):
        for j in range(k):
            val = int(counts[j, i])
            rel_val = rel_freqs[j, i]
            if val > 0:
                txt_color = "white" if val > max_c * 0.6 else "black"
                label_text = f"{val}\n({rel_val:.3f})" if k <= 10 else f"{val}"
                ax.text(xc[i], yc[j], label_text, ha='center', va='center', fontsize=7.5, color=txt_color, fontweight='bold')

    ax.set_title(f"2D-гістограма частот n_ij та p_ij* (сітка {k} × {k})", fontsize=10, fontweight='bold')
    ax.set_xlabel("Класи ознаки X", fontsize=9)
    ax.set_ylabel("Класи ознаки Y", fontsize=9)
    return ax, mesh


def plot_3d_histogram(x: np.ndarray, y: np.ndarray, k: int = 7, ax=None):
    """Побудова 3D-стовпчикової гістограми розподілу частот."""
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')

    ax.clear()
    grp = compute_2d_grouping(x, y, k=k)
    counts = grp["counts"]
    xe = grp["x_edges"]
    ye = grp["y_edges"]

    # Позиції стовпчиків
    xpos, ypos = np.meshgrid(xe[:-1], ye[:-1], indexing='ij')
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)

    dx = grp["hx"] * 0.82
    dy = grp["hy"] * 0.82
    dz = counts.ravel()

    # Градієнтне забарвлення за висотою стовпчика
    max_dz = max(1.0, float(np.max(dz)))
    colors = plt.cm.viridis(dz / max_dz)

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, edgecolor='darkgray', alpha=0.9)

    ax.set_title(f"3D-гістограма частот n_ij (k={k})", fontsize=10, fontweight='bold')
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    ax.set_zlabel("Частота n_ij", fontsize=8)
    return ax