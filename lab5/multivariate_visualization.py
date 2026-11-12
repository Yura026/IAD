"""
multivariate_visualization.py — Побудова всіх обов'язкових діаграм ЛР 5 (п. 4.3 та п. 6):
1) Тривимірне кореляційне поле в осях x1, x2, x3 з можливістю обертання;
2) Матриця діаграм розсіювання 3x3 (кореляційні поля, оцінки r_{i,j}, гістограми на діагоналі);
3) Графік паралельних координат (для очищеної або початкової вибірки з виділенням аномалій);
4) Теплова карта кореляційної матриці R{xi} з дивергентною шкалою [-1; +1];
5) Бульбашкова діаграма (радіус R = sqrt(S / pi), S — нормоване значення x3);
6) Діаграма Найтінгейл (полярна діаграма);
7) Scree Plot (графік кам'янистого осипу власних чисел / часток дисперсії);
8) Біплот (проекція спостережень на перші дві ГК + вектори ознак).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_3d_scatter(raw_data: np.ndarray, clean_mask: np.ndarray, names: list, show_outliers: bool = True, ax=None):
    """
    1. Тривимірне кореляційне поле в осях x1, x2, x3 (п. 4.3.1).
    Дозволяє відображати як очищену вибірку, так і початкову з виділенням аномалій іншим кольором.
    """
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')

    ax.clear()
    clean = raw_data[clean_mask]
    outliers = raw_data[~clean_mask]

    ax.scatter(
        clean[:, 0], clean[:, 1], clean[:, 2],
        c='royalblue', edgecolors='k', linewidth=0.4, s=32, alpha=0.75,
        label=f"Очищені дані (N'={len(clean)})"
    )

    if show_outliers and len(outliers) > 0:
        ax.scatter(
            outliers[:, 0], outliers[:, 1], outliers[:, 2],
            c='crimson', marker='^', s=85, edgecolors='black', linewidth=0.8,
            label=f"Відкинуті аномалії 3σ ({len(outliers)})"
        )
        title_str = "Тривимірне кореляційне поле (початкова вибірка з аномаліями)"
    else:
        title_str = "Тривимірне кореляційне поле (очищена вибірка)"

    ax.set_title(title_str, fontsize=9.5, fontweight='bold')
    ax.set_xlabel(names[0], fontsize=8)
    ax.set_ylabel(names[1], fontsize=8)
    ax.set_zlabel(names[2], fontsize=8)
    ax.legend(fontsize=8, loc='upper left')
    return ax


def plot_scatter_matrix_3x3(raw_data: np.ndarray, clean_mask: np.ndarray, r_mat: np.ndarray,
                            names: list, show_outliers: bool = False, fig=None):
    """
    2. Матриця діаграм розсіювання 3x3 (п. 4.3.1):
    - на діагоналі — гістограми ознак;
    - у нижньому трикутнику — кореляційні поля пар (x1,x2), (x1,x3), (x2,x3);
    - у симетричних верхніх клітинках — оцінки парних коефіцієнтів кореляції r_{i,j}.
    """
    if fig is None:
        fig, axes = plt.subplots(3, 3, figsize=(7, 6))
    else:
        fig.clear()
        axes = fig.subplots(3, 3)

    clean = raw_data[clean_mask]
    outliers = raw_data[~clean_mask]

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            if i == j:
                ax.hist(clean[:, i], bins=12, color='skyblue', edgecolor='black', alpha=0.8)
                ax.set_title(f"Гістограма {names[i]}", fontsize=8, fontweight='bold')
            elif i > j:
                ax.scatter(clean[:, j], clean[:, i], c='royalblue', s=14, alpha=0.6, label="Очищені")
                if show_outliers and len(outliers) > 0:
                    ax.scatter(outliers[:, j], outliers[:, i], c='crimson', marker='^', s=35, label="Аномалії")
                ax.grid(True, linestyle=':', alpha=0.5)
            else:
                r_val = r_mat[i, j]
                strength = "сильний" if abs(r_val) >= 0.7 else ("помірний" if abs(r_val) >= 0.4 else "слабкий")
                direction = "прямий" if r_val > 0 else "обернений"
                color = 'darkred' if abs(r_val) >= 0.5 else 'navy'
                ax.text(
                    0.5, 0.5,
                    f"r({names[i]},{names[j]}) = {r_val:+.4f}\n({strength} {direction})",
                    ha='center', va='center', fontsize=8.5, fontweight='bold', color=color
                )
                ax.set_facecolor('#f7f9fc')
                ax.set_xticks([])
                ax.set_yticks([])

            if j == 0 and i != j:
                ax.set_ylabel(names[i], fontsize=8)
            if i == 2:
                ax.set_xlabel(names[j], fontsize=8)
            ax.tick_params(labelsize=6.5)

    mode_str = "з виділенням аномалій" if show_outliers else "очищена вибірка"
    fig.suptitle(f"Матриця діаграм розсіювання 3×3 ({mode_str})", fontsize=10, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_parallel_coordinates(raw_data: np.ndarray, clean_mask: np.ndarray, names: list,
                              show_outliers: bool = True, ax=None):
    """
    3. Графік паралельних координат (п. 4.3.2):
    будується як для очищеної, так і для початкової вибірки з виділенням аномалій іншим кольором.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.clear()
    base_data = raw_data if show_outliers else raw_data[clean_mask]
    mins = np.min(base_data, axis=0)
    maxs = np.max(base_data, axis=0)
    span = np.where((maxs - mins) > 1e-12, maxs - mins, 1.0)

    norm_raw = (raw_data - mins) / span
    x_coords = np.arange(len(names))

    # Лінії очищених спостережень
    clean_norm = norm_raw[clean_mask]
    for idx, row in enumerate(clean_norm):
        lbl = f"Очищені (N'={len(clean_norm)})" if idx == 0 else None
        ax.plot(x_coords, row, color='royalblue', alpha=0.25, lw=1.1, label=lbl)

    # Лінії аномальних спостережень (якщо увімкнено показ початкової вибірки)
    if show_outliers:
        out_norm = norm_raw[~clean_mask]
        for idx, row in enumerate(out_norm):
            lbl = f"Відкинуті аномалії 3σ ({len(out_norm)})" if idx == 0 else None
            ax.plot(x_coords, row, color='crimson', alpha=0.9, lw=2.2, linestyle='--', marker='o', markersize=4, label=lbl)

    ax.set_xticks(x_coords)
    ax.set_xticklabels([f"{names[k]}\n[{mins[k]:.2f}; {maxs[k]:.2f}]" for k in range(len(names))], fontsize=8)
    ax.set_ylabel("Нормоване значення S ∈ [0; 1]", fontsize=8)
    title_mode = "початкова вибірка з аномаліями" if show_outliers else "очищена вибірка"
    ax.set_title(f"Графік паралельних координат ({title_mode})", fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8, loc='upper right')
    return ax


def plot_correlation_heatmap(r_mat: np.ndarray, names: list, fig=None):
    """
    4. Теплова карта кореляційної матриці R{xi} з дивергентною шкалою [-1; +1]
    та числовими значеннями у клітинках (п. 4.3.3).
    """
    if fig is None:
        fig = plt.figure(figsize=(5.5, 4.5))
    else:
        fig.clear()

    ax = fig.add_subplot(111)
    cax = ax.imshow(r_mat, cmap='coolwarm', vmin=-1.0, vmax=1.0)
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Коефіцієнт кореляції r", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    for i in range(3):
        for j in range(3):
            val = r_mat[i, j]
            text_color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:+.4f}", ha='center', va='center',
                    fontsize=9.5, fontweight='bold', color=text_color)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(names, fontsize=8.5, fontweight='bold')
    ax.set_yticklabels(names, fontsize=8.5, fontweight='bold')
    ax.set_title("Теплова карта кореляційної матриці R{ξ}", fontsize=10, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_bubble_chart(clean_data: np.ndarray, names: list, idx_x=0, idx_y=1, idx_size=2, fig=None):
    """
    5. Бульбашкова діаграма (п. 4.3.4, п. 5.6):
    - вісь абсцис — x1, вісь ординат — x2;
    - min-max нормування S_{k,l} = (x - min) / (max - min) in [0, 1];
    - масштабування у [S_min, S_max], радіус R = sqrt(S / pi), площа бульбашки pi * R^2 = S.
    """
    if fig is None:
        fig = plt.figure(figsize=(6, 4.5))
    else:
        fig.clear()

    ax = fig.add_subplot(111)
    x_vals = clean_data[:, idx_x]
    y_vals = clean_data[:, idx_y]
    z_vals = clean_data[:, idx_size]

    # Формула п. 5.6: Min-max нормування у [0, 1] та відображення у [S_min, S_max]
    z_min, z_max = float(np.min(z_vals)), float(np.max(z_vals))
    span = z_max - z_min if (z_max - z_min) > 1e-12 else 1.0
    s_norm_01 = (z_vals - z_min) / span

    s_min_area, s_max_area = 25.0, 280.0
    s_scaled = s_min_area + s_norm_01 * (s_max_area - s_min_area)
    radii = np.sqrt(s_scaled / np.pi)
    areas_for_scatter = np.pi * (radii ** 2)  # у matplotlib параметр s — це площа S = pi * R^2

    scatter = ax.scatter(
        x_vals, y_vals, s=areas_for_scatter, c=z_vals,
        cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.7
    )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Значення {names[idx_size]}", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(
        f"Бульбашкова діаграма: {names[idx_x]} vs {names[idx_y]} (площа S ~ {names[idx_size]}, R=√(S/π))",
        fontsize=9.5, fontweight='bold'
    )
    ax.set_xlabel(names[idx_x], fontsize=8.5)
    ax.set_ylabel(names[idx_y], fontsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    return fig


def plot_nightingale_rose(clean_data: np.ndarray, names: list, ax=None):
    """
    6. Діаграма Найтінгейл (Nightingale Rose Diagram, п. 6) для порівняння середніх та СКВ ознак.
    """
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, polar=True)

    ax.clear()
    means = np.mean(clean_data, axis=0)
    stds = np.std(clean_data, axis=0, ddof=1)
    p = len(names)

    angles = np.linspace(0, 2 * np.pi, p, endpoint=False)
    width = 2 * np.pi / p

    # Радіуси секторів пропорційні середнім значенням та СКВ
    ax.bar(
        angles, means, width=width * 0.88, bottom=0.0,
        color=['#4e79a7', '#59a14f', '#e15759'], edgecolor='black', alpha=0.75, label='Середнє x̄_k'
    )
    ax.bar(
        angles, stds, width=width * 0.45, bottom=0.0,
        color=['#a0cbe8', '#8cd17d', '#ff9d9a'], edgecolor='black', alpha=0.9, label='СКВ σ_k'
    )

    ax.set_xticks(angles)
    ax.set_xticklabels([f"{names[k]}\nx̄={means[k]:.2f}, σ={stds[k]:.2f}" for k in range(p)], fontsize=8)
    ax.set_title("Діаграма Найтінгейл (співвідношення x̄_k та σ_k)", fontsize=9.5, fontweight='bold', pad=15)
    ax.legend(fontsize=7.5, loc='upper right', bbox_to_anchor=(1.2, 1.1))
    return ax


def plot_scree(eigvals: np.ndarray, var_shares: np.ndarray, cum_shares: np.ndarray, ax=None):
    """
    7. Scree Plot (графік кам'янистого осипу, п. 6, п. 7):
    відображає частку дисперсії кожної ГК (%), накопичену частку (%) та поріг відкидання 5%.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.clear()
    x_pos = np.arange(1, len(var_shares) + 1)

    ax.bar(x_pos, var_shares, width=0.5, color='steelblue', edgecolor='black', alpha=0.85, label='Частка дисперсії λ_k / n (%)')
    ax.plot(x_pos, cum_shares, color='darkorange', marker='o', lw=2.2, label='Накопичена частка Σ% дисперсії')
    ax.axhline(5.0, color='crimson', linestyle='--', lw=1.5, label='Поріг відкидання (5%)')

    for i, (lam, v, c) in enumerate(zip(eigvals, var_shares, cum_shares)):
        ax.text(x_pos[i], v + 1.8, f"λ={lam:.3f}\n({v:.2f}%)", ha='center', fontsize=7.5, fontweight='bold')
        ax.text(x_pos[i], min(103.0, c + 2.0), f"Σ={c:.2f}%", ha='center', fontsize=7.5, color='chocolate', fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"ГК x'_{i}" for i in x_pos], fontsize=8.5)
    ax.set_ylabel("Частка дисперсії, %", fontsize=8.5)
    ax.set_title("Scree Plot (кам'янистий осип головних компонент)", fontsize=10, fontweight='bold')
    ax.set_ylim(0, 114)
    ax.legend(fontsize=8, loc='center right')
    ax.grid(True, linestyle='--', alpha=0.5)
    return ax


def plot_biplot(scores: np.ndarray, matrix_a: np.ndarray, var_shares: np.ndarray, names: list, ax=None):
    """
    8. Біплот (Biplot, п. 6, п. 7):
    проекція спостережень на площину перших двох головних компонент (x'_1, x'_2)
    разом із векторами коефіцієнтів (навантажень) вихідних ознак.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.clear()
    x_pc1 = scores[:, 0]
    y_pc2 = scores[:, 1]
    ax.scatter(x_pc1, y_pc2, c='cornflowerblue', edgecolors='navy', linewidth=0.4, s=22, alpha=0.65, label="Об'єкти (scores)")

    scale = max(np.max(np.abs(x_pc1)), np.max(np.abs(y_pc2))) * 0.8
    colors = ['crimson', 'darkgreen', 'darkmagenta']

    for v in range(len(names)):
        dx = matrix_a[0, v] * scale
        dy = matrix_a[1, v] * scale
        ax.arrow(0, 0, dx, dy, color=colors[v], width=0.03, head_width=0.18, length_includes_head=True, zorder=5)
        ax.text(
            dx * 1.14, dy * 1.14,
            f"{names[v]}\n({matrix_a[0, v]:+.2f}; {matrix_a[1, v]:+.2f})",
            color=colors[v], fontsize=8, fontweight='bold', ha='center', va='center'
        )

    ax.axhline(0, color='gray', linestyle=':', lw=1)
    ax.axvline(0, color='gray', linestyle=':', lw=1)
    ax.set_xlabel(f"Перша головна компонента x'₁ ({var_shares[0]:.2f}%)", fontsize=8.5)
    ax.set_ylabel(f"Друга головна компонента x'₂ ({var_shares[1]:.2f}%)", fontsize=8.5)
    ax.set_title("Біплот (Biplot у просторі перших двох головних компонент)", fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8, loc='upper left')
    return ax