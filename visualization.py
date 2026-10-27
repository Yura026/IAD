import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_time_series(raw_df, cleaned_df, missing_idx, anomaly_idx, real_events_idx=None, ax=None):
    """
    Графік добової динаміки з чітким маркуванням:
    - пропусків (фіолетовий круг)
    - технічних аномалій (червоний хрестик)
    - реальних подій (бурштиновий ромб)
    - екстремумів Min / Max
    """
    if real_events_idx is None:
        real_events_idx = []
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        
    ax.clear()
    
    cleaned = cleaned_df.dropna(subset=['value']).copy()
    if cleaned.empty:
        return ax
        
    times = cleaned['time'].astype(str).tolist()
    vals = cleaned['value'].tolist()
    x_positions = range(len(cleaned))
    cleaned_indices = cleaned.index.tolist()
    
    ax.plot(x_positions, vals, label='Показник', color='royalblue', lw=2, marker='o', markersize=3)
    
    # Позначення замінених пропусків
    cur_missing_x = [i for i, orig_idx in enumerate(cleaned_indices) if orig_idx in missing_idx]
    if cur_missing_x:
        ax.scatter(cur_missing_x, [vals[i] for i in cur_missing_x], color='purple', s=70, zorder=5, label='Замінений пропуск')
        
    # Позначення технічних аномалій (замінених)
    cur_tech_x = [i for i, orig_idx in enumerate(cleaned_indices) if orig_idx in anomaly_idx and orig_idx not in real_events_idx]
    if cur_tech_x:
        ax.scatter(cur_tech_x, [vals[i] for i in cur_tech_x], color='red', s=80, marker='x', zorder=5, label='Технічна аномалія')
        
    # Позначення реальних подій (залишених у вибірці)
    cur_real_x = [i for i, orig_idx in enumerate(cleaned_indices) if orig_idx in real_events_idx]
    if cur_real_x:
        ax.scatter(cur_real_x, [vals[i] for i in cur_real_x], color='darkorange', s=90, marker='D', zorder=5, label='Реальна подія')

    # Екстремуми Min / Max
    max_i = int(np.argmax(vals))
    min_i = int(np.argmin(vals))
    ax.scatter([max_i], [vals[max_i]], color='green', s=100, marker='^', zorder=6, label=f'Макс: {vals[max_i]:.1f} ({times[max_i]})')
    ax.scatter([min_i], [vals[min_i]], color='brown', s=100, marker='v', zorder=6, label=f'Мін: {vals[min_i]:.1f} ({times[min_i]})')

    ax.set_title("Часовий ряд показника за добу", fontsize=11, fontweight='bold')
    ax.set_xlabel("Час доби", fontsize=9)
    ax.set_ylabel("Кількість", fontsize=9)
    
    step = max(1, len(times) // 10)
    ticks = list(range(0, len(times), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([times[i] for i in ticks], rotation=45)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)
    return ax

def plot_histogram(data, num_bins=None, ax=None):
    """
    Гістограма відносних частот за формулою Стерджеса або вказаною кількістю класів k.
    Повертає значення k та кроку h.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.clear()
    
    x = np.array(data, dtype=float)
    N = len(x)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    
    # Визначення кількості класів k та кроку розбиття h
    if num_bins is not None and num_bins > 0:
        k = int(num_bins)
        bins = np.linspace(x_min, x_max, k + 1)
        h = (x_max - x_min) / k
        mode_text = f"ручна к-ть класів k={k}, h={h:.2f}"
    else:
        # За правилом Стерджеса: k = 1 + 3.322 * lg(N)
        k = int(np.round(1 + 3.322 * np.log10(N)))
        bins = np.linspace(x_min, x_max, k + 1)
        h = (x_max - x_min) / k
        mode_text = f"Стерджес (k={k}, h={h:.2f})"
        
    counts, edges = np.histogram(x, bins=bins)
    rel_freqs = counts / N
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_widths = edges[1:] - edges[:-1]
    
    ax.bar(bin_centers, rel_freqs, width=bin_widths * 0.9, color='skyblue', edgecolor='black', alpha=0.8)
    ax.set_title(f"Гістограма відносних частот [{mode_text}]", fontsize=11, fontweight='bold')
    ax.set_xlabel("Значення показника", fontsize=9)
    ax.set_ylabel("Відносна частота (p_i)", fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.6)
    return ax, k, h

def plot_distribution_functions(data, gamma=0.95, ax=None):
    """
    Графік емпіричної функції розподілу F_n(x), теоретичної нормальної кривої
    та сходинкової довірчої смуги (Колмогоров-Смирнов).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.clear()
    
    x = np.sort(np.array(data, dtype=float))
    N = len(x)
    x_bar = np.mean(x)
    s = np.std(x, ddof=1)
    
    # Сітка по осі X
    grid_x = np.linspace(np.min(x) - 0.5 * s, np.max(x) + 0.5 * s, 300)
    norm_cdf = stats.norm.cdf(grid_x, loc=x_bar, scale=s)
    
    # Довірча межа: epsilon = sqrt(ln(2 / alpha) / (2 * N))
    alpha = 1.0 - gamma
    eps = np.sqrt(np.log(2.0 / alpha) / (2.0 * N))
    
    # Точна ступінчаста ЕФР для довірчої смуги
    fn_steps = np.searchsorted(x, grid_x, side='right') / N
    upper_band = np.clip(fn_steps + eps, 0.0, 1.0)
    lower_band = np.clip(fn_steps - eps, 0.0, 1.0)
    
    # Перевірка узгодженості
    is_compatible = np.all((norm_cdf >= lower_band) & (norm_cdf <= upper_band))
    fit_status = "Узгоджується з нормальним" if is_compatible else "Відхиляється від нормального"
    
    # Побудова графіків
    fn_y = np.arange(1, N + 1) / N
    ax.step(x, fn_y, where='post', color='darkblue', lw=2, label='Емпірична Fn(x)')
    ax.plot(grid_x, norm_cdf, color='red', linestyle='--', lw=2, label=f'Теоретична N({x_bar:.1f}, {s:.1f})')
    ax.fill_between(grid_x, lower_band, upper_band, color='royalblue', alpha=0.18, step='post', label=f'Довірча смуга (γ={gamma}, ε={eps:.3f})')
    
    ax.set_title(f"Функції розподілу та довірча смуга ({fit_status})", fontsize=11, fontweight='bold')
    ax.set_xlabel("Значення x", fontsize=9)
    ax.set_ylabel("F(x)", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right', fontsize=8)
    return ax