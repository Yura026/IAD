"""
statistics_2d.py — Первинний статистичний аналіз, оцінки параметрів,
довірчі інтервали (Стьюдент, Хі-квадрат), перевірка кореляції t-тестом,
2D-групування та критерій узгодженості Хі-квадрат Пірсона.
Всі формули реалізовано власноруч без сторонніх статистичних бібліотек (SciPy — тільки для квантилів).
"""

import numpy as np
from scipy import stats


def compute_point_estimates(x: np.ndarray, y: np.ndarray):
    """Обчислення вибіркових точкових оцінок за власними формулами (п. 1.2)."""
    n = len(x)
    if n != len(y) or n < 4:
        raise ValueError("Розмірності вибірок X та Y мають збігатися (N >= 4).")

    # 1. Середні
    x_bar = float(np.sum(x) / n)
    y_bar = float(np.sum(y) / n)

    # 2. Незсунені дисперсії та СКВ (ділення на N - 1)
    s2_x = float(np.sum((x - x_bar) ** 2) / (n - 1))
    s2_y = float(np.sum((y - y_bar) ** 2) / (n - 1))
    s_x = float(np.sqrt(s2_x))
    s_y = float(np.sqrt(s2_y))

    # 3. Коваріація
    k_xy = float(np.sum((x - x_bar) * (y - y_bar)) / (n - 1))

    # 4. Коефіцієнт кореляції Пірсона
    denom = np.sqrt(np.sum((x - x_bar) ** 2) * np.sum((y - y_bar) ** 2))
    r_xy = float(np.sum((x - x_bar) * (y - y_bar)) / denom) if denom > 0 else 0.0
    r_xy = float(np.clip(r_xy, -1.0, 1.0))

    return {
        "N": n,
        "x_bar": x_bar,
        "y_bar": y_bar,
        "s2_x": s2_x,
        "s2_y": s2_y,
        "s_x": s_x,
        "s_y": s_y,
        "k_xy": k_xy,
        "r_xy": r_xy
    }


def compute_confidence_intervals(estimates: dict, gamma: float = 0.95):
    """
    Довірчі інтервали (п. 1.2):
    - для математичних сподівань mx, my: розподіл Стьюдента t(N-1)
    - для дисперсій sigma_x^2, sigma_y^2: розподіл Хі-квадрат chi2(N-1)
    """
    n = estimates["N"]
    alpha = 1.0 - gamma
    nu = n - 1

    # Квантиль Стьюдента t_{1 - alpha/2; N-1}
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=nu))
    delta_mx = t_crit * estimates["s_x"] / np.sqrt(n)
    delta_my = t_crit * estimates["s_y"] / np.sqrt(n)

    mx_ci = (estimates["x_bar"] - delta_mx, estimates["x_bar"] + delta_mx)
    my_ci = (estimates["y_bar"] - delta_my, estimates["y_bar"] + delta_my)

    # Квантилі Хі-квадрат chi2_{alpha/2; N-1} та chi2_{1 - alpha/2; N-1}
    chi2_left = float(stats.chi2.ppf(alpha / 2.0, df=nu))
    chi2_right = float(stats.chi2.ppf(1.0 - alpha / 2.0, df=nu))

    s2x_ci = ((nu * estimates["s2_x"]) / chi2_right, (nu * estimates["s2_x"]) / chi2_left)
    s2y_ci = ((nu * estimates["s2_y"]) / chi2_right, (nu * estimates["s2_y"]) / chi2_left)

    return {
        "gamma": gamma,
        "alpha": alpha,
        "nu": nu,
        "t_crit": t_crit,
        "mx_ci": mx_ci,
        "my_ci": my_ci,
        "chi2_left": chi2_left,
        "chi2_right": chi2_right,
        "s2x_ci": s2x_ci,
        "s2y_ci": s2y_ci
    }


def test_correlation_significance(r_xy: float, n: int, alpha: float = 0.05):
    """
    Перевірка нульової гіпотези H0: rho = 0 проти H1: rho != 0.
    t_calc = (r * sqrt(N - 2)) / sqrt(1 - r^2)
    """
    if abs(r_xy) >= 1.0:
        return {"t_calc": float('inf'), "t_crit": 0.0, "nu": n - 2, "is_significant": True, "p_value": 0.0}

    nu = n - 2
    t_calc = float((r_xy * np.sqrt(nu)) / np.sqrt(1.0 - r_xy ** 2))
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=nu))
    is_significant = abs(t_calc) > t_crit
    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_calc), df=nu)))

    return {
        "t_calc": t_calc,
        "t_crit": t_crit,
        "nu": nu,
        "is_significant": is_significant,
        "p_value": p_val
    }


def compute_2d_grouping(x: np.ndarray, y: np.ndarray, k: int = 7):
    """
    Розбиття областей значень X і Y на k класів кожну (п. 1.4).
    Повертає матрицю абсолютних частот nij, відносних частот pij*, межі та центри класів.
    """
    n = len(x)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))

    # Кроки розбиття
    hx = (xmax - xmin) / k
    hy = (ymax - ymin) / k

    # Межі класів
    x_edges = np.linspace(xmin, xmax, k + 1)
    y_edges = np.linspace(ymin, ymax, k + 1)

    # Центри класів (x_i^c, y_j^c)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    # Двовимірний підрахунок частот n_ij та p_ij* = n_ij / N
    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    rel_freqs = counts / n

    return {
        "k": k,
        "hx": hx,
        "hy": hy,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "counts": counts,
        "rel_freqs": rel_freqs
    }


def density_2d(xv, yv, mx: float, my: float, sx: float, sy: float, r: float):
    """Теоретична щільність двовимірного нормального розподілу f(x, y) (п. 1.5)."""
    denom = 2.0 * np.pi * sx * sy * np.sqrt(1.0 - r ** 2)
    q = (
        ((xv - mx) / sx) ** 2
        - 2.0 * r * ((xv - mx) / sx) * ((yv - my) / sy)
        + ((yv - my) / sy) ** 2
    )
    return (1.0 / denom) * np.exp(-q / (2.0 * (1.0 - r ** 2)))


def test_chi2_goodness_of_fit(x: np.ndarray, y: np.ndarray, k: int = 7, alpha: float = 0.05):
    """
    Перевірка достовірності відтворення за критерієм Хі-квадрат Пірсона (п. 1.6).
    Теоретична ймовірність p_ij = f(x_i^c, y_j^c) * hx * hy.
    Класи з N * p_ij < 5 об'єднуються із сусідніми.
    Число ступенів вільності: nu = K_pooled - 1 - 5.
    """
    n = len(x)
    pts = compute_point_estimates(x, y)
    grp = compute_2d_grouping(x, y, k=k)

    hx, hy = grp["hx"], grp["hy"]
    xc, yc = grp["x_centers"], grp["y_centers"]
    counts = grp["counts"]

    # Сітка центрів комірок (x_i^c, y_j^c)
    xc_grid, yc_grid = np.meshgrid(xc, yc, indexing='ij')

    # Теоретичні ймовірності строго за формулою п. 1.6: p_ij = f(x_i^c, y_j^c) * hx * hy
    p_theo = density_2d(xc_grid, yc_grid, pts["x_bar"], pts["y_bar"], pts["s_x"], pts["s_y"], pts["r_xy"]) * hx * hy
    expected = n * p_theo

    # Розгортаємо у одновимірні масиви для виконання процедури об'єднання (pooling) класів з N*p_ij < 5
    flat_o = counts.flatten()
    flat_e = expected.flatten()

    pooled_o = []
    pooled_e = []
    accum_o = 0.0
    accum_e = 0.0

    for obs_val, exp_val in zip(flat_o, flat_e):
        if exp_val >= 5.0 and accum_e == 0.0:
            pooled_o.append(obs_val)
            pooled_e.append(exp_val)
        else:
            accum_o += obs_val
            accum_e += exp_val
            if accum_e >= 5.0:
                pooled_o.append(accum_o)
                pooled_e.append(accum_e)
                accum_o = 0.0
                accum_e = 0.0

    # Залишок додаємо до останньої сформованої групи
    if accum_e > 0.0:
        if len(pooled_o) > 0:
            pooled_o[-1] += accum_o
            pooled_e[-1] += accum_e
        else:
            pooled_o.append(accum_o)
            pooled_e.append(accum_e)

    arr_o = np.array(pooled_o)
    arr_e = np.array(pooled_e)
    k_pooled = len(arr_o)

    # Розрахунок статистики Хі-квадрат
    chi2_calc = float(np.sum(((arr_o - arr_e) ** 2) / arr_e))

    # Ступені вільності: nu = K_pooled - 1 - 5 (5 оцінених параметрів: x_bar, y_bar, s_x, s_y, r_xy)
    nu = k_pooled - 1 - 5
    if nu < 1:
        nu = 1  # захист на випадок малої кількості класів після об'єднання

    chi2_crit = float(stats.chi2.ppf(1.0 - alpha, df=nu))
    p_value = float(1.0 - stats.chi2.cdf(chi2_calc, df=nu))
    is_valid = chi2_calc <= chi2_crit

    return {
        "chi2_calc": chi2_calc,
        "chi2_crit": chi2_crit,
        "df": nu,
        "k_initial": k * k,
        "k_pooled": k_pooled,
        "p_value": p_value,
        "is_valid": is_valid
    }