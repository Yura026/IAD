"""
statistics_2d.py — Статистичне ядро для Лабораторних робіт № 3 та № 4:
- Фільтрація аномальних спостережень за правилом 3-сигма (ЛР 4, п. 3).
- Первинний статистичний аналіз та довірчі інтервали (ЛР 3, п. 1.2).
- Коефіцієнт Пірсона r, його t-тест та довірчий інтервал через z-перетворення Фішера (ЛР 4, п. 4.1, 5).
- Ранговий коефіцієнт кореляції Спірмена rho та його t-тест (ЛР 4, п. 1, 6, 7).
- Лінійна регресія Y = a + bX за МНК, залишкова дисперсія, СКВ оцінок, ДІ, t-тести для a і b, R^2 (ЛР 4, п. 4.2, 4.3, 5).
- 2D-групування та критерій узгодженості Хі-квадрат Пірсона (ЛР 3, п. 1.4, 1.6).
Усі формули реалізовано власноруч; SciPy використовується лише для квантилів розподілів.
"""

import numpy as np
from scipy import stats


def filter_outliers_3sigma(x: np.ndarray, y: np.ndarray, check_y: bool = False):
    """
    Відкидання аномальних спостережень за правилом 3-сигма (ЛР 4, п. 3):
    Значення вважається аномальним і відкидається, якщо |x_i - x_bar| > 3 * S_x
    (опціонально також |y_i - y_bar| > 3 * S_y).
    """
    n = len(x)
    x_bar = float(np.sum(x) / n)
    s_x = float(np.sqrt(np.sum((x - x_bar) ** 2) / (n - 1)))

    mask = np.abs(x - x_bar) <= 3.0 * s_x

    if check_y:
        y_bar = float(np.sum(y) / n)
        s_y = float(np.sqrt(np.sum((y - y_bar) ** 2) / (n - 1)))
        mask = mask & (np.abs(y - y_bar) <= 3.0 * s_y)

    x_clean = x[mask]
    y_clean = y[mask]
    removed_count = int(n - len(x_clean))

    return x_clean, y_clean, removed_count


def compute_point_estimates(x: np.ndarray, y: np.ndarray):
    """Обчислення вибіркових точкових оцінок за власними формулами."""
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
    r_xy = float(np.clip(r_xy, -0.9999999, 0.9999999))

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
    """Довірчі інтервали для mx, my, sigma_x^2, sigma_y^2 та r_xy (через z-перетворення Фішера)."""
    n = estimates["N"]
    alpha = 1.0 - gamma
    nu = n - 1

    # Квантиль Стьюдента t_{1 - alpha/2; N-1}
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=nu))
    delta_mx = t_crit * estimates["s_x"] / np.sqrt(n)
    delta_my = t_crit * estimates["s_y"] / np.sqrt(n)

    mx_ci = (estimates["x_bar"] - delta_mx, estimates["x_bar"] + delta_mx)
    my_ci = (estimates["y_bar"] - delta_my, estimates["y_bar"] + delta_my)

    # Квантилі Хі-квадрат
    chi2_left = float(stats.chi2.ppf(alpha / 2.0, df=nu))
    chi2_right = float(stats.chi2.ppf(1.0 - alpha / 2.0, df=nu))

    s2x_ci = ((nu * estimates["s2_x"]) / chi2_right, (nu * estimates["s2_x"]) / chi2_left)
    s2y_ci = ((nu * estimates["s2_y"]) / chi2_right, (nu * estimates["s2_y"]) / chi2_left)

    # Довірчий інтервал для коефіцієнта Пірсона r через z-перетворення Фішера (ЛР 4, п. 5)
    r = estimates["r_xy"]
    u_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_val = 0.5 * np.log((1.0 + r) / (1.0 - r))
    se_z = 1.0 / np.sqrt(max(1, n - 3))
    z_low = z_val - u_crit * se_z
    z_high = z_val + u_crit * se_z
    r_ci = (float(np.tanh(z_low)), float(np.tanh(z_high)))

    return {
        "gamma": gamma,
        "alpha": alpha,
        "nu": nu,
        "t_crit": t_crit,
        "u_crit": u_crit,
        "mx_ci": mx_ci,
        "my_ci": my_ci,
        "chi2_left": chi2_left,
        "chi2_right": chi2_right,
        "s2x_ci": s2x_ci,
        "s2y_ci": s2y_ci,
        "z_fisher": float(z_val),
        "r_ci": r_ci
    }


def test_correlation_significance(r_xy: float, n: int, alpha: float = 0.05):
    """Перевірка значущості коефіцієнта кореляції t-тестом (nu = N - 2)."""
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


def _compute_ranks(arr: np.ndarray) -> np.ndarray:
    """Власна функція обчислення рангів (від 1 до N) із усередненням для зв'язаних рангів."""
    n = len(arr)
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def compute_spearman_correlation(x: np.ndarray, y: np.ndarray, alpha: float = 0.05):
    """
    Обчислення рангового коефіцієнта кореляції Спірмена rho та його значущості (ЛР 4, п. 1, 6, 7).
    """
    n = len(x)
    rx = _compute_ranks(x)
    ry = _compute_ranks(y)

    rx_bar = np.sum(rx) / n
    ry_bar = np.sum(ry) / n

    denom = np.sqrt(np.sum((rx - rx_bar) ** 2) * np.sum((ry - ry_bar) ** 2))
    rho = float(np.sum((rx - rx_bar) * (ry - ry_bar)) / denom) if denom > 0 else 0.0
    rho = float(np.clip(rho, -0.9999999, 0.9999999))

    t_res = test_correlation_significance(rho, n, alpha=alpha)

    return {
        "rho": rho,
        "t_calc": t_res["t_calc"],
        "t_crit": t_res["t_crit"],
        "nu": t_res["nu"],
        "is_significant": t_res["is_significant"],
        "p_value": t_res["p_value"]
    }


def compute_linear_regression(x: np.ndarray, y: np.ndarray, gamma: float = 0.95):
    """
    Відтворення лінійної регресії Y = a + b*X методом найменших квадратів (МНК),
    оцінка точності й значущості параметрів a та b, коефіцієнт детермінації R^2 (ЛР 4, п. 4.2, 4.3, 5).
    """
    pts = compute_point_estimates(x, y)
    n = pts["N"]
    nu = n - 2
    alpha = 1.0 - gamma

    x_bar = pts["x_bar"]
    y_bar = pts["y_bar"]
    s_x = pts["s_x"]
    s_y = pts["s_y"]
    s2_x = pts["s2_x"]
    r_xy = pts["r_xy"]

    # 1. МНК-оцінки параметрів регресії (п. 5)
    b_hat = float(r_xy * (s_y / s_x))
    a_hat = float(y_bar - b_hat * x_bar)

    # Модельні значення та залишки
    y_hat = a_hat + b_hat * x
    residuals = y - y_hat
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y_bar) ** 2))

    # 2. Залишкова дисперсія S_зал^2 та СКВ S_зал (nu = N - 2)
    s2_res = float(ss_res / nu)
    s_res = float(np.sqrt(s2_res))

    # 3. Стандартні похибки (СКВ) оцінок параметрів S_b та S_a (п. 5)
    s_b = float(s_res / (s_x * np.sqrt(n - 1)))
    s_a = float(s_res * np.sqrt((1.0 / n) + (x_bar ** 2) / (s2_x * (n - 1))))

    # 4. Довірчі інтервали для a та b при nu = N - 2
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=nu))
    a_ci = (a_hat - t_crit * s_a, a_hat + t_crit * s_a)
    b_ci = (b_hat - t_crit * s_b, b_hat + t_crit * s_b)

    # 5. Перевірка значущості параметрів a та b за t-тестом
    t_a = float(a_hat / s_a) if s_a > 0 else 0.0
    t_b = float(b_hat / s_b) if s_b > 0 else 0.0

    p_val_a = float(2.0 * (1.0 - stats.t.cdf(abs(t_a), df=nu)))
    p_val_b = float(2.0 * (1.0 - stats.t.cdf(abs(t_b), df=nu)))

    sig_a = abs(t_a) > t_crit
    sig_b = abs(t_b) > t_crit

    # 6. Коефіцієнт детермінації R^2
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    r2 = float(np.clip(r2, 0.0, 1.0))

    return {
        "a_hat": a_hat,
        "b_hat": b_hat,
        "s2_res": s2_res,
        "s_res": s_res,
        "s_a": s_a,
        "s_b": s_b,
        "a_ci": a_ci,
        "b_ci": b_ci,
        "t_a": t_a,
        "t_b": t_b,
        "p_val_a": p_val_a,
        "p_val_b": p_val_b,
        "sig_a": sig_a,
        "sig_b": sig_b,
        "t_crit": t_crit,
        "nu": nu,
        "r2": r2
    }


def compute_2d_grouping(x: np.ndarray, y: np.ndarray, k: int = 7):
    """Розбиття областей значень X і Y на k класів кожну."""
    n = len(x)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))

    hx = (xmax - xmin) / k
    hy = (ymax - ymin) / k

    x_edges = np.linspace(xmin, xmax, k + 1)
    y_edges = np.linspace(ymin, ymax, k + 1)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

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
    """Теоретична щільність двовимірного нормального розподілу f(x, y)."""
    denom = 2.0 * np.pi * sx * sy * np.sqrt(1.0 - r ** 2)
    q = (
        ((xv - mx) / sx) ** 2
        - 2.0 * r * ((xv - mx) / sx) * ((yv - my) / sy)
        + ((yv - my) / sy) ** 2
    )
    return (1.0 / denom) * np.exp(-q / (2.0 * (1.0 - r ** 2)))


def test_chi2_goodness_of_fit(x: np.ndarray, y: np.ndarray, k: int = 7, alpha: float = 0.05):
    """Перевірка достовірності відтворення за критерієм Хі-квадрат Пірсона (ЛР 3, п. 1.6)."""
    n = len(x)
    pts = compute_point_estimates(x, y)
    grp = compute_2d_grouping(x, y, k=k)

    hx, hy = grp["hx"], grp["hy"]
    xc, yc = grp["x_centers"], grp["y_centers"]
    counts = grp["counts"]

    xc_grid, yc_grid = np.meshgrid(xc, yc, indexing='ij')
    p_theo = density_2d(xc_grid, yc_grid, pts["x_bar"], pts["y_bar"], pts["s_x"], pts["s_y"], pts["r_xy"]) * hx * hy
    expected = n * p_theo

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

    chi2_calc = float(np.sum(((arr_o - arr_e) ** 2) / arr_e))
    nu = max(1, k_pooled - 1 - 5)

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