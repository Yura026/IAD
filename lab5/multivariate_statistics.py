"""
multivariate_statistics.py — Статистичні обчислення для ЛР 5 (за формулами п. 5 методички):
- описові статистики до та після чищення;
- багатовимірне правило 3-сигма (п. 5.1);
- вектор математичних сподівань E{xi}, матриці DC{xi} та R{xi} (п. 5.2);
- перевірка значущості парних коефіцієнтів кореляції;
- перевірка невиродженості кореляційної матриці (|R| != 0);
- часткові коефіцієнти кореляції за формулою 1-го порядку та через R^-1 + t-тест (п. 5.3);
- множинні коефіцієнти кореляції за прямою формулою та через R^-1 + F-тест (п. 5.4).
"""

import numpy as np
from scipy import stats


def compute_univariate_descriptors(data: np.ndarray, names: list):
    """
    Обчислення описових статистик для кожної ознаки: Min, Max, Середнє, СКВ (п. 4.1.2).
    За формулами п. 5.1 використовується незміщене СКВ з діленням на (N - 1).
    """
    n, p = data.shape
    desc = []
    for k in range(p):
        col = data[:, k]
        mean_val = float(np.sum(col) / n)
        sigma_val = float(np.sqrt(np.sum((col - mean_val) ** 2) / (n - 1)))
        desc.append({
            "name": names[k],
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "mean": mean_val,
            "std": sigma_val
        })
    return desc


def filter_outliers_3sigma_nd(data: np.ndarray, names: list):
    """
    Багатовимірне правило 3-сигма (п. 4.2, п. 5.1):
    Рядок X_l відкидається цілком, якщо хоча б для однієї ознаки |x_{k,l} - x_bar_k| > 3 * sigma_k.
    """
    n, p = data.shape
    means = np.sum(data, axis=0) / n
    stds = np.sqrt(np.sum((data - means) ** 2, axis=0) / (n - 1))

    is_outlier = np.zeros(n, dtype=bool)
    outlier_details = []

    for l_idx in range(n):
        row = data[l_idx]
        for k in range(p):
            diff = abs(row[k] - means[k])
            limit = 3.0 * stds[k]
            if diff > limit:
                is_outlier[l_idx] = True
                outlier_details.append({
                    "row_idx": l_idx + 1,  # 1-based номер рядка
                    "feature": names[k],
                    "val": float(row[k]),
                    "mean": float(means[k]),
                    "std": float(stds[k]),
                    "low_bound": float(means[k] - limit),
                    "upp_bound": float(means[k] + limit),
                    "diff_sigmas": float(diff / stds[k])
                })

    clean_data = data[~is_outlier].copy()
    return clean_data, outlier_details, ~is_outlier


def compute_multivariate_matrices(data: np.ndarray):
    """
    Власна реалізація обчислення вектора середніх E{xi},
    ДК-матриці DC{xi} та кореляційної матриці R{xi} (п. 5.2).
    """
    n, p = data.shape
    # 1. Вектор математичних сподівань E{xi} = (1 / N) * sum(X_l)
    mean_vec = np.sum(data, axis=0) / n

    # 2. Дисперсійно-коваріаційна матриця DC{xi} = (1 / (N - 1)) * sum((X_l - E)(X_l - E)^T)
    centered = data - mean_vec
    dc_mat = (centered.T @ centered) / (n - 1)

    # 3. Кореляційна матриця R{xi}: r_{i,j} = cov(xi_i, xi_j) / (sigma_i * sigma_j)
    sigmas = np.sqrt(np.diag(dc_mat))
    if np.any(sigmas <= 1e-12):
        raise ValueError("Одна з ознак має нульову дисперсію (всі значення однакові)!")

    outer_sigmas = np.outer(sigmas, sigmas)
    r_mat = dc_mat / outer_sigmas
    np.fill_diagonal(r_mat, 1.0)
    r_mat = np.clip(r_mat, -1.0, 1.0)

    return mean_vec, dc_mat, r_mat


def check_matrix_non_singular(r_mat: np.ndarray, tol: float = 1e-8):
    """
    Перевірка невиродженості кореляційної матриці R (вимога п. 6: |R| ≈ 0).
    """
    det_r = float(np.linalg.det(r_mat))
    if abs(det_r) < tol:
        raise ValueError(
            f"Кореляційна матриця R є виродженою (|R| = {det_r:.3e} ≈ 0)! "
            f"Ознаки є лінійно залежними, обчислення R^-1 неможливе."
        )
    return det_r


def compute_pairwise_significance(r_mat: np.ndarray, n_samples: int, names: list, alpha: float = 0.05):
    """
    Перевірка значущості парних коефіцієнтів кореляції r_{1,2}, r_{1,3}, r_{2,3} (п. 4.4.3)
    за t-критерієм Стьюдента з nu = N - 2 ступенями вільності.
    """
    nu = n_samples - 2
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=nu))
    pairs = [(0, 1), (0, 2), (1, 2)]
    results = []

    for i, j in pairs:
        r_val = float(r_mat[i, j])
        denom = np.sqrt(max(1e-12, 1.0 - r_val ** 2))
        t_stat = float(r_val * np.sqrt(nu) / denom)
        p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=nu)))
        results.append({
            "pair": f"{names[i]}-{names[j]}",
            "r_val": r_val,
            "t_stat": t_stat,
            "t_crit": t_crit,
            "p_val": p_val,
            "is_significant": abs(t_stat) > t_crit
        })
    return results


def compute_partial_correlations(r_mat: np.ndarray, n_clean: int, names: list = None, alpha: float = 0.05):
    """
    Обчислення часткових коефіцієнтів кореляції першого порядку (п. 4.5.1, п. 5.3):
    1) за прямою формулою r_{ij.d} = (r_{ij} - r_{id}*r_{jd}) / sqrt((1 - r_{id}^2)(1 - r_{jd}^2));
    2) перевірка через обернену матрицю P = R^-1: r_{ij.d} = -p_{ij} / sqrt(p_{ii} * p_{jj});
    3) перевірка значущості за t-критерієм (nu = N - 3).
    """
    if names is None:
        names = ["X1", "X2", "X3"]

    check_matrix_non_singular(r_mat)
    p_mat = np.linalg.inv(r_mat)

    nu = n_clean - 3
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=nu))

    triplets = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 2, 0)
    ]

    results = []
    for i, j, d in triplets:
        r_ij = float(r_mat[i, j])
        r_id = float(r_mat[i, d])
        r_jd = float(r_mat[j, d])

        # 1. Формула першого порядку (п. 5.3)
        num = r_ij - r_id * r_jd
        denom = np.sqrt(max(1e-12, (1.0 - r_id ** 2) * (1.0 - r_jd ** 2)))
        r_partial_formula = float(np.clip(num / denom, -0.999999, 0.999999))

        # 2. Перевірка через обернену матрицю P = R^-1
        r_partial_inv = float(-p_mat[i, j] / np.sqrt(p_mat[i, i] * p_mat[j, j]))

        # 3. Перевірка значущості (t-критерій, nu = N - 3)
        t_stat = float((r_partial_formula * np.sqrt(nu)) / np.sqrt(1.0 - r_partial_formula ** 2))
        p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=nu)))
        is_sig = abs(t_stat) > t_crit

        results.append({
            "pair": f"{names[i]}-{names[j]}",
            "excluded": names[d],
            "notation": f"r_{i+1},{j+1}·{d+1}",
            "label": f"{names[i]}-{names[j]} | {names[d]}",
            "r_pair": r_ij,
            "r_partial": r_partial_formula,
            "r_partial_inv": r_partial_inv,
            "diff_check": abs(r_partial_formula - r_partial_inv),
            "t_stat": t_stat,
            "t_crit": t_crit,
            "p_val": p_val,
            "is_significant": is_sig
        })

    return results


def compute_multiple_correlations(r_mat: np.ndarray, n_clean: int, names: list = None, alpha: float = 0.05):
    """
    Обчислення множинних коефіцієнтів кореляції та детермінації r^2 (п. 4.5.2, п. 5.4):
    1) за прямою формулою через парні кореляції:
       r_{x_k . x_i, x_j} = sqrt((r_{ki}^2 + r_{kj}^2 - 2*r_{ki}*r_{kj}*r_{ij}) / (1 - r_{ij}^2));
    2) за формулою через обернену матрицю P = R^-1:
       r_{x_k . x_i, x_j} = sqrt(1 - 1 / p_{k,k}) = sqrt(1 - |R| / R_{k,k});
    3) перевірка значущості за F-критерієм Фішера: f = ((N - 3) / 2) * (r^2 / (1 - r^2)), nu1 = 2, nu2 = N - 3.
    """
    if names is None:
        names = ["X1", "X2", "X3"]

    det_r = check_matrix_non_singular(r_mat)
    p_mat = np.linalg.inv(r_mat)

    nu1 = 2
    nu2 = n_clean - 3
    f_crit = float(stats.f.ppf(1.0 - alpha, dfn=nu1, dfd=nu2))

    triplets = [
        (0, 1, 2),
        (1, 0, 2),
        (2, 0, 1)
    ]

    results = []
    for k, i, j in triplets:
        r_ki = float(r_mat[k, i])
        r_kj = float(r_mat[k, j])
        r_ij = float(r_mat[i, j])

        # 1. Пряма формула з п. 5.4 (алгебраїчне доповнення R_{k,k} = 1 - r_{i,j}^2)
        r_kk_minor = 1.0 - r_ij ** 2
        num = (r_ki ** 2) + (r_kj ** 2) - 2.0 * r_ki * r_kj * r_ij
        r2_formula = float(max(0.0, num / r_kk_minor))
        r_mult_formula = float(np.sqrt(r2_formula))

        # 2. Перевірка через елемент оберненої матриці p_{k,k} та визначник |R| / R_{k,k}
        p_kk = float(p_mat[k, k])
        r_mult_inv = float(np.sqrt(max(0.0, 1.0 - 1.0 / p_kk)))
        r_mult_det = float(np.sqrt(max(0.0, 1.0 - det_r / r_kk_minor)))

        # F-критерій значущості (п. 5.4)
        if r2_formula < 1.0:
            f_stat = float((nu2 / nu1) * (r2_formula / (1.0 - r2_formula)))
        else:
            f_stat = float('inf')

        p_val = float(1.0 - stats.f.cdf(f_stat, dfn=nu1, dfd=nu2)) if f_stat < 1e8 else 0.0
        is_sig = f_stat > f_crit

        results.append({
            "target": names[k],
            "predictors": f"{names[i]}, {names[j]}",
            "notation": f"r_{names[k]}·{names[i]},{names[j]}",
            "r_mult": r_mult_formula,
            "r_mult_inv": r_mult_inv,
            "r_mult_det": r_mult_det,
            "r2": r2_formula,
            "f_stat": f_stat,
            "f_crit": f_crit,
            "p_val": p_val,
            "is_significant": is_sig
        })

    return results