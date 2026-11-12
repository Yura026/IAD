"""
pca.py — Метод головних компонент (МГК / PCA) згідно з п. 4.6 та п. 5.5 методички:
- стандартизація даних x*_{k,l} = (x_{k,l} - x_bar_k) / sigma_k;
- знаходження власних чисел lambda_1 >= lambda_2 >= lambda_3 та власних векторів A_{x'_k};
- перевірка умов: sum(lambda) = 3, prod(lambda) = |R|, діагоналізація A * R * A^T = Lambda;
- формування таблиці «Характеристики головних компонент»;
- пряме та зворотне перетворення з відкиданням компонент із часткою дисперсії < 5%;
- контрольна перевірка через sklearn.decomposition.PCA та numpy.corrcoef.
"""

import numpy as np


def standardize_matrix(data: np.ndarray):
    """
    Стандартизація даних (п. 5.5):
    x*_{k,l} = (x_{k,l} - x_bar_k) / sigma_k, де x*_bar_k = 0, sigma*_k = 1.
    """
    n = len(data)
    means = np.sum(data, axis=0) / n
    stds = np.sqrt(np.sum((data - means) ** 2, axis=0) / (n - 1))
    std_data = (data - means) / stds
    return std_data, means, stds


def fit_pca_custom(r_mat: np.ndarray):
    """
    Розв'язання власної задачі R{xi} * A_{x'_k} = lambda_k * A_{x'_k}.
    Повертає:
    - eigvals: вектор власних чисел lambda = (lambda_1, lambda_2, lambda_3), lambda_1 >= lambda_2 >= lambda_3;
    - matrix_a: ортогональна матриця A (рядки — власні вектори A_{x'_k}^T);
    - matrix_a_T: матриця власних векторів A^T = [A_{x'_1}, A_{x'_2}, A_{x'_3}] (стовпці — власні вектори);
    - checks: результати перевірки властивостей МГК.
    """
    raw_vals, raw_vecs = np.linalg.eigh(r_mat)

    # Впорядкування за спаданням власних чисел: lambda_1 >= lambda_2 >= lambda_3
    idx = np.argsort(raw_vals)[::-1]
    eigvals = raw_vals[idx]
    sorted_vecs = raw_vecs[:, idx]  # стовпці — власні вектори A_{x'_1}, A_{x'_2}, A_{x'_3}

    # Фіксація напрямку власних векторів (щоб найбільша за модулем компонента була додатною)
    for col in range(sorted_vecs.shape[1]):
        max_abs_idx = np.argmax(np.abs(sorted_vecs[:, col]))
        if sorted_vecs[max_abs_idx, col] < 0:
            sorted_vecs[:, col] *= -1.0

    # Матриця власних векторів A^T = [A_{x'_1}, A_{x'_2}, A_{x'_3}] (розмір 3x3, вектори у стовпцях)
    matrix_a_T = sorted_vecs.copy()
    # Ортогональна матриця A (рядки — власні вектори alpha_{k,1}, alpha_{k,2}, alpha_{k,3})
    matrix_a = matrix_a_T.T

    # Перевірка математичних властивостей (п. 4.6.2, п. 4.6.3, п. 5.5)
    sum_lambdas = float(np.sum(eigvals))
    prod_lambdas = float(np.prod(eigvals))
    det_r = float(np.linalg.det(r_mat))

    # Перевірка діагоналізації: A * R * A^T = Lambda = diag(lambda_1, lambda_2, lambda_3)
    lambda_mat = matrix_a @ r_mat @ matrix_a_T
    diag_diff = float(np.max(np.abs(lambda_mat - np.diag(eigvals))))

    # Перевірка нормування власних векторів: sum_{v=1}^n alpha_{k,v}^2 = 1
    norms_sq = np.sum(matrix_a ** 2, axis=1)

    checks = {
        "sum_lambdas": sum_lambdas,
        "is_sum_3": bool(np.isclose(sum_lambdas, 3.0, atol=1e-5)),
        "prod_lambdas": prod_lambdas,
        "det_r": det_r,
        "is_prod_det": bool(np.isclose(prod_lambdas, det_r, atol=1e-5)),
        "lambda_mat": lambda_mat,
        "diag_diff": diag_diff,
        "is_diagonalized": bool(diag_diff < 1e-5),
        "norms_sq": norms_sq
    }

    return eigvals, matrix_a, matrix_a_T, checks


def build_pca_summary_table(eigvals: np.ndarray, matrix_a: np.ndarray, names: list = None):
    """
    Формування таблиці «Характеристики головних компонент» точно за зразком п. 4.6.4:
    Стовпці: x'_1, x'_2, x'_3
    Рядки: x_1, x_2, x_3 (коефіцієнти alpha_{k,v}), lambda, %, Sigma%.
    """
    if names is None:
        names = ["x_1", "x_2", "x_3"]

    n_comp = len(eigvals)
    var_shares = (eigvals / n_comp) * 100.0
    cum_shares = np.cumsum(var_shares)

    table_data = {
        "headers": [f"x'_{k + 1}" for k in range(n_comp)],
        "rows": []
    }

    # Рядки x_1, x_2, x_3: у стовпці k стоїть коефіцієнт alpha_{k,v}
    for v in range(3):
        row_vals = [float(matrix_a[k, v]) for k in range(n_comp)]
        table_data["rows"].append({
            "name": f"x_{v + 1}",
            "values": row_vals
        })

    table_data["rows"].append({
        "name": "λ",
        "values": [float(val) for val in eigvals]
    })
    table_data["rows"].append({
        "name": "%",
        "values": [float(val) for val in var_shares]
    })
    table_data["rows"].append({
        "name": "Σ%",
        "values": [float(val) for val in cum_shares]
    })

    return table_data, var_shares, cum_shares


def transform_and_reconstruct(clean_data: np.ndarray, matrix_a: np.ndarray, var_shares: np.ndarray, threshold_pct: float = 5.0):
    """
    Пряме та зворотне перетворення МГК (п. 4.6.5 та п. 5.5):
    - пряме: x'_k = sum_{v=1}^n alpha_{k,v} * x*_v;
    - відкидання компонент, у яких частка дисперсії < 5% (вибір w);
    - зворотне: x*_k = sum_{v=1}^w alpha_{v,k} * x'_v, потім дестандартизація x_k = x*_k * sigma_k + x_bar_k.
    Якщо всі 3 компоненти мають частку >= 5%, додатково обчислюється реконструкція при w = 2
    для наочної демонстрації зниження розмірності (w < n).
    """
    std_data, means, stds = standardize_matrix(clean_data)

    # Пряме перетворення для всіх 3 компонент: Scores (N x 3)
    scores = std_data @ matrix_a.T

    # Вибір w за правилом частки дисперсії >= threshold_pct (5%)
    kept_indices = [k for k, share in enumerate(var_shares) if share >= threshold_pct]
    if not kept_indices:
        kept_indices = [0]
    w_rule = len(kept_indices)

    # Якщо за порогом 5% не відкинулася жодна компонента (w_rule == 3),
    # для виконання формули п. 5.5 (w < n, відкидання останньої найменшої ГК)
    # також рахуємо варіант з w = 2
    w_effective = w_rule if w_rule < 3 else 2

    # Зворотне перетворення за правилом порогу 5%
    std_rec_rule = scores[:, :w_rule] @ matrix_a[:w_rule, :]
    data_rec_rule = std_rec_rule * stds + means
    mse_rule = float(np.mean((clean_data - data_rec_rule) ** 2))
    mae_rule = float(np.mean(np.abs(clean_data - data_rec_rule)))

    # Зворотне перетворення при відкиданні 3-ї компоненти (w = 2)
    std_rec_w2 = scores[:, :2] @ matrix_a[:2, :]
    data_rec_w2 = std_rec_w2 * stds + means
    mse_w2 = float(np.mean((clean_data - data_rec_w2) ** 2))
    mae_w2 = float(np.mean(np.abs(clean_data - data_rec_w2)))

    return {
        "std_data": std_data,
        "scores": scores,
        "w_rule": w_rule,
        "w_effective": w_effective,
        "reconstructed_rule": data_rec_rule,
        "mse_rule": mse_rule,
        "mae_rule": mae_rule,
        "reconstructed_w2": data_rec_w2,
        "mse_w2": mse_w2,
        "mae_w2": mae_w2,
        "reconstructed_active": data_rec_rule if w_rule < 3 else data_rec_w2
    }


def compare_with_sklearn(clean_data: np.ndarray):
    """
    Контрольна перевірка результатів МГК через numpy.corrcoef та sklearn.decomposition.PCA (п. 6).
    Використовується та сама стандартизація з ddof=1 (п. 5.5), що забезпечує точний збіг власних чисел.
    """
    std_data, _, _ = standardize_matrix(clean_data)
    r_np = np.corrcoef(clean_data, rowvar=False)

    result = {
        "r_corrcoef": r_np,
        "sklearn_available": False
    }

    try:
        from sklearn.decomposition import PCA
        pca_sk = PCA(n_components=3)
        pca_sk.fit(std_data)

        result["sklearn_available"] = True
        result["lambdas"] = pca_sk.explained_variance_
        result["ratios"] = pca_sk.explained_variance_ratio_ * 100.0
        result["components"] = pca_sk.components_
    except Exception as e:
        result["error"] = str(e)

    return result