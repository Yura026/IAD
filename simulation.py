"""
simulation.py — Імітаційне моделювання вибірок методом зворотної функції,
перевірка параметрів T-тестом (як для однієї вибірки, так і для R повторень)
та оцінювання точності від N (ЛР 2).
Викликає функції оцінювання з distributions.py та math_statistics.py без дублювання.
"""

import numpy as np
from scipy import stats
from distributions import (
    norm_inv, uniform_inv, exp_inv,
    fit_normal, fit_uniform, fit_exponential
)
from math_statistics import compute_all_statistics


def generate_sample(dist_name, params, size, seed=None):
    """
    Генерація вибірки заданого розміру методом зворотної функції F^-1(u).
    Використовується виключно базовий генератор np.random.uniform(0, 1).
    """
    if seed is not None:
        np.random.seed(seed)

    u = np.random.uniform(0.0, 1.0, size=size)

    if dist_name == "Нормальний":
        m = params["m"]
        sigma = params["sigma"]
        return norm_inv(u, m, sigma)
    elif dist_name == "Рівномірний":
        a = params["a"]
        b = params["b"]
        return uniform_inv(u, a, b)
    elif dist_name == "Експоненціальний":
        lam = params["lambda"]
        return exp_inv(u, lam)
    else:
        raise ValueError(f"Невідомий закон розподілу: {dist_name}")


def run_single_experiment(dist_name, true_params, N, alpha=0.05, seed=None):
    """
    Один експеримент: моделювання однієї вибірки обсягом N,
    оцінювання параметрів та розрахунок T-статистик:
    t = (theta_0 - theta_hat) / sigma{theta_hat}.
    """
    sample = generate_sample(dist_name, true_params, size=N, seed=seed)
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=N - 1))
    results = {}

    if dist_name == "Нормальний":
        fit = fit_normal(sample)
        t_m = (true_params["m"] - fit["m"]) / fit["se_m"]
        rej_m = abs(t_m) > t_crit
        t_s = (true_params["sigma"] - fit["sigma"]) / fit["se_sigma"]
        rej_s = abs(t_s) > t_crit
        results["m"] = {
            "theta_0": true_params["m"],
            "est": fit["m"],
            "se_est": fit["se_m"],
            "t": t_m,
            "t_crit": t_crit,
            "rejected": rej_m
        }
        results["sigma"] = {
            "theta_0": true_params["sigma"],
            "est": fit["sigma"],
            "se_est": fit["se_sigma"],
            "t": t_s,
            "t_crit": t_crit,
            "rejected": rej_s
        }

    elif dist_name == "Рівномірний":
        fit = fit_uniform(sample)
        t_a = (true_params["a"] - fit["a"]) / fit["se_a"]
        rej_a = abs(t_a) > t_crit
        t_b = (true_params["b"] - fit["b"]) / fit["se_b"]
        rej_b = abs(t_b) > t_crit
        results["a"] = {
            "theta_0": true_params["a"],
            "est": fit["a"],
            "se_est": fit["se_a"],
            "t": t_a,
            "t_crit": t_crit,
            "rejected": rej_a
        }
        results["b"] = {
            "theta_0": true_params["b"],
            "est": fit["b"],
            "se_est": fit["se_b"],
            "t": t_b,
            "t_crit": t_crit,
            "rejected": rej_b
        }

    elif dist_name == "Експоненціальний":
        fit = fit_exponential(sample)
        t_lam = (true_params["lambda"] - fit["lambda"]) / fit["se_lambda"]
        rej_lam = abs(t_lam) > t_crit
        results["lambda"] = {
            "theta_0": true_params["lambda"],
            "est": fit["lambda"],
            "se_est": fit["se_lambda"],
            "t": t_lam,
            "t_crit": t_crit,
            "rejected": rej_lam
        }

    return results, sample


def compare_single_sample_with_theory(dist_name, params, size=48, alpha=0.05, seed=42):
    """
    Пункт 4.3.2 + одиночний Т-тест:
    Моделювання однієї вибірки обсягом N, порівняння її вибіркових характеристик
    (x_bar, S^2, A, E) із теоретичними значеннями та виконання Т-тесту для цієї вибірки.
    Викликає compute_all_statistics із math_statistics.py та run_single_experiment.
    """
    t_test_res, sample = run_single_experiment(dist_name, params, N=size, alpha=alpha, seed=seed)
    pts, ci = compute_all_statistics(sample, gamma=1.0 - alpha)
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=size - 1))

    if dist_name == "Нормальний":
        m, s = params["m"], params["sigma"]
        theo = {"mean": m, "variance": s ** 2, "std": s, "skewness": 0.0, "kurtosis": 0.0}
    elif dist_name == "Рівномірний":
        a, b = params["a"], params["b"]
        var_u = ((b - a) ** 2) / 12.0
        theo = {"mean": 0.5 * (a + b), "variance": var_u, "std": np.sqrt(var_u), "skewness": 0.0, "kurtosis": -1.2}
    elif dist_name == "Експоненціальний":
        lam = params["lambda"]
        theo = {"mean": 1.0 / lam, "variance": 1.0 / (lam ** 2), "std": 1.0 / lam, "skewness": 2.0, "kurtosis": 6.0}
    else:
        raise ValueError(f"Невідомий закон розподілу: {dist_name}")

    comparison = {
        "N": size,
        "nu": size - 1,
        "alpha": alpha,
        "seed": seed,
        "t_crit": t_crit,
        "sample": sample,
        "sample_stats": pts,
        "sample_ci": ci,
        "theory_stats": theo,
        "t_test": t_test_res
    }
    return comparison


def run_full_simulation(dist_name, true_params, N_list, R=200, alpha=0.05, master_seed=42, progress_callback=None):
    """
    Повна серія експериментів з R повтореннями для списку обсягів N_list.
    Повертає підсумкові зведені таблиці T-тесту та сирі значення t для гістограм.
    Підтримує progress_callback(current_step, total_steps) для індикації прогресу в GUI.
    """
    np.random.seed(master_seed)
    summary_tables = {}  # param -> list of row dicts
    raw_t_values = {}    # (param, N) -> list of t

    param_keys = list(true_params.keys())
    for p in param_keys:
        summary_tables[p] = []

    total_steps = len(N_list) * R
    current_step = 0

    for N in N_list:
        nu = N - 1
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=nu))

        sim_data = {p: {"ests": [], "se_ests": [], "t_vals": [], "rejections": 0} for p in param_keys}

        for r_idx in range(R):
            res, _ = run_single_experiment(dist_name, true_params, N, alpha)
            for p in param_keys:
                sim_data[p]["ests"].append(res[p]["est"])
                sim_data[p]["se_ests"].append(res[p]["se_est"])
                sim_data[p]["t_vals"].append(res[p]["t"])
                if res[p]["rejected"]:
                    sim_data[p]["rejections"] += 1

            current_step += 1
            if progress_callback is not None and (r_idx % 20 == 0 or current_step == total_steps):
                progress_callback(current_step, total_steps)

        for p in param_keys:
            raw_t_values[(p, N)] = sim_data[p]["t_vals"]
            theta_0 = true_params[p]
            mean_est = float(np.mean(sim_data[p]["ests"]))
            std_est = float(np.std(sim_data[p]["ests"], ddof=1))
            mean_se = float(np.mean(sim_data[p]["se_ests"]))
            mean_t = float(np.mean(sim_data[p]["t_vals"]))
            std_t = float(np.std(sim_data[p]["t_vals"], ddof=1))
            rej_percent = (sim_data[p]["rejections"] / R) * 100.0

            delta = (t_crit * mean_se) / abs(theta_0) if theta_0 != 0 else 0.0

            row = {
                "N": N,
                "nu": nu,
                "t_crit": t_crit,
                "mean_est": mean_est,
                "std_est": std_est,
                "theory_se": mean_se,
                "mean_t": mean_t,
                "std_t": std_t,
                "rej_rate": rej_percent,
                "delta": delta
            }
            summary_tables[p].append(row)

    return summary_tables, raw_t_values


def compute_required_sample_sizes(dist_name, true_params, alpha=0.05, delta_targets=(0.15, 0.10, 0.05, 0.03, 0.02, 0.01)):
    """
    Розрахунок таблиці необхідного обсягу вибірки N* (п. 1 та п. 7.7 методички)
    для заданих порогів відносної похибки delta = (t_{1-alpha/2, N-1} * sigma{theta}) / |theta_0|.
    """
    results = []

    def theoretical_se(param_name, n):
        if dist_name == "Нормальний":
            s = true_params["sigma"]
            return s / np.sqrt(n) if param_name == "m" else s / np.sqrt(2.0 * n)
        elif dist_name == "Рівномірний":
            span = true_params["b"] - true_params["a"]
            return span * np.sqrt(2.0 / (15.0 * n))
        elif dist_name == "Експоненціальний":
            lam = true_params["lambda"]
            return lam / np.sqrt(n)
        return 0.0

    param_keys = list(true_params.keys())
    for d_target in delta_targets:
        row = {"delta_pct": d_target * 100.0}
        for p in param_keys:
            theta_0 = abs(true_params[p])
            if theta_0 == 0:
                row[p] = None
                continue
            z_approx = stats.norm.ppf(1.0 - alpha / 2.0)
            se_at_1 = theoretical_se(p, 1.0)
            n_est = max(4, int(np.floor(((z_approx * se_at_1) / (d_target * theta_0)) ** 2)) - 5)
            n_found = None
            for n in range(max(4, n_est), n_est + 50000):
                t_c = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
                d_val = (t_c * theoretical_se(p, n)) / theta_0
                if d_val <= d_target:
                    n_found = n
                    break
            row[p] = n_found
        results.append(row)

    return results