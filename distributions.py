"""
distributions.py — Математичні моделі розподілів, оцінювання параметрів,
зворотні функції F^-1(u) та розрахунок критерію узгодженості D (ЛР 2).
"""

import numpy as np
from scipy import stats

# ----------------- 1. НОРМАЛЬНИЙ РОЗПОДІЛ N(m, sigma) -----------------
def norm_pdf(x, m, sigma):
    """Щільність нормального розподілу f(x)."""
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - m) / sigma) ** 2)

def norm_cdf(x, m, sigma):
    """Функція нормального розподілу F(x)."""
    return stats.norm.cdf(x, loc=m, scale=sigma)

def norm_inv(u, m, sigma):
    """Зворотна функція нормального розподілу F^-1(u) = m + sigma * Phi^-1(u)."""
    return m + sigma * stats.norm.ppf(u)

def fit_normal(data):
    """Оцінювання параметрів (m, sigma) та їхніх СКВ."""
    x = np.array(data, dtype=float)
    N = len(x)
    m_hat = float(np.mean(x))
    s_hat = float(np.std(x, ddof=1))
    
    se_m = s_hat / np.sqrt(N)
    se_s = s_hat / np.sqrt(2.0 * N)
    return {"m": m_hat, "sigma": s_hat, "se_m": se_m, "se_sigma": se_s, "N": N}


# ----------------- 2. РІВНОМІРНИЙ РОЗПОДІЛ U(a, b) -----------------
def uniform_pdf(x, a, b):
    """Щільність рівномірного розподілу f(x)."""
    x = np.asarray(x, dtype=float)
    res = np.zeros_like(x)
    mask = (x >= a) & (x <= b)
    res[mask] = 1.0 / (b - a)
    return res

def uniform_cdf(x, a, b):
    """Функція рівномірного розподілу F(x)."""
    x = np.asarray(x, dtype=float)
    res = np.zeros_like(x)
    res[x > b] = 1.0
    mask = (x >= a) & (x <= b)
    res[mask] = (x[mask] - a) / (b - a)
    return res

def uniform_inv(u, a, b):
    """Зворотна функція рівномірного розподілу F^-1(u) = a + (b - a) * u."""
    return a + (b - a) * u

def fit_uniform(data):
    """Оцінювання параметрів (a, b) методом моментів та їхніх СКВ."""
    x = np.array(data, dtype=float)
    N = len(x)
    mean = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    
    a_hat = mean - np.sqrt(3.0) * s
    b_hat = mean + np.sqrt(3.0) * s
    se = (b_hat - a_hat) * np.sqrt(2.0 / (15.0 * N))
    return {"a": a_hat, "b": b_hat, "se_a": se, "se_b": se, "N": N}


# ----------------- 3. ЕКСПОНЕНЦІАЛЬНИЙ РОЗПОДІЛ Exp(lambda) -----------------
def exp_pdf(x, lam):
    """Щільність експоненціального розподілу f(x) = lam * exp(-lam * x)."""
    x = np.asarray(x, dtype=float)
    res = np.zeros_like(x)
    mask = x >= 0
    res[mask] = lam * np.exp(-lam * x[mask])
    return res

def exp_cdf(x, lam):
    """Функція експоненціального розподілу F(x) = 1 - exp(-lam * x)."""
    x = np.asarray(x, dtype=float)
    res = np.zeros_like(x)
    mask = x >= 0
    res[mask] = 1.0 - np.exp(-lam * x[mask])
    return res

def exp_inv(u, lam):
    """Зворотна функція F^-1(u) = -ln(1 - u) / lam."""
    # Беремо 1 - u для уникнення ln(0)
    return -np.log(1.0 - u) / lam

def fit_exponential(data):
    """Оцінювання параметра lambda та його СКВ."""
    x = np.array(data, dtype=float)
    N = len(x)
    mean = float(np.mean(x))
    if mean <= 0:
        raise ValueError("Експоненціальний розподіл вимагає додатних значень.")
    lam_hat = 1.0 / mean
    se_lam = lam_hat / np.sqrt(N)
    return {"lambda": lam_hat, "se_lambda": se_lam, "N": N}


# ----------------- 4. РОЗРАХУНОК ВІДХИЛЕННЯ D (КОЛМОГОРОВ) -----------------
def compute_ks_distance(data, cdf_func, *params):
    """
    Власний розрахунок максимального абсолютного відхилення:
    D = max |Fn(x) - F(x)|.
    """
    x_sorted = np.sort(np.asarray(data, dtype=float))
    N = len(x_sorted)
    
    # Теоретичні значення F(x_i)
    f_theo = cdf_func(x_sorted, *params)
    
    # Емпірична функція робить стрибок від (i-1)/N до i/N
    i_over_n = np.arange(1, N + 1) / N
    i_minus_one_over_n = np.arange(0, N) / N
    
    d_plus = np.max(i_over_n - f_theo)
    d_minus = np.max(f_theo - i_minus_one_over_n)
    
    return float(max(d_plus, d_minus))