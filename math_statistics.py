import numpy as np
from scipy import stats

def compute_all_statistics(data, gamma=0.95):
    """
    Обчислення точкових та інтервальних оцінок за формулами з Лекції 1 та Лекції 2.
    Включає перевірку гіпотез щодо симетрії та ексцесу (Пункт 4.3.4 методички).
    """
    x = np.array(data, dtype=float)
    N = len(x)
    if N < 4:
        raise ValueError("Вибірка занадто мала (потрібно N >= 4).")

    # 1. Точкові оцінки (Лекція 1)
    x_bar = float(np.mean(x))
    median = float(np.median(x))
    
    # Незсунена дисперсія S^2 та стандартне відхилення S (Лекція 1, слайд 20)
    s2 = float(np.sum((x - x_bar) ** 2) / (N - 1))
    s = float(np.sqrt(s2))
    
    # Зсунені центральні моменти mu_k (Лекція 1, слайди 17, 26)
    mu_2 = float(np.mean((x - x_bar) ** 2))
    hat_sigma = float(np.sqrt(mu_2))
    mu_3 = float(np.mean((x - x_bar) ** 3))
    mu_4 = float(np.mean((x - x_bar) ** 4))
    
    # Коефіцієнти асиметрії та ексцесу
    a = float(mu_3 / (hat_sigma ** 3)) if hat_sigma > 0 else 0.0
    e = float((mu_4 / (hat_sigma ** 4)) - 3.0) if hat_sigma > 0 else 0.0

    # 2. Інтервальні оцінки (Лекція 2, слайди 8, 12-14)
    alpha = 1.0 - gamma
    # Квантиль t-розподілу Стьюдента для v = N - 1
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=N - 1))
    
    # Середньоквадратичні відхилення самих оцінок (Лекція 2, слайд 8)
    sigma_xbar = float(s / np.sqrt(N))
    sigma_s2 = float(np.sqrt(2.0 / (N - 1)) * s2)
    sigma_a = float(np.sqrt(6.0 * (N - 2) / ((N + 1) * (N + 3))))
    sigma_e = float(np.sqrt(24.0 * N * ((N - 1) ** 2) / ((N - 3) * (N - 2) * (N + 3) * (N + 5))))
    
    # Межі довірчих інтервалів
    a_lower = a - t_crit * sigma_a
    a_upper = a + t_crit * sigma_a
    e_lower = e - t_crit * sigma_e
    e_upper = e + t_crit * sigma_e

    # Перевірка накриття 0 довірчими інтервалами (Пункт 4.3.4 методички)
    a_covers_zero = (a_lower <= 0.0 <= a_upper)
    e_covers_zero = (e_lower <= 0.0 <= e_upper)

    if a_covers_zero:
        a_conclusion = "ДІ містить 0 → розподіл симетричний"
    elif a > 0:
        a_conclusion = "ДІ не містить 0 (A > 0) → правостороння асиметрія"
    else:
        a_conclusion = "ДІ не містить 0 (A < 0) → лівостороння асиметрія"

    if e_covers_zero:
        e_conclusion = "ДІ містить 0 → гостроверхість відповідає нормі"
    elif e > 0:
        e_conclusion = "ДІ не містить 0 (E > 0) → гостровершинний розподіл"
    else:
        e_conclusion = "ДІ не містить 0 (E < 0) → плосковершинний розподіл"

    ci = {
        "x_bar": {
            "name": "Середнє (x̄)",
            "theta_l": x_bar - t_crit * sigma_xbar,
            "theta": x_bar,
            "theta_u": x_bar + t_crit * sigma_xbar,
            "sigma": sigma_xbar
        },
        "S2": {
            "name": "Дисперсія (S²)",
            "theta_l": max(0.0, s2 - t_crit * sigma_s2),
            "theta": s2,
            "theta_u": s2 + t_crit * sigma_s2,
            "sigma": sigma_s2
        },
        "A": {
            "name": "Асиметрія (A)",
            "theta_l": a_lower,
            "theta": a,
            "theta_u": a_upper,
            "sigma": sigma_a,
            "covers_zero": a_covers_zero,
            "conclusion": a_conclusion
        },
        "E": {
            "name": "Ексцес (E)",
            "theta_l": e_lower,
            "theta": e,
            "theta_u": e_upper,
            "sigma": sigma_e,
            "covers_zero": e_covers_zero,
            "conclusion": e_conclusion
        }
    }
    
    point_estimates = {
        "N": N,
        "mean": x_bar,
        "median": median,
        "variance": s2,
        "std": s,
        "skewness": a,
        "kurtosis": e,
        "t_crit": t_crit
    }
    
    return point_estimates, ci