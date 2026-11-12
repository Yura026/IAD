"""
data_io.py — Завантаження, валідація та експорт багатовимірних вибірок (ЛР 5):
- строга перевірка формату: рівно 3 стовпці, однакова довжина рядків, числові значення;
- підтримка розділювачів: пробіл або табуляція;
- підтримка крапки та коми як десяткового розділювача;
- збереження значень головних компонент (pc_scores_NN.txt) та відновлених даних.
"""

import os
import re
import numpy as np


def load_multivariate_data(filepath: str):
    """
    Завантажує тривимірну вибірку (N x 3) з текстового файлу .txt / .tsv / .csv.
    Генерує зрозумілі повідомлення про помилки для користувача згідно з п. 6 методички:
    - нечислові значення;
    - рядки різної довжини;
    - кількість стовпців, відмінна від 3.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл '{filepath}' не знайдено!")

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        raw_lines = [(idx + 1, line.strip()) for idx, line in enumerate(f)]

    non_empty_lines = [(line_num, line) for line_num, line in raw_lines if line]
    if not non_empty_lines:
        raise ValueError("Обраний файл порожній!")

    # Перевіряємо перший рядок на наявність заголовків (літер)
    first_line_num, first_line = non_empty_lines[0]
    first_tokens = [t for t in re.split(r'[\s;]+', first_line) if t]
    has_header = any(re.search(r'[a-zA-Zа-яА-ЯіїєґІЇЄҐ_]', tok) for tok in first_tokens)

    if has_header:
        if len(first_tokens) != 3:
            raise ValueError(
                f"Помилка структури заголовка (рядок {first_line_num}): "
                f"очікується рівно 3 стовпці, але знайдено {len(first_tokens)} ({first_tokens})."
            )
        feature_names = first_tokens
        data_lines = non_empty_lines[1:]
    else:
        feature_names = ["X1", "X2", "X3"]
        data_lines = non_empty_lines

    if not data_lines:
        raise ValueError("Файл містить лише заголовок і не містить числових спостережень!")

    parsed_rows = []
    expected_cols = len(re.split(r'[\s;]+', data_lines[0][1]))

    for line_num, line in data_lines:
        tokens = [t for t in re.split(r'[\s;]+', line) if t]

        # 1. Перевірка на рядки різної довжини
        if len(tokens) != expected_cols:
            raise ValueError(
                f"Виявлено рядки різної довжини! У рядку {line_num} знайдено "
                f"{len(tokens)} елементів замість {expected_cols}:\n'{line}'"
            )

        # 2. Перевірка, що кількість стовпців дорівнює 3
        if len(tokens) != 3:
            raise ValueError(
                f"Некоректна кількість стовпців у рядку {line_num}: "
                f"потрібно рівно 3 ознаки, а знайдено {len(tokens)}."
            )

        # 3. Перевірка на нечислові значення
        row_floats = []
        for col_idx, tok in enumerate(tokens, start=1):
            clean_tok = tok.replace(',', '.')
            try:
                val = float(clean_tok)
                if not np.isfinite(val):
                    raise ValueError()
                row_floats.append(val)
            except ValueError:
                raise ValueError(
                    f"Нечислове або некоректне значення '{tok}' "
                    f"у рядку {line_num}, стовпець {col_idx} ({feature_names[col_idx - 1]})!"
                )

        parsed_rows.append(row_floats)

    data = np.array(parsed_rows, dtype=float)
    if len(data) < 10:
        raise ValueError(
            f"Вибірка занадто мала (N = {len(data)}). "
            f"Для статистичного аналізу потрібно щонайменше 10 спостережень!"
        )

    return data, feature_names


def save_pc_scores(filepath: str, scores: np.ndarray, feature_names=None):
    """
    Збереження значень головних компонент у файл pc_scores_NN.txt.
    """
    if feature_names is None:
        feature_names = [f"x'_{i + 1}" for i in range(scores.shape[1])]

    header = "\t".join(feature_names)
    np.savetxt(filepath, scores, fmt="%.6f", delimiter="\t", header=header, comments="")


def save_reconstructed_data(filepath: str, reconstructed: np.ndarray, feature_names=None):
    """
    Збереження відфільтрованих (відновлених після МГК) даних у файл .txt.
    """
    if feature_names is None:
        feature_names = ["X1_hat", "X2_hat", "X3_hat"]

    header = "\t".join(feature_names)
    np.savetxt(filepath, reconstructed, fmt="%.6f", delimiter="\t", header=header, comments="")


def generate_sample_var01(n_samples: int = 200, seed: int = 100):
    """
    Генерація демонстраційної тривимірної вибірки за зразком Варіанта 1 (N = 200)
    з першими двома рядками з методички та присутніми аномаліями 3-сигма.
    Використовується, якщо файл var_01.txt ще не покладено в директорію програми.
    """
    np.random.seed(seed)
    mean = [25.0, 75.0, 70.0]
    cov = [
        [16.0,  18.0, -12.0],
        [18.0,  36.0, -20.0],
        [-12.0, -20.0, 25.0]
    ]
    data = np.random.multivariate_normal(mean, cov, size=n_samples)

    # Перші два рядки точно з фрагмента var_01.txt у методичці (п. 3)
    data[0] = [18.313, 70.271, 77.328]
    data[1] = [26.090, 80.238, 68.627]

    # Додавання явних аномалій за межами 3-сигма
    data[12, 0] += 5.5 * np.sqrt(cov[0][0])
    data[45, 1] -= 6.0 * np.sqrt(cov[1][1])
    data[110, 2] += 5.0 * np.sqrt(cov[2][2])

    names = ["X1", "X2", "X3"]
    return data, names