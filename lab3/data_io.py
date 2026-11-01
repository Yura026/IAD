"""
data_io.py — Завантаження та збереження вибірок у форматах .csv, .txt, .xlsx.
"""

import os
import pandas as pd
import numpy as np


def load_2d_data(filepath: str):
    """
    Завантажує двовимірні дані з файлу .csv, .txt або .xlsx.
    Повертає два одновимірних масиви (x, y).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл {filepath} не знайдено!")

    ext = os.path.splitext(filepath)[1].lower()

    if ext in ('.xlsx', '.xls'):
        df = pd.read_excel(filepath)
    elif ext in ('.csv', '.txt'):
        try:
            df = pd.read_csv(filepath, sep=None, engine='python', decimal=',')
        except Exception:
            df = pd.read_csv(filepath, sep=None, engine='python', decimal='.')
    else:
        raise ValueError("Непідтримуваний формат файлу! Дозволені: .csv, .txt, .xlsx")

    # Беремо перші два числові стовпчики
    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.shape[1] < 2:
        # Якщо стовпці були прочитані як рядки (наприклад, через кому або пробіли)
        df_converted = df.iloc[:, :2].apply(
            lambda col: pd.to_numeric(col.astype(str).str.replace(',', '.'), errors='coerce')
        )
        df_numeric = df_converted.dropna()

    if df_numeric.shape[1] < 2:
        raise ValueError("Файл повинен містити щонайменше 2 числових стовпчики (X та Y)!")

    clean_data = df_numeric.iloc[:, [0, 1]].dropna().values
    if len(clean_data) < 10:
        raise ValueError("Вибірка містить менше 10 коректних числових рядків!")

    x = np.array(clean_data[:, 0], dtype=float)
    y = np.array(clean_data[:, 1], dtype=float)
    return x, y


def save_2d_data(filepath: str, x: np.ndarray, y: np.ndarray):
    """Зберігає масиви x та y у файл .csv, .txt або .xlsx."""
    df = pd.DataFrame({"x": x, "y": y})
    ext = os.path.splitext(filepath)[1].lower()

    if ext in ('.xlsx', '.xls'):
        df.to_excel(filepath, index=False)
    elif ext == '.txt':
        df.to_csv(filepath, sep='\t', index=False)
    else:
        df.to_csv(filepath, index=False)