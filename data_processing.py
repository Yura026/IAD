import pandas as pd
import numpy as np

def get_excel_sheet_names(filepath):
    """Отримання списку назв усіх аркушів Excel-файлу."""
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        try:
            xl = pd.ExcelFile(filepath)
            return xl.sheet_names
        except Exception:
            return []
    return []

def load_data(filepath, sheet_name=0):
    """
    Завантаження даних із Excel, CSV або TXT (з підтримкою коми або крапки).
    """
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        # Якщо вказано число як рядок, перетворюємо на int для індексу аркуша
        if isinstance(sheet_name, str) and sheet_name.isdigit():
            sheet_name = int(sheet_name)
        df = pd.read_excel(filepath, sheet_name=sheet_name)
    else:
        try:
            df = pd.read_csv(filepath, sep=None, engine='python', decimal=',')
        except Exception:
            df = pd.read_csv(filepath, sep=None, engine='python', decimal='.')
            
    # Приводимо перші два стовпчики до стандарту
    df = df.iloc[:, [0, 1]].copy()
    df.columns = ['time', 'value']
    # Примусове перетворення значень на числовий тип
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df

def get_period(index):
    """
    Визначення періоду доби за індексом інтервалу (по 30 хв):
    0-11: Ніч (00:00-05:30)
    12-23: Ранок (06:00-11:30)
    24-35: День (12:00-17:30)
    36-47: Вечір (18:00-23:30)
    """
    idx_mod = index % 48
    if 0 <= idx_mod <= 11:
        return 'Ніч'
    elif 12 <= idx_mod <= 23:
        return 'Ранок'
    elif 24 <= idx_mod <= 35:
        return 'День'
    else:
        return 'Вечір'

def detect_defects(df, lower_limit=4.0, upper_limit=55.0):
    """
    Пошук пропусків (NaN) та значень за межами нормального діапазону.
    """
    missing_indices = df[df['value'].isna()].index.tolist()
    anomaly_indices = df[(df['value'] < lower_limit) | (df['value'] > upper_limit)].index.tolist()
    return missing_indices, anomaly_indices

def compute_period_means(df, lower_limit=4.0, upper_limit=55.0, real_events_indices=None):
    """
    Обчислення середніх за періодами доби тільки за коректними даними.
    """
    if real_events_indices is None:
        real_events_indices = []
    
    df_temp = df.copy()
    df_temp['period'] = [get_period(i) for i in df_temp.index]
    
    is_missing = df_temp['value'].isna()
    is_anomaly = (df_temp['value'] < lower_limit) | (df_temp['value'] > upper_limit)
    is_technical = is_anomaly & (~df_temp.index.isin(real_events_indices))
    valid_mask = (~is_missing) & (~is_technical)
    
    period_means = {}
    for p in ['Ніч', 'Ранок', 'День', 'Вечір']:
        subset = df_temp[valid_mask & (df_temp['period'] == p)]['value']
        period_means[p] = float(subset.mean()) if len(subset) > 0 else 0.0
    return period_means

def clean_data_by_period_mean(df, lower_limit=4.0, upper_limit=55.0, real_events_indices=None):
    """
    Заміна пропусків та технічних аномалій середнім значенням відповідного періоду доби.
    Реальні події залишаються без змін!
    """
    if real_events_indices is None:
        real_events_indices = []

    cleaned_df = df.copy()
    cleaned_df['period'] = [get_period(i) for i in cleaned_df.index]
    period_means = compute_period_means(cleaned_df, lower_limit, upper_limit, real_events_indices)
    
    is_missing = cleaned_df['value'].isna()
    is_anomaly = (cleaned_df['value'] < lower_limit) | (cleaned_df['value'] > upper_limit)
    is_technical = is_anomaly & (~cleaned_df.index.isin(real_events_indices))
    
    for idx in cleaned_df.index:
        if is_missing[idx] or is_technical[idx]:
            p = cleaned_df.loc[idx, 'period']
            cleaned_df.loc[idx, 'value'] = period_means[p]
            
    return cleaned_df, period_means

def clean_data_by_global_mean(df, lower_limit=4.0, upper_limit=55.0, real_events_indices=None):
    """Заміна загальним середнім (реальні події зберігаються)."""
    if real_events_indices is None:
        real_events_indices = []
    cleaned_df = df.copy()
    is_missing = cleaned_df['value'].isna()
    is_anomaly = (cleaned_df['value'] < lower_limit) | (cleaned_df['value'] > upper_limit)
    is_technical = is_anomaly & (~cleaned_df.index.isin(real_events_indices))
    
    global_mean = float(cleaned_df[(~is_missing) & (~is_technical)]['value'].mean())
    for idx in cleaned_df.index:
        if is_missing[idx] or is_technical[idx]:
            cleaned_df.loc[idx, 'value'] = global_mean
    return cleaned_df

def clean_data_by_drop(df, lower_limit=4.0, upper_limit=55.0, real_events_indices=None):
    """Вилучення некоректних рядків (реальні події зберігаються)."""
    if real_events_indices is None:
        real_events_indices = []
    cleaned_df = df.copy()
    is_missing = cleaned_df['value'].isna()
    is_anomaly = (cleaned_df['value'] < lower_limit) | (cleaned_df['value'] > upper_limit)
    is_technical = is_anomaly & (~cleaned_df.index.isin(real_events_indices))
    # Зберігаємо оригінальні індекси рядків
    return cleaned_df[(~is_missing) & (~is_technical)].copy()

def build_variational_series(series):
    """
    Побудова варіаційного ряду (Лекція 1, слайди 4-5):
    Ранжовані варіанти x_i, їх абсолютні частоти n_i та відносні частоти p_i.
    """
    vals = pd.Series(series).dropna().sort_values().values
    N = len(vals)
    if N == 0:
        return pd.DataFrame(columns=['x_i', 'n_i', 'p_i'])
    
    unique_vals, counts = np.unique(vals, return_counts=True)
    rel_freqs = counts / N
    var_df = pd.DataFrame({
        'x_i': unique_vals,
        'n_i': counts,
        'p_i': rel_freqs
    })
    return var_df

# def apply_log_transform(series):
#     """Логарифмування y = ln(x) із коректним зсувом (Лекція 1, слайд 22)."""
#     s = pd.Series(series, dtype=float).copy()
#     valid_vals = s.dropna().values
#     if len(valid_vals) == 0:
#         return s, 0.0

#     min_val = np.min(valid_vals)
#     shift = 0.0
#     if min_val <= 0:
#         shift = abs(min_val) + 1.0
#         s = s + shift
    
#     result = np.log(s)
#     return result, shift
def apply_log_transform(series, shift=None):
    """
    Логарифмічне перетворення даних y = ln(x + shift) (Лекція 1, слайд 22).
    
    Параметри:
        series: вхідні дані (pd.Series, np.ndarray або список)
        shift: фіксований зсув (якщо None — обчислюється автоматично за вибіркою)
        
    Повертає:
        result (pd.Series): перетворені логарифмовані дані
        shift (float): застосоване значення зсуву
    """
    orig_index = getattr(series, 'index', None)
    s = pd.Series(series, index=orig_index, dtype=float).copy()
    
    # Фільтруємо нескінченності та пропуски для безпечного пошуку мінімуму
    valid = s.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return s, 0.0

    # Якщо зсув не задано ззовні (наприклад, з train-вибірки) — обчислюємо адаптивно
    if shift is None:
        min_val = valid.min()
        # За формулою лекції: якщо є x <= 0, зсуваємо на |min| + 1, щоб min став 1 (ln(1) = 0)
        shift = float(abs(min_val) + 1.0) if min_val <= 0 else 0.0
    
    # Застосовуємо зсув та натуральний логарифм
    result = np.log(s + shift)
    return result, shift

def apply_standardization(series):
    """Класична стандартизація Z = (x - mean) / std (Лекція 2, слайд 3)."""
    x = np.array(series, dtype=float)
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    if std == 0:
        return x - mean
    return (x - mean) / std

def apply_shift(series, c):
    """Зсув y = x + c."""
    return np.array(series, dtype=float) + c