"""
gui.py — Графічний інтерфейс користувача (ЛР 1 + ЛР 2).
Включає первинну обробку, оцінювання характеристик, порівняння трьох законів
розподілу за критерієм D (з окремими стовпцями θн, θ̂, θв, σ{θ̂} та V),
імітаційне моделювання методом зворотної функції, окрему кнопку для моделювання
1 вибірки та перевірки її T-тестом (з висновком H0/H1 та порівнянням x̄, S², A, E),
серію T-тесту на R повторень, гістограми t для N=50 і N=1000 (п. 4.4.3),
таблицю необхідного обсягу вибірки N* (п. 7.7), індикатор прогресу та експорт результатів.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Модулі з ЛР 1
from data_processing import (
    load_data, detect_defects, get_excel_sheet_names,
    clean_data_by_period_mean, clean_data_by_global_mean, clean_data_by_drop,
    compute_period_means, build_variational_series,
    apply_log_transform, apply_standardization, apply_shift
)
from math_statistics import compute_all_statistics
from visualization import plot_time_series, plot_histogram, plot_distribution_functions

# Нові модулі для ЛР 2
from distributions import (
    norm_pdf, norm_cdf, uniform_pdf, uniform_cdf, exp_pdf, exp_cdf,
    fit_normal, fit_uniform, fit_exponential, compute_ks_distance
)
from simulation import (
    generate_sample, compare_single_sample_with_theory,
    run_full_simulation, compute_required_sample_sizes
)


class DataMiningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ІАД — Комплексний аналіз та імітаційне моделювання (ЛР 1 + ЛР 2)")
        self.root.geometry("1480x950")

        # Дані вибірки
        self.filepath = None
        self.raw_df = None
        self.working_df = None
        self.missing_indices = []
        self.anomaly_indices = []

        # Дані моделювання (ЛР 2)
        self.fitted_models = {}
        self.best_law_name = "Нормальний"
        self.sim_summary_tables = None
        self.sim_raw_t = None
        self.current_sim_params = None
        self.single_sample_comp = None
        self.req_sample_sizes = None
        self.single_click_counter = 0

        self.setup_ui()

    def setup_ui(self):
        # 1. Верхня панель: вибір файлу та параметри очищення
        top_frame = ttk.LabelFrame(self.root, text=" 1. Вхідні дані та фільтрація ")
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)

        # 2. Панель дій: методи очищення та перетворення
        action_frame = ttk.LabelFrame(self.root, text=" 2. Обробка та перетворення (ЛР 1) ")
        action_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)

        # Головне розділене вікно
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=3)

        left_frame = ttk.Frame(main_paned, width=410)
        right_frame = ttk.Frame(main_paned, width=1050)
        main_paned.add(left_frame, weight=1)
        main_paned.add(right_frame, weight=3)

        # --- Наповнення верхньої панелі ---
        ttk.Button(top_frame, text="Обрати файл...", command=self.browse_file).grid(row=0, column=0, padx=4, pady=2)
        self.file_label = ttk.Label(top_frame, text="Файл не вибрано", foreground="gray")
        self.file_label.grid(row=0, column=1, columnspan=2, sticky=tk.W, padx=4)

        ttk.Label(top_frame, text="Аркуш:").grid(row=0, column=3, padx=4)
        self.sheet_combo = ttk.Combobox(top_frame, width=16)
        self.sheet_combo.grid(row=0, column=4, padx=4)

        ttk.Button(top_frame, text="Завантажити", command=self.load_dataset).grid(row=0, column=5, padx=4)

        ttk.Label(top_frame, text="Min:").grid(row=1, column=0, padx=3, pady=2, sticky=tk.E)
        self.min_entry = ttk.Entry(top_frame, width=6)
        self.min_entry.insert(0, "29")
        self.min_entry.grid(row=1, column=1, sticky=tk.W, padx=2)

        ttk.Label(top_frame, text="Max:").grid(row=1, column=2, padx=3, sticky=tk.E)
        self.max_entry = ttk.Entry(top_frame, width=6)
        self.max_entry.insert(0, "580")
        self.max_entry.grid(row=1, column=3, sticky=tk.W, padx=2)

        ttk.Label(top_frame, text="Рівень γ:").grid(row=1, column=4, padx=3, sticky=tk.E)
        self.gamma_entry = ttk.Entry(top_frame, width=5)
        self.gamma_entry.insert(0, "0.95")
        self.gamma_entry.grid(row=1, column=5, sticky=tk.W, padx=2)

        ttk.Label(top_frame, text="Реальні події:").grid(row=1, column=6, padx=4, sticky=tk.E)
        self.real_events_entry = ttk.Entry(top_frame, width=12)
        self.real_events_entry.grid(row=1, column=7, sticky=tk.W, padx=2)

        ttk.Label(top_frame, text="Класи k:").grid(row=1, column=8, padx=4, sticky=tk.E)
        self.bins_entry = ttk.Entry(top_frame, width=4)
        self.bins_entry.grid(row=1, column=9, sticky=tk.W, padx=2)

        ttk.Button(top_frame, text="Оновити гістограму", command=self.update_display).grid(row=1, column=10, padx=4)

        # --- Панель операцій ЛР 1 ---
        ttk.Button(action_frame, text="Замінити середнім періоду", command=self.handle_period_mean).pack(side=tk.LEFT, padx=3, pady=2)
        ttk.Button(action_frame, text="Замінити загальним середнім", command=self.handle_global_mean).pack(side=tk.LEFT, padx=3)
        ttk.Button(action_frame, text="Видалити дефекти", command=self.handle_drop).pack(side=tk.LEFT, padx=3)
        ttk.Separator(action_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=2)

        ttk.Button(action_frame, text="ln(x)", command=self.handle_log).pack(side=tk.LEFT, padx=3)
        ttk.Button(action_frame, text="Стандартизація Z", command=self.handle_standardize).pack(side=tk.LEFT, padx=3)

        ttk.Label(action_frame, text="c:").pack(side=tk.LEFT, padx=(4, 1))
        self.shift_c_entry = ttk.Entry(action_frame, width=4)
        self.shift_c_entry.insert(0, "5")
        self.shift_c_entry.pack(side=tk.LEFT, padx=1)
        ttk.Button(action_frame, text="Зсув (+c)", command=self.handle_shift).pack(side=tk.LEFT, padx=3)

        ttk.Button(action_frame, text="Скинути зміни", command=self.handle_reset).pack(side=tk.LEFT, padx=6)

        # --- Ліва панель: Інфо, Таблиця оцінок та Висновки ЛР 1 ---
        info_frame = ttk.LabelFrame(left_frame, text=" Інформаційна панель дефектів ")
        info_frame.pack(side=tk.TOP, fill=tk.X, padx=3, pady=2)
        self.info_text = tk.StringVar(value="Оберіть файл та натисніть 'Завантажити'.")
        ttk.Label(info_frame, textvariable=self.info_text, justify=tk.LEFT, font=("Consolas", 8)).pack(anchor=tk.W, padx=4, pady=2)

        table_frame = ttk.LabelFrame(left_frame, text=" Оцінки характеристик (ЛР 1) ")
        table_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=3, pady=2)

        cols = ("param", "lower", "point", "upper")
        self.table = ttk.Treeview(table_frame, columns=cols, show='headings', height=7)
        self.table.heading("param", text="Характеристика")
        self.table.heading("lower", text="θн")
        self.table.heading("point", text="θ")
        self.table.heading("upper", text="θв")
        self.table.column("param", width=130, anchor=tk.W)
        self.table.column("lower", width=70, anchor=tk.CENTER)
        self.table.column("point", width=70, anchor=tk.CENTER)
        self.table.column("upper", width=70, anchor=tk.CENTER)
        self.table.pack(fill=tk.BOTH, expand=True, padx=3, pady=2)

        conclusion_frame = ttk.LabelFrame(left_frame, text=" Висновки за ДІ (ЛР 1) ")
        conclusion_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=3, pady=2)
        self.conclusion_text = tk.StringVar(value="Очікування даних...")
        ttk.Label(conclusion_frame, textvariable=self.conclusion_text, justify=tk.LEFT, foreground="darkblue", font=("Arial", 8)).pack(anchor=tk.W, padx=4, pady=2)

        # --- Права панель: Вкладки Notebook (ЛР 1 + ЛР 2) ---
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        self.tab4 = ttk.Frame(self.notebook)
        self.tab5 = ttk.Frame(self.notebook)
        self.tab6 = ttk.Frame(self.notebook)

        self.notebook.add(self.tab1, text="Часовий ряд (ЛР1)")
        self.notebook.add(self.tab2, text="Гістограма (ЛР1)")
        self.notebook.add(self.tab3, text="Функції розподілу (ЛР1)")
        self.notebook.add(self.tab4, text="Варіаційний ряд (ЛР1)")
        self.notebook.add(self.tab5, text="2. Закон розподілу (ЛР2)")
        self.notebook.add(self.tab6, text="3. Моделювання та T-тест (ЛР2)")

        # Полотна графіків ЛР 1
        self.fig1, self.ax1 = plt.subplots(figsize=(6, 4))
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.tab1)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig2, self.ax2 = plt.subplots(figsize=(6, 4))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.tab2)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig3, self.ax3 = plt.subplots(figsize=(6, 4))
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self.tab3)
        self.canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Таблиця варіаційного ряду
        var_cols = ("idx", "val", "freq", "rel_freq")
        self.var_table = ttk.Treeview(self.tab4, columns=var_cols, show='headings')
        self.var_table.heading("idx", text="№")
        self.var_table.heading("val", text="Варіанта x_i")
        self.var_table.heading("freq", text="Частота n_i")
        self.var_table.heading("rel_freq", text="Відносна частота p_i")
        self.var_table.column("idx", width=40, anchor=tk.CENTER)
        self.var_table.column("val", width=90, anchor=tk.CENTER)
        self.var_table.column("freq", width=90, anchor=tk.CENTER)
        self.var_table.column("rel_freq", width=130, anchor=tk.CENTER)
        var_scroll = ttk.Scrollbar(self.tab4, orient=tk.VERTICAL, command=self.var_table.yview)
        self.var_table.configure(yscrollcommand=var_scroll.set)
        self.var_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        var_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Налаштування нових вкладок ЛР 2
        self.setup_tab_law_identification()
        self.setup_tab_simulation()

    # -------------------------------------------------------------
    # ВКЛАДКА 5: ІДЕНТИФІКАЦІЯ ЗАКОНУ РОЗПОДІЛУ (ЛР 2, п. 4.2)
    # -------------------------------------------------------------
    def setup_tab_law_identification(self):
        top_ctrl = ttk.Frame(self.tab5)
        top_ctrl.pack(side=tk.TOP, fill=tk.X, padx=5, pady=3)

        ttk.Button(top_ctrl, text="Оцінити 3 закони та порівняти D", command=self.update_law_identification).pack(side=tk.LEFT, padx=5)
        self.best_law_label = ttk.Label(top_ctrl, text="Рекомендований закон: —", font=("Arial", 9, "bold"), foreground="darkgreen")
        self.best_law_label.pack(side=tk.LEFT, padx=10)

        ttk.Button(top_ctrl, text="Взяти оцінки з даних для моделювання ➔", command=self.apply_best_law_to_sim).pack(side=tk.RIGHT, padx=5)

        # Панель попереднього вибору за A, E та V (п. 4.2.1, 4.2.5)
        pre_frame = ttk.LabelFrame(self.tab5, text=" Попередній аналіз за формою розподілу (A, E, V) ")
        pre_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        self.law_pre_text = tk.StringVar(value="Натисніть 'Оцінити 3 закони та порівняти D' після очищення даних.")
        ttk.Label(pre_frame, textvariable=self.law_pre_text, font=("Consolas", 8), foreground="navy", justify=tk.LEFT).pack(anchor=tk.W, padx=5, pady=2)

        # Таблиця порівняння законів за п. 4.2.2 («Закон», «Параметр», «θн», «θ̂», «θв», «σ{θ̂}», «Теор. A/E», «Факт. A/E», «D»)
        tbl_frame = ttk.LabelFrame(self.tab5, text=" Таблиця оцінок параметрів («Закон», «Параметр», «θн», «θ̂», «θв»), СКВ та критерію D ")
        tbl_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        law_cols = ("law", "param", "theta_l", "theta", "theta_u", "se", "th_ae", "obs_ae", "d_val")
        self.law_table = ttk.Treeview(tbl_frame, columns=law_cols, show='headings', height=5)
        self.law_table.heading("law", text="Закон")
        self.law_table.heading("param", text="Параметр")
        self.law_table.heading("theta_l", text="θн")
        self.law_table.heading("theta", text="θ̂")
        self.law_table.heading("theta_u", text="θв")
        self.law_table.heading("se", text="σ{θ̂}")
        self.law_table.heading("th_ae", text="Теор. A / E")
        self.law_table.heading("obs_ae", text="Факт. A / E")
        self.law_table.heading("d_val", text="D = max|Fn - F|")

        self.law_table.column("law", width=135, anchor=tk.W)
        self.law_table.column("param", width=65, anchor=tk.CENTER)
        self.law_table.column("theta_l", width=85, anchor=tk.CENTER)
        self.law_table.column("theta", width=85, anchor=tk.CENTER)
        self.law_table.column("theta_u", width=85, anchor=tk.CENTER)
        self.law_table.column("se", width=80, anchor=tk.CENTER)
        self.law_table.column("th_ae", width=95, anchor=tk.CENTER)
        self.law_table.column("obs_ae", width=115, anchor=tk.CENTER)
        self.law_table.column("d_val", width=110, anchor=tk.CENTER)
        self.law_table.pack(fill=tk.X, expand=True, padx=4, pady=2)

        # Графіки щільностей та функцій розподілу для трьох законів
        plots_frame = ttk.Frame(self.tab5)
        plots_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=2)

        self.fig_law, (self.ax_law_pdf, self.ax_law_cdf) = plt.subplots(1, 2, figsize=(9, 3.8))
        self.canvas_law = FigureCanvasTkAgg(self.fig_law, master=plots_frame)
        self.canvas_law.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # -------------------------------------------------------------
    # ВКЛАДКА 6: ІМІТАЦІЙНЕ МОДЕЛЮВАННЯ ТА T-ТЕСТ (ЛР 2, п. 4.3, 4.4)
    # -------------------------------------------------------------
    def setup_tab_simulation(self):
        ctrl_frame = ttk.LabelFrame(self.tab6, text=" Налаштування імітаційного експерименту ")
        ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        # Рядок 0: Вибір закону, параметри, N_один та окрема кнопка T-тесту 1 вибірки
        ttk.Label(ctrl_frame, text="Закон:").grid(row=0, column=0, padx=3, pady=2, sticky=tk.E)
        self.sim_law_combo = ttk.Combobox(ctrl_frame, values=["Нормальний", "Рівномірний", "Експоненціальний"], width=14, state="readonly")
        self.sim_law_combo.set("Нормальний")
        self.sim_law_combo.grid(row=0, column=1, padx=3, pady=2)
        self.sim_law_combo.bind("<<ComboboxSelected>>", self.on_sim_law_changed)

        ttk.Button(ctrl_frame, text="Взяти оцінки з даних", command=self.apply_best_law_to_sim).grid(row=0, column=2, padx=3)

        self.lbl_p1 = ttk.Label(ctrl_frame, text="m₀:")
        self.lbl_p1.grid(row=0, column=3, padx=2, sticky=tk.E)
        self.ent_p1 = ttk.Entry(ctrl_frame, width=9)
        self.ent_p1.insert(0, "273.6477")
        self.ent_p1.grid(row=0, column=4, padx=2)

        self.lbl_p2 = ttk.Label(ctrl_frame, text="σ₀:")
        self.lbl_p2.grid(row=0, column=5, padx=2, sticky=tk.E)
        self.ent_p2 = ttk.Entry(ctrl_frame, width=9)
        self.ent_p2.insert(0, "49.4248")
        self.ent_p2.grid(row=0, column=6, padx=2)

        ttk.Label(ctrl_frame, text="N (1 виб.):").grid(row=0, column=7, padx=3, sticky=tk.E)
        self.single_n_entry = ttk.Entry(ctrl_frame, width=6)
        self.single_n_entry.insert(0, "48")
        self.single_n_entry.grid(row=0, column=8, sticky=tk.W, padx=2)

        # КНОПКА ОДИНОЧНОГО Т-ТЕСТУ (на вимогу викладача)
        ttk.Button(
            ctrl_frame,
            text="🎲 Змоделювати 1 вибірку + T-тест",
            command=self.run_single_sample_ttest
        ).grid(row=0, column=9, columnspan=2, padx=4, pady=2, sticky=tk.EW)

        # Рядок 1: Налаштування серії експериментів R, alpha, N_list, seed, запуск R повторень і прогрес-бар
        ttk.Label(ctrl_frame, text="R (повторень):").grid(row=1, column=0, padx=3, pady=2, sticky=tk.E)
        self.sim_r_entry = ttk.Entry(ctrl_frame, width=6)
        self.sim_r_entry.insert(0, "200")
        self.sim_r_entry.grid(row=1, column=1, sticky=tk.W, padx=3)

        ttk.Label(ctrl_frame, text="Рівень α:").grid(row=1, column=2, padx=3, sticky=tk.E)
        self.sim_alpha_entry = ttk.Entry(ctrl_frame, width=5)
        self.sim_alpha_entry.insert(0, "0.05")
        self.sim_alpha_entry.grid(row=1, column=3, sticky=tk.W, padx=2)

        ttk.Label(ctrl_frame, text="Обсяги N:").grid(row=1, column=4, padx=2, sticky=tk.E)
        self.sim_n_entry = ttk.Entry(ctrl_frame, width=14)
        self.sim_n_entry.insert(0, "50, 400, 1000")
        self.sim_n_entry.grid(row=1, column=5, columnspan=2, sticky=tk.W, padx=2)

        ttk.Label(ctrl_frame, text="Seed:").grid(row=1, column=7, padx=3, sticky=tk.E)
        self.sim_seed_entry = ttk.Entry(ctrl_frame, width=5)
        self.sim_seed_entry.insert(0, "42")
        self.sim_seed_entry.grid(row=1, column=8, sticky=tk.W, padx=2)

        ttk.Button(ctrl_frame, text="▶ Запустити серію (R повт.)", command=self.run_simulation).grid(row=1, column=9, padx=3)
        ttk.Button(ctrl_frame, text="Експорт (Excel/CSV)", command=self.export_sim_tables).grid(row=1, column=10, padx=3)

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(ctrl_frame, variable=self.progress_var, maximum=100.0, length=85)
        self.progress_bar.grid(row=0, column=11, padx=3)
        self.progress_lbl = ttk.Label(ctrl_frame, text="Готово", font=("Arial", 8))
        self.progress_lbl.grid(row=1, column=11, padx=3)

        # Панель результатів ОДИНОЧНОЇ ВИБІРКИ ТА ЇЇ Т-ТЕСТУ (п. 4.3.2 + конкретний Т-тест)
        single_frame = ttk.LabelFrame(
            self.tab6,
            text=" Результат для ОДНІЄЇ змодельованої вибірки: характеристики (п. 4.3.2) та конкретний T-тест "
        )
        single_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        single_left = ttk.Frame(single_frame)
        single_left.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4, pady=2)

        s_cols = ("param", "theta0", "est", "se", "t_calc", "t_crit", "verdict")
        self.single_ttest_table = ttk.Treeview(single_left, columns=s_cols, show='headings', height=2)
        self.single_ttest_table.heading("param", text="Параметр")
        self.single_ttest_table.heading("theta0", text="Задане θ₀")
        self.single_ttest_table.heading("est", text="Оцінка θ̂ (виб.)")
        self.single_ttest_table.heading("se", text="СКВ σ{θ̂}")
        self.single_ttest_table.heading("t_calc", text="t = (θ₀ - θ̂)/σ{θ̂}")
        self.single_ttest_table.heading("t_crit", text="t_кр (α, ν=N-1)")
        self.single_ttest_table.heading("verdict", text="Рішення Т-тесту для цієї вибірки")

        self.single_ttest_table.column("param", width=65, anchor=tk.CENTER)
        self.single_ttest_table.column("theta0", width=80, anchor=tk.CENTER)
        self.single_ttest_table.column("est", width=95, anchor=tk.CENTER)
        self.single_ttest_table.column("se", width=75, anchor=tk.CENTER)
        self.single_ttest_table.column("t_calc", width=95, anchor=tk.CENTER)
        self.single_ttest_table.column("t_crit", width=90, anchor=tk.CENTER)
        self.single_ttest_table.column("verdict", width=220, anchor=tk.CENTER)
        self.single_ttest_table.pack(side=tk.TOP, fill=tk.X, expand=True)

        self.single_sample_text = tk.StringVar(
            value="Натисніть '🎲 Змоделювати 1 вибірку + T-тест' або '▶ Запустити серію (R повт.)'."
        )
        ttk.Label(
            single_left,
            textvariable=self.single_sample_text,
            font=("Consolas", 8),
            foreground="darkgreen",
            justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(2, 0))

        # Середня частина: Зведена таблиця T-тесту з динамічним заголовком (п. 4.4.2)
        self.res_frame = ttk.LabelFrame(self.tab6, text=" Зведена таблиця результатів T-тесту за R повторень (Закон: —, θ₀: —, R: —, α: —, seed: —) ")
        self.res_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        t_top = ttk.Frame(self.res_frame)
        t_top.pack(side=tk.TOP, fill=tk.X, padx=4, pady=1)
        ttk.Label(t_top, text="Показати результати для параметра:").pack(side=tk.LEFT, padx=4)
        self.sim_param_selector = ttk.Combobox(t_top, state="readonly", width=10)
        self.sim_param_selector.pack(side=tk.LEFT, padx=4)
        self.sim_param_selector.bind("<<ComboboxSelected>>", self.on_param_selector_changed)

        sim_cols = ("N", "nu", "tcrit", "mean_est", "std_est", "theo_se", "mean_t", "std_t", "rej", "delta")
        self.sim_table = ttk.Treeview(self.res_frame, columns=sim_cols, show='headings', height=4)
        self.sim_table.heading("N", text="N")
        self.sim_table.heading("nu", text="ν = N - 1")
        self.sim_table.heading("tcrit", text="t_кр")
        self.sim_table.heading("mean_est", text="Середня θ̂")
        self.sim_table.heading("std_est", text="СКВ θ̂ (за повт.)")
        self.sim_table.heading("theo_se", text="Теор. σ{θ̂}")
        self.sim_table.heading("mean_t", text="Середнє t")
        self.sim_table.heading("std_t", text="СКВ t")
        self.sim_table.heading("rej", text="Відхилень H₀, %")
        self.sim_table.heading("delta", text="Похибка δ, %")

        for c in sim_cols:
            self.sim_table.column(c, width=85, anchor=tk.CENTER)
        self.sim_table.pack(fill=tk.X, expand=True, padx=4, pady=1)

        # Нижня частина: 4 графіки моделювання (вибірка, t для N=50, t для N=1000, похибка delta(N))
        sim_plots_frame = ttk.Frame(self.tab6)
        sim_plots_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=2)

        self.fig_sim, self.sim_axes = plt.subplots(1, 4, figsize=(12.5, 3.0))
        self.ax_sim_sample, self.ax_sim_t50, self.ax_sim_t1000, self.ax_sim_delta = self.sim_axes
        self.canvas_sim = FigureCanvasTkAgg(self.fig_sim, master=sim_plots_frame)
        self.canvas_sim.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def on_sim_law_changed(self, event=None):
        law = self.sim_law_combo.get()
        if law == "Нормальний":
            self.lbl_p1.config(text="m₀:")
            self.lbl_p2.config(text="σ₀:")
            self.lbl_p2.grid()
            self.ent_p2.grid()
        elif law == "Рівномірний":
            self.lbl_p1.config(text="a₀:")
            self.lbl_p2.config(text="b₀:")
            self.lbl_p2.grid()
            self.ent_p2.grid()
        elif law == "Експоненціальний":
            self.lbl_p1.config(text="λ₀:")
            self.lbl_p2.grid_remove()
            self.ent_p2.grid_remove()

    def on_param_selector_changed(self, event=None):
        self.display_sim_table()
        self.plot_simulation_results()

    def _read_and_validate_law_params(self):
        law = self.sim_law_combo.get()
        try:
            p1 = float(self.ent_p1.get())
            p2 = float(self.ent_p2.get()) if law != "Експоненціальний" else None
        except ValueError:
            messagebox.showerror("Помилка", "Параметри закону розподілу мають бути числами!")
            return None, None, None

        true_params = {}
        if law == "Нормальний":
            if p2 <= 0:
                messagebox.showerror("Помилка", "Стандартне відхилення σ повинно бути строго > 0!")
                return None, None, None
            true_params = {"m": p1, "sigma": p2}
        elif law == "Рівномірний":
            if p1 >= p2:
                messagebox.showerror("Помилка", "Межа 'a' повинна бути строго меншою за 'b' (a < b)!")
                return None, None, None
            true_params = {"a": p1, "b": p2}
        elif law == "Експоненціальний":
            if p1 <= 0:
                messagebox.showerror("Помилка", "Параметр інтенсивності λ повинен бути строго > 0!")
                return None, None, None
            true_params = {"lambda": p1}

        try:
            alpha = float(self.sim_alpha_entry.get())
            if not (0.0 < alpha < 1.0):
                messagebox.showerror("Помилка", "Рівень значущості α повинен належати інтервалу (0; 1)!")
                return None, None, None
        except ValueError:
            messagebox.showerror("Помилка", "α повинно бути числом!")
            return None, None, None

        return law, true_params, alpha

    def run_single_sample_ttest(self):
        """
        Генерує одну конкретну вибірку обсягом N (з поля 'N (1 виб.)'),
        виконує для неї Т-тест по кожному параметру, оновлює таблицю одиночного
        Т-тесту, порівняння x̄, S², A, E з теоретичними та гістограму цієї вибірки.
        """
        law, true_params, alpha = self._read_and_validate_law_params()
        if law is None:
            return

        try:
            n_single = int(self.single_n_entry.get())
            if n_single < 4:
                messagebox.showerror("Помилка", "Обсяг N для однієї вибірки має бути цілим числом ≥ 4!")
                return
        except ValueError:
            messagebox.showerror("Помилка", "Обсяг N (1 виб.) має бути цілим числом!")
            return

        try:
            base_seed = int(self.sim_seed_entry.get())
        except ValueError:
            base_seed = 42

        used_seed = base_seed + self.single_click_counter
        self.single_click_counter += 1

        self.single_sample_comp = compare_single_sample_with_theory(
            law, true_params, size=n_single, alpha=alpha, seed=used_seed
        )
        self.req_sample_sizes = compute_required_sample_sizes(law, true_params, alpha=alpha)

        self._update_single_sample_ui(law, true_params)
        self._plot_single_sample_histogram(law, true_params)

    def _update_single_sample_ui(self, law, true_params):
        if self.single_sample_comp is None:
            return

        for row in self.single_ttest_table.get_children():
            self.single_ttest_table.delete(row)

        t_res = self.single_sample_comp["t_test"]
        for p_name, d in t_res.items():
            verdict = (
                f"|t| ≤ t_кр → H₀ ПРИЙМАЄТЬСЯ ({p_name}={p_name}₀)"
                if not d["rejected"]
                else f"|t| > t_кр → H₀ ВІДХИЛЯЄТЬСЯ (похибка I роду)"
            )
            fmt = ".6f" if p_name == "lambda" else ".4f"
            self.single_ttest_table.insert("", tk.END, values=(
                p_name,
                f"{d['theta_0']:{fmt}}",
                f"{d['est']:{fmt}}",
                f"{d['se_est']:{fmt}}",
                f"{d['t']:+.4f} (|t|={abs(d['t']):.4f})",
                f"±{d['t_crit']:.4f}",
                verdict
            ))

        s_pts = self.single_sample_comp["sample_stats"]
        t_pts = self.single_sample_comp["theory_stats"]
        n_single = self.single_sample_comp["N"]
        used_seed = self.single_sample_comp["seed"]
        alpha = self.single_sample_comp["alpha"]

        req_str_parts = []
        if self.req_sample_sizes:
            for r_item in self.req_sample_sizes:
                if r_item["delta_pct"] in (10.0, 5.0, 2.0, 1.0):
                    p_vals = ", ".join([f"N*({k})={r_item[k]}" for k in true_params.keys()])
                    req_str_parts.append(f"δ={int(r_item['delta_pct'])}%: [{p_vals}]")
        req_line = "  |  ".join(req_str_parts)

        comp_text = (
            f"• П. 4.3.2 (1 вибірка, N={n_single}, seed={used_seed}): "
            f"x̄ = {s_pts['mean']:.4f} (теор. {t_pts['mean']:.4f}), "
            f"S² = {s_pts['variance']:.4f} (теор. {t_pts['variance']:.4f}), "
            f"A = {s_pts['skewness']:.4f} (теор. {t_pts['skewness']:.1f}), "
            f"E = {s_pts['kurtosis']:.4f} (теор. {t_pts['kurtosis']:.1f})\n"
            f"• П. 7.7 (Необхідний обсяг N* при α={alpha}):  {req_line}"
        )
        self.single_sample_text.set(comp_text)

    def _plot_single_sample_histogram(self, law, params):
        if self.single_sample_comp is None:
            return
        self.ax_sim_sample.clear()
        sample = self.single_sample_comp["sample"]
        n_sample = self.single_sample_comp["N"]
        used_seed = self.single_sample_comp["seed"]

        counts, edges = np.histogram(sample, bins=9, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = edges[1:] - edges[:-1]
        self.ax_sim_sample.bar(
            centers, counts, width=widths * 0.9,
            color='lightskyblue', edgecolor='black', alpha=0.75,
            label=f'Вибірка (N={n_sample})'
        )

        grid_x = np.linspace(np.min(sample) - 0.15 * np.std(sample), np.max(sample) + 0.15 * np.std(sample), 200)
        if law == "Нормальний":
            self.ax_sim_sample.plot(grid_x, norm_pdf(grid_x, params["m"], params["sigma"]), 'r-', lw=2, label='Теор. f(x)')
        elif law == "Рівномірний":
            self.ax_sim_sample.plot(grid_x, uniform_pdf(grid_x, params["a"], params["b"]), 'r-', lw=2, label='Теор. f(x)')
        elif law == "Експоненціальний":
            self.ax_sim_sample.plot(grid_x, exp_pdf(grid_x, params["lambda"]), 'r-', lw=2, label='Теор. f(x)')

        self.ax_sim_sample.set_title(f"П. 4.3.2: 1 вибірка (N={n_sample}, seed={used_seed})", fontsize=8.5, fontweight='bold')
        self.ax_sim_sample.legend(fontsize=7)
        self.ax_sim_sample.grid(True, linestyle='--', alpha=0.5)
        self.fig_sim.tight_layout()
        self.canvas_sim.draw()

    def update_law_identification(self):
        if self.working_df is None:
            messagebox.showwarning("Увага", "Спочатку завантажте та очистіть вибірку!")
            return

        clean_vals = self.working_df['value'].dropna().tolist()
        if len(clean_vals) < 4:
            return

        x = np.sort(np.array(clean_vals, dtype=float))
        N = len(x)
        gamma = self.get_gamma()
        alpha = 1.0 - gamma
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=N - 1))

        pts, ci = compute_all_statistics(clean_vals, gamma=gamma)
        obs_a = pts["skewness"]
        obs_e = pts["kurtosis"]
        obs_v = pts["std"] / pts["mean"] if pts["mean"] != 0 else 0.0

        fit_n = fit_normal(x)
        fit_u = fit_uniform(x)

        has_negative = np.any(x <= 0)
        fit_e = fit_exponential(x) if not has_negative else None

        d_norm = compute_ks_distance(x, norm_cdf, fit_n["m"], fit_n["sigma"])
        d_unif = compute_ks_distance(x, uniform_cdf, fit_u["a"], fit_u["b"])
        d_exp = compute_ks_distance(x, exp_cdf, fit_e["lambda"]) if fit_e else 999.0

        self.fitted_models = {
            "Нормальний": {"params": {"m": fit_n["m"], "sigma": fit_n["sigma"]}, "D": d_norm},
            "Рівномірний": {"params": {"a": fit_u["a"], "b": fit_u["b"]}, "D": d_unif},
            "Експоненціальний": {"params": {"lambda": fit_e["lambda"] if fit_e else 0.0}, "D": d_exp}
        }

        for row in self.law_table.get_children():
            self.law_table.delete(row)

        # 1. Нормальний
        m_l, m_u = fit_n['m'] - t_crit * fit_n['se_m'], fit_n['m'] + t_crit * fit_n['se_m']
        s_l, s_u = max(0.0, fit_n['sigma'] - t_crit * fit_n['se_sigma']), fit_n['sigma'] + t_crit * fit_n['se_sigma']
        self.law_table.insert("", tk.END, values=(
            "Нормальний N(m, σ)", "m",
            f"{m_l:.4f}", f"{fit_n['m']:.4f}", f"{m_u:.4f}", f"{fit_n['se_m']:.4f}",
            "A=0, E=0", f"A={obs_a:.3f}, E={obs_e:.3f}", f"{d_norm:.4f}"
        ))
        self.law_table.insert("", tk.END, values=(
            "", "σ",
            f"{s_l:.4f}", f"{fit_n['sigma']:.4f}", f"{s_u:.4f}", f"{fit_n['se_sigma']:.4f}",
            "", f"V={obs_v:.3f}", ""
        ))

        # 2. Рівномірний
        a_l, a_u = fit_u['a'] - t_crit * fit_u['se_a'], fit_u['a'] + t_crit * fit_u['se_a']
        b_l, b_u = fit_u['b'] - t_crit * fit_u['se_b'], fit_u['b'] + t_crit * fit_u['se_b']
        self.law_table.insert("", tk.END, values=(
            "Рівномірний U(a, b)", "a",
            f"{a_l:.4f}", f"{fit_u['a']:.4f}", f"{a_u:.4f}", f"{fit_u['se_a']:.4f}",
            "A=0, E=-1.2", f"A={obs_a:.3f}, E={obs_e:.3f}", f"{d_unif:.4f}"
        ))
        self.law_table.insert("", tk.END, values=(
            "", "b",
            f"{b_l:.4f}", f"{fit_u['b']:.4f}", f"{b_u:.4f}", f"{fit_u['se_b']:.4f}",
            "", f"V={obs_v:.3f}", ""
        ))

        # 3. Експоненціальний
        if fit_e:
            lam_l = max(0.0, fit_e['lambda'] - t_crit * fit_e['se_lambda'])
            lam_u = fit_e['lambda'] + t_crit * fit_e['se_lambda']
            self.law_table.insert("", tk.END, values=(
                "Експоненціальний Exp(λ)", "λ",
                f"{lam_l:.6f}", f"{fit_e['lambda']:.6f}", f"{lam_u:.6f}", f"{fit_e['se_lambda']:.6f}",
                "A=2, E=6", f"A={obs_a:.3f}, E={obs_e:.3f}", f"{d_exp:.4f}"
            ))
        else:
            self.law_table.insert("", tk.END, values=(
                "Експоненціальний Exp(λ)", "λ",
                "—", "Н/Д (є x ≤ 0)", "—", "—",
                "A=2, E=6", f"A={obs_a:.3f}, E={obs_e:.3f}", "—"
            ))

        d_dict = {"Нормальний": d_norm, "Рівномірний": d_unif, "Експоненціальний": d_exp}
        best_law = min(d_dict, key=d_dict.get)
        self.best_law_name = best_law
        self.best_law_label.config(text=f"Рекомендований закон: {best_law} (min D = {d_dict[best_law]:.4f})")

        a_ci_str = f"[{ci['A']['theta_l']:.3f}; {ci['A']['theta_u']:.3f}]"
        e_ci_str = f"[{ci['E']['theta_l']:.3f}; {ci['E']['theta_u']:.3f}]"
        pre_info = (
            f"• ДІ для A: {a_ci_str} (точкова A = {obs_a:.4f})  |  ДІ для E: {e_ci_str} (точкова E = {obs_e:.4f})  |  Коеф. варіації V = S/x̄ = {obs_v:.4f}\n"
            f"• Висновок (п. 4.2.5): Найменше відхилення D має {best_law} закон (D = {d_dict[best_law]:.4f}), що повністю узгоджується з формою (A, E, V)."
        )
        self.law_pre_text.set(pre_info)

        self.ax_law_pdf.clear()
        self.ax_law_cdf.clear()

        grid_x = np.linspace(np.min(x) - 0.2 * np.std(x), np.max(x) + 0.2 * np.std(x), 300)

        counts, edges = np.histogram(x, bins=7, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = edges[1:] - edges[:-1]
        self.ax_law_pdf.bar(centers, counts, width=widths * 0.9, color='lightgray', edgecolor='black', alpha=0.6, label='Дані (гістограма)')

        self.ax_law_pdf.plot(grid_x, norm_pdf(grid_x, fit_n["m"], fit_n["sigma"]), 'b-', lw=2, label=f'Норм. (D={d_norm:.4f})')
        self.ax_law_pdf.plot(grid_x, uniform_pdf(grid_x, fit_u["a"], fit_u["b"]), 'g--', lw=2, label=f'Рівном. (D={d_unif:.4f})')
        if fit_e:
            self.ax_law_pdf.plot(grid_x, exp_pdf(grid_x, fit_e["lambda"]), 'r:', lw=2, label=f'Експ. (D={d_exp:.4f})')

        self.ax_law_pdf.set_title("Щільності f(x) трьох законів", fontsize=10, fontweight='bold')
        self.ax_law_pdf.set_xlabel("x")
        self.ax_law_pdf.set_ylabel("Щільність ймовірності f(x)")
        self.ax_law_pdf.grid(True, linestyle='--', alpha=0.5)
        self.ax_law_pdf.legend(fontsize=8)

        fn_y = np.arange(1, N + 1) / N
        self.ax_law_cdf.step(x, fn_y, where='post', color='black', lw=2, label='Емпірична Fn(x)')
        self.ax_law_cdf.plot(grid_x, norm_cdf(grid_x, fit_n["m"], fit_n["sigma"]), 'b-', lw=1.8, label='F(x) Нормальний')
        self.ax_law_cdf.plot(grid_x, uniform_cdf(grid_x, fit_u["a"], fit_u["b"]), 'g--', lw=1.8, label='F(x) Рівномірний')
        if fit_e:
            self.ax_law_cdf.plot(grid_x, exp_cdf(grid_x, fit_e["lambda"]), 'r:', lw=1.8, label='F(x) Експоненціальний')

        self.ax_law_cdf.set_title("Функції розподілу Fn(x) та F(x)", fontsize=10, fontweight='bold')
        self.ax_law_cdf.set_xlabel("x")
        self.ax_law_cdf.set_ylabel("F(x)")
        self.ax_law_cdf.grid(True, linestyle='--', alpha=0.5)
        self.ax_law_cdf.legend(fontsize=8, loc='lower right')

        self.fig_law.tight_layout()
        self.canvas_law.draw()

    def apply_best_law_to_sim(self):
        if not self.fitted_models:
            self.update_law_identification()
        if not self.fitted_models:
            return

        law = self.best_law_name
        self.sim_law_combo.set(law)
        self.on_sim_law_changed()

        params = self.fitted_models[law]["params"]
        if law == "Нормальний":
            self.ent_p1.delete(0, tk.END)
            self.ent_p1.insert(0, f"{params['m']:.4f}")
            self.ent_p2.delete(0, tk.END)
            self.ent_p2.insert(0, f"{params['sigma']:.4f}")
        elif law == "Рівномірний":
            self.ent_p1.delete(0, tk.END)
            self.ent_p1.insert(0, f"{params['a']:.4f}")
            self.ent_p2.delete(0, tk.END)
            self.ent_p2.insert(0, f"{params['b']:.4f}")
        elif law == "Експоненціальний":
            self.ent_p1.delete(0, tk.END)
            self.ent_p1.insert(0, f"{params['lambda']:.6f}")

        if self.working_df is not None:
            n_clean = len(self.working_df['value'].dropna())
            self.single_n_entry.delete(0, tk.END)
            self.single_n_entry.insert(0, str(n_clean))

        self.notebook.select(self.tab6)

    def update_progress(self, current_step, total_steps):
        pct = (current_step / total_steps) * 100.0 if total_steps > 0 else 100.0
        self.progress_var.set(pct)
        self.progress_lbl.config(text=f"{int(pct)}%")
        self.root.update_idletasks()

    def run_simulation(self):
        law, true_params, alpha = self._read_and_validate_law_params()
        if law is None:
            return

        try:
            R = int(self.sim_r_entry.get())
            if not (200 <= R <= 500):
                messagebox.showerror("Помилка", "Кількість повторень R повинна бути в межах [200; 500]!")
                return
        except ValueError:
            messagebox.showerror("Помилка", "R повинно бути цілим числом!")
            return

        try:
            n_tokens = self.sim_n_entry.get().split(',')
            N_list = [int(tok.strip()) for tok in n_tokens if tok.strip()]
            if not N_list or any(n < 4 for n in N_list):
                messagebox.showerror("Помилка", "Обсяги N мають бути цілими числами ≥ 4!")
                return
        except ValueError:
            messagebox.showerror("Помилка", "Некоректний перелік чисел N (вводьте через кому, напр. 50, 400, 1000)!")
            return

        try:
            seed = int(self.sim_seed_entry.get())
        except ValueError:
            seed = 42

        try:
            n_single = int(self.single_n_entry.get())
            if n_single < 4:
                n_single = 48
        except ValueError:
            n_single = 48

        self.current_sim_params = {"law": law, "params": true_params, "R": R, "alpha": alpha, "N_list": N_list, "seed": seed}

        self.single_click_counter = 0
        self.single_sample_comp = compare_single_sample_with_theory(law, true_params, size=n_single, alpha=alpha, seed=seed)

        self.root.config(cursor="watch")
        self.progress_var.set(0.0)
        self.progress_lbl.config(text="0%")
        self.root.update()
        try:
            self.sim_summary_tables, self.sim_raw_t = run_full_simulation(
                law, true_params, N_list, R=R, alpha=alpha, master_seed=seed,
                progress_callback=self.update_progress
            )
            self.req_sample_sizes = compute_required_sample_sizes(law, true_params, alpha=alpha)
        finally:
            self.root.config(cursor="")
            self.progress_lbl.config(text="Готово (100%)")

        self._update_single_sample_ui(law, true_params)

        param_keys = list(true_params.keys())
        self.sim_param_selector["values"] = param_keys
        self.sim_param_selector.set(param_keys[0])
        self.display_sim_table()
        self.plot_simulation_results()

        messagebox.showinfo("Успіх", f"Моделювання для {law} закону ({R} повторень) успішно виконано!")

    def display_sim_table(self, event=None):
        if not self.sim_summary_tables or not self.current_sim_params:
            return
        p = self.sim_param_selector.get()
        if p not in self.sim_summary_tables:
            return

        cp = self.current_sim_params
        theta_0_val = cp["params"][p]
        params_str = ", ".join([f"{k}₀={v:.4f}" for k, v in cp["params"].items()])
        self.res_frame.config(
            text=f" Зведена таблиця T-тесту | Закон: {cp['law']} ({params_str}) | Параметр: {p} (θ₀={theta_0_val:.4f}) | R={cp['R']}, α={cp['alpha']}, seed={cp['seed']} "
        )

        for row in self.sim_table.get_children():
            self.sim_table.delete(row)

        for row_data in self.sim_summary_tables[p]:
            self.sim_table.insert("", tk.END, values=(
                row_data["N"],
                row_data["nu"],
                f"{row_data['t_crit']:.4f}",
                f"{row_data['mean_est']:.4f}",
                f"{row_data['std_est']:.4f}",
                f"{row_data['theory_se']:.4f}",
                f"{row_data['mean_t']:.4f}",
                f"{row_data['std_t']:.4f}",
                f"{row_data['rej_rate']:.1f}%",
                f"{row_data['delta'] * 100.0:.2f}%"
            ))

    def plot_simulation_results(self):
        if not self.current_sim_params or not self.sim_summary_tables:
            return

        law = self.current_sim_params["law"]
        params = self.current_sim_params["params"]
        N_list = self.current_sim_params["N_list"]
        alpha = self.current_sim_params["alpha"]
        p_name = self.sim_param_selector.get()

        for ax in self.sim_axes:
            ax.clear()

        if self.single_sample_comp is not None:
            sample = self.single_sample_comp["sample"]
            n_sample = self.single_sample_comp["N"]
            used_seed = self.single_sample_comp["seed"]
        else:
            n_sample = N_list[0]
            used_seed = self.current_sim_params["seed"]
            sample = generate_sample(law, params, size=n_sample, seed=used_seed)

        counts, edges = np.histogram(sample, bins=9, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = edges[1:] - edges[:-1]
        self.ax_sim_sample.bar(centers, counts, width=widths * 0.9, color='lightskyblue', edgecolor='black', alpha=0.7, label=f'Вибірка (N={n_sample})')

        grid_x = np.linspace(np.min(sample) - 0.1 * np.std(sample), np.max(sample) + 0.1 * np.std(sample), 200)
        if law == "Нормальний":
            self.ax_sim_sample.plot(grid_x, norm_pdf(grid_x, params["m"], params["sigma"]), 'r-', lw=2, label='Теор. f(x)')
        elif law == "Рівномірний":
            self.ax_sim_sample.plot(grid_x, uniform_pdf(grid_x, params["a"], params["b"]), 'r-', lw=2, label='Теор. f(x)')
        elif law == "Експоненціальний":
            self.ax_sim_sample.plot(grid_x, exp_pdf(grid_x, params["lambda"]), 'r-', lw=2, label='Теор. f(x)')

        self.ax_sim_sample.set_title(f"П. 4.3.2: 1 вибірка (N={n_sample}, seed={used_seed})", fontsize=8.5, fontweight='bold')
        self.ax_sim_sample.legend(fontsize=7)
        self.ax_sim_sample.grid(True, linestyle='--', alpha=0.5)

        n_first = 50 if (p_name, 50) in self.sim_raw_t else N_list[0]
        self._draw_t_hist(self.ax_sim_t50, p_name, n_first, alpha, color='navajowhite')

        n_last = 1000 if (p_name, 1000) in self.sim_raw_t else N_list[-1]
        self._draw_t_hist(self.ax_sim_t1000, p_name, n_last, alpha, color='lightgreen')

        rows = self.sim_summary_tables[p_name]
        ns = [r["N"] for r in rows]
        deltas = [r["delta"] * 100.0 for r in rows]

        self.ax_sim_delta.plot(ns, deltas, 'o-', color='purple', lw=2, label=f'δ(N) для {p_name}')
        self.ax_sim_delta.axhline(5.0, color='gray', linestyle=':', label='Поріг 5%')
        self.ax_sim_delta.set_xscale('log')
        self.ax_sim_delta.set_title(f"Точність δ(N) для {p_name} (log)", fontsize=8.5, fontweight='bold')
        self.ax_sim_delta.set_xlabel("N (log-шкала)", fontsize=8)
        self.ax_sim_delta.set_ylabel("δ, %", fontsize=8)
        self.ax_sim_delta.legend(fontsize=7)
        self.ax_sim_delta.grid(True, which="both", linestyle='--', alpha=0.5)

        self.fig_sim.tight_layout()
        self.canvas_sim.draw()

    def _draw_t_hist(self, ax, p_name, n_val, alpha, color='navajowhite'):
        t_vals = self.sim_raw_t.get((p_name, n_val), [])
        if not t_vals:
            return
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n_val - 1))
        t_counts, t_edges = np.histogram(t_vals, bins=14, density=True)
        t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
        t_widths = t_edges[1:] - t_edges[:-1]
        ax.bar(t_centers, t_counts, width=t_widths * 0.9, color=color, edgecolor='black', alpha=0.75)

        grid_t = np.linspace(-4, 4, 200)
        ax.plot(grid_t, stats.norm.pdf(grid_t), 'b-', lw=1.8, label='N(0, 1)')
        ax.axvline(t_crit, color='red', linestyle='--', lw=1.5, label=f'±t_кр ({t_crit:.2f})')
        ax.axvline(-t_crit, color='red', linestyle='--', lw=1.5)
        ax.set_title(f"П. 4.4.3: t ({p_name}, N={n_val})", fontsize=8.5, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.5)

    def export_sim_tables(self):
        if not self.sim_summary_tables:
            messagebox.showwarning("Увага", "Спочатку запустіть моделювання!")
            return

        fpath = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")])
        if not fpath:
            return

        try:
            if fpath.endswith(".xlsx"):
                with pd.ExcelWriter(fpath) as writer:
                    for p, rows in self.sim_summary_tables.items():
                        df = pd.DataFrame(rows)
                        df.to_excel(writer, sheet_name=f"T_тест_{p}", index=False)
                    if self.req_sample_sizes:
                        pd.DataFrame(self.req_sample_sizes).to_excel(writer, sheet_name="Необхідний_обсяг_N", index=False)
            else:
                p = self.sim_param_selector.get()
                pd.DataFrame(self.sim_summary_tables[p]).to_csv(fpath, index=False)
            messagebox.showinfo("Успіх", f"Таблиці успішно експортовано у:\n{fpath}")
        except Exception as e:
            messagebox.showerror("Помилка експорту", str(e))

    def browse_file(self):
        f = filedialog.askopenfilename(filetypes=[("Data files", "*.xlsx *.xls *.csv *.txt")])
        if f:
            self.filepath = f
            self.file_label.config(text=os.path.basename(f), foreground="black")
            sheet_names = get_excel_sheet_names(f)
            if sheet_names:
                self.sheet_combo['values'] = sheet_names
                self.sheet_combo.set(sheet_names[1] if len(sheet_names) > 1 else sheet_names[0])
            else:
                self.sheet_combo['values'] = []
                self.sheet_combo.set("")

    def get_real_events_indices(self):
        text = self.real_events_entry.get().strip()
        if not text or self.raw_df is None:
            return []
        real_idx = []
        tokens = [t.strip() for t in text.split(',') if t.strip()]
        times_map = {str(self.raw_df.loc[i, 'time']).strip(): i for i in self.raw_df.index}
        for token in tokens:
            if token in times_map:
                real_idx.append(times_map[token])
            elif token.isdigit():
                i = int(token)
                if i in self.raw_df.index:
                    real_idx.append(i)
        return sorted(list(set(real_idx)))

    def load_dataset(self):
        if not self.filepath:
            messagebox.showwarning("Помилка", "Будь ласка, спочатку оберіть файл.")
            return
        sheet = self.sheet_combo.get().strip()
        if not sheet:
            sheet = 0
        try:
            self.raw_df = load_data(self.filepath, sheet_name=sheet)
            self.working_df = self.raw_df.copy()
            self.process_defects()
            self.update_display()
        except Exception as e:
            messagebox.showerror("Помилка завантаження", str(e))

    def process_defects(self):
        try:
            l = float(self.min_entry.get())
            u = float(self.max_entry.get())
        except ValueError:
            messagebox.showerror("Помилка", "Межі аномалій мають бути числовими.")
            return

        self.missing_indices, self.anomaly_indices = detect_defects(self.raw_df, l, u)
        real_idx = self.get_real_events_indices()

        miss_times = [str(self.raw_df.loc[i, 'time']) for i in self.missing_indices]
        miss_str = ", ".join(miss_times) if miss_times else "немає"

        anom_info = [f"{self.raw_df.loc[i, 'time']} ({self.raw_df.loc[i, 'value']})" for i in self.anomaly_indices]
        anom_str = ", ".join(anom_info) if anom_info else "немає"

        p_means = compute_period_means(self.raw_df, l, u, real_idx)
        means_str = f"Ніч: {p_means['Ніч']:.1f} | Ранок: {p_means['Ранок']:.1f} | День: {p_means['День']:.1f} | Вечір: {p_means['Вечір']:.1f}"

        info = (
            f"Записів N = {len(self.raw_df)}\n"
            f"Пропуски ({len(self.missing_indices)}): {miss_str}\n"
            f"Аномалії поза [{l}; {u}] ({len(self.anomaly_indices)}): {anom_str}\n"
            f"Реальні події: {len(real_idx)} | Технічні збої: {len(self.anomaly_indices) - len(real_idx)}\n"
            f"Середні періодів: {means_str}"
        )
        self.info_text.set(info)

    def get_gamma(self):
        try:
            g = float(self.gamma_entry.get())
            if 0 < g < 1:
                return g
        except ValueError:
            pass
        return 0.95

    def update_display(self):
        if self.working_df is None:
            return
        gamma = self.get_gamma()
        clean_vals = self.working_df['value'].dropna().tolist()
        if len(clean_vals) < 4:
            return

        real_idx = self.get_real_events_indices()

        pts, ci = compute_all_statistics(clean_vals, gamma=gamma)
        for row in self.table.get_children():
            self.table.delete(row)

        display_keys = ["x_bar", "S2", "A", "E"]
        for key in display_keys:
            item = ci[key]
            self.table.insert("", tk.END, values=(
                item["name"],
                f"{item['theta_l']:.4f}",
                f"{item['theta']:.4f}",
                f"{item['theta_u']:.4f}"
            ))

        self.table.insert("", tk.END, values=("СКВ (S)", "-", f"{pts['std']:.4f}", "-"))
        self.table.insert("", tk.END, values=("Медіана (Me)", "-", f"{pts['median']:.4f}", "-"))
        v_coeff = pts['std'] / pts['mean'] if pts['mean'] != 0 else 0.0
        self.table.insert("", tk.END, values=("Коеф. варіації (V)", "-", f"{v_coeff:.4f}", "-"))

        c_text = (
            f"• Асиметрія [{ci['A']['theta_l']:.2f}; {ci['A']['theta_u']:.2f}]: {ci['A']['conclusion']}\n"
            f"• Ексцес [{ci['E']['theta_l']:.2f}; {ci['E']['theta_u']:.2f}]: {ci['E']['conclusion']}"
        )
        self.conclusion_text.set(c_text)

        plot_time_series(self.raw_df, self.working_df, self.missing_indices, self.anomaly_indices, real_events_idx=real_idx, ax=self.ax1)
        self.fig1.tight_layout()
        self.canvas1.draw()

        num_bins = None
        b_val = self.bins_entry.get().strip()
        if b_val and b_val.isdigit():
            num_bins = int(b_val)

        _, k_calc, h_calc = plot_histogram(clean_vals, num_bins=num_bins, ax=self.ax2)
        self.fig2.tight_layout()
        self.canvas2.draw()

        plot_distribution_functions(clean_vals, gamma=gamma, ax=self.ax3)
        self.fig3.tight_layout()
        self.canvas3.draw()

        var_df = build_variational_series(clean_vals)
        for row in self.var_table.get_children():
            self.var_table.delete(row)
        for i, r in var_df.iterrows():
            self.var_table.insert("", tk.END, values=(
                i + 1,
                f"{r['x_i']:.2f}",
                int(r['n_i']),
                f"{r['p_i']:.4f}"
            ))

    def handle_period_mean(self):
        if self.raw_df is None:
            return
        try:
            l = float(self.min_entry.get())
            u = float(self.max_entry.get())
        except ValueError:
            messagebox.showerror("Помилка", "Межі мають бути числовими.")
            return
        real_idx = self.get_real_events_indices()
        self.working_df, _ = clean_data_by_period_mean(self.raw_df, l, u, real_events_indices=real_idx)
        self.process_defects()
        self.update_display()
        self.update_law_identification()
        messagebox.showinfo("Успіх", "Пропуски та технічні збої замінено середніми відповідних періодів.")

    def handle_global_mean(self):
        if self.raw_df is None:
            return
        try:
            l = float(self.min_entry.get())
            u = float(self.max_entry.get())
        except ValueError:
            messagebox.showerror("Помилка", "Межі мають бути числовими.")
            return
        real_idx = self.get_real_events_indices()
        self.working_df = clean_data_by_global_mean(self.raw_df, l, u, real_events_indices=real_idx)
        self.process_defects()
        self.update_display()
        self.update_law_identification()

    def handle_drop(self):
        if self.raw_df is None:
            return
        try:
            l = float(self.min_entry.get())
            u = float(self.max_entry.get())
        except ValueError:
            messagebox.showerror("Помилка", "Межі мають бути числовими.")
            return
        real_idx = self.get_real_events_indices()
        self.working_df = clean_data_by_drop(self.raw_df, l, u, real_events_indices=real_idx)
        self.process_defects()
        self.update_display()
        self.update_law_identification()

    def handle_log(self):
        if self.working_df is None:
            return
        before_vals = self.working_df['value'].dropna().tolist()
        if len(before_vals) < 4:
            return
        pts_before, _ = compute_all_statistics(before_vals)

        vals, shift = apply_log_transform(self.working_df['value'])
        self.working_df['value'] = vals
        self.update_display()

        after_vals = self.working_df['value'].dropna().tolist()
        pts_after, _ = compute_all_statistics(after_vals)

        msg = (
            f"Виконано логарифмування y = ln(x) (зсув c = {shift:.1f})\n\n"
            f"Порівняння показників форми розподілу:\n"
            f"• Асиметрія (A):  до = {pts_before['skewness']:.4f}  →  після = {pts_after['skewness']:.4f}\n"
            f"• Ексцес (E):     до = {pts_before['kurtosis']:.4f}  →  після = {pts_after['kurtosis']:.4f}\n\n"
            f"Висновок: |A| {'зменшилась (наближення до 0)' if abs(pts_after['skewness']) < abs(pts_before['skewness']) else 'збільшилась'}."
        )
        messagebox.showinfo("Порівняння до/після логарифмування", msg)

    def handle_standardize(self):
        if self.working_df is None:
            return
        self.working_df['value'] = apply_standardization(self.working_df['value'])
        self.update_display()
        messagebox.showinfo("Перетворення", "Виконано стандартизацію: Z = (x - x̄) / S.")

    def handle_shift(self):
        if self.working_df is None:
            return
        try:
            c = float(self.shift_c_entry.get())
        except ValueError:
            messagebox.showerror("Помилка", "Значення константи c має бути числом.")
            return
        self.working_df['value'] = apply_shift(self.working_df['value'], c)
        self.update_display()
        messagebox.showinfo("Зсув", f"Виконано лінійний зсув: y = x + {c}.")

    def handle_reset(self):
        if self.raw_df is not None:
            self.working_df = self.raw_df.copy()
            self.process_defects()
            self.update_display()