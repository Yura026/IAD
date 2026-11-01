"""
gui_lab3.py — Графічний інтерфейс програми (Tkinter) для ЛР 3 та ЛР 4:
- фільтрація аномалій за правилом 3-сигма;
- точкові й інтервальні оцінки характеристик;
- оцінка сили та значущості зв'язку за Пірсоном (r + ДІ Фішера) та Спірменом (rho);
- таблиця «Лінійна регресія» (МНК-оцінки a та b, СКВ, ДІ, t-статистики, p-value, висновки, R^2);
- вкладки візуалізації: Кореляційне поле з лінією регресії, 2D-гістограма, 3D-гістограма,
  3D-поверхня щільності f(x, y) та таблиці частот (n_ij, p_ij*).
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

from generator_2d import generate_2d_normal
from data_io import load_2d_data, save_2d_data
from statistics_2d import (
    filter_outliers_3sigma, compute_point_estimates, compute_confidence_intervals,
    test_correlation_significance, compute_spearman_correlation,
    compute_linear_regression, compute_2d_grouping, test_chi2_goodness_of_fit
)
from visualization_2d import (
    plot_scatter_and_ellipses, plot_2d_heatmap,
    plot_3d_histogram, plot_3d_density_surface
)


class Lab3App:
    def __init__(self, root):
        self.root = root
        self.root.title("ІАД — Лабораторні роботи № 3 та № 4: Двовимірні вибірки та Лінійна регресія")
        self.root.geometry("1480x940")

        # Сирі (початкові) та робочі (після 3-сигма) дані вибірки
        self.raw_x = None
        self.raw_y = None
        self.x = None
        self.y = None
        self.current_filepath = None
        self.is_generated = True

        self.setup_ui()
        # Генеруємо початкові дані за Варіантом 1 автоматично при старті
        self.handle_generate()

    def setup_ui(self):
        # 1. Верхня панель параметрів та дій
        top_frame = ttk.LabelFrame(self.root, text=" 1. Генерація, завантаження даних та налаштування аналізу ")
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=3)

        row0 = ttk.Frame(top_frame)
        row0.pack(fill=tk.X, padx=4, pady=2)

        ttk.Label(row0, text="mx:").pack(side=tk.LEFT, padx=2)
        self.ent_mx = ttk.Entry(row0, width=6)
        self.ent_mx.insert(0, "37")
        self.ent_mx.pack(side=tk.LEFT, padx=2)

        ttk.Label(row0, text="my:").pack(side=tk.LEFT, padx=2)
        self.ent_my = ttk.Entry(row0, width=6)
        self.ent_my.insert(0, "49")
        self.ent_my.pack(side=tk.LEFT, padx=2)

        ttk.Label(row0, text="σx:").pack(side=tk.LEFT, padx=2)
        self.ent_sx = ttk.Entry(row0, width=5)
        self.ent_sx.insert(0, "5")
        self.ent_sx.pack(side=tk.LEFT, padx=2)

        ttk.Label(row0, text="σy:").pack(side=tk.LEFT, padx=2)
        self.ent_sy = ttk.Entry(row0, width=5)
        self.ent_sy.insert(0, "6")
        self.ent_sy.pack(side=tk.LEFT, padx=2)

        ttk.Label(row0, text="rxy:").pack(side=tk.LEFT, padx=2)
        self.ent_r = ttk.Entry(row0, width=6)
        self.ent_r.insert(0, "0.45")
        self.ent_r.pack(side=tk.LEFT, padx=2)

        ttk.Label(row0, text="N:").pack(side=tk.LEFT, padx=2)
        self.ent_n = ttk.Entry(row0, width=6)
        self.ent_n.insert(0, "300")
        self.ent_n.pack(side=tk.LEFT, padx=2)

        ttk.Label(row0, text="Seed:").pack(side=tk.LEFT, padx=2)
        self.ent_seed = ttk.Entry(row0, width=6)
        self.ent_seed.insert(0, "100")
        self.ent_seed.pack(side=tk.LEFT, padx=2)

        ttk.Button(row0, text="🎲 Згенерувати вибірку", command=self.handle_generate).pack(side=tk.LEFT, padx=6)
        ttk.Button(row0, text="📂 Завантажити з файлу...", command=self.handle_load_file).pack(side=tk.LEFT, padx=4)
        ttk.Button(row0, text="💾 Зберегти дані...", command=self.handle_save_file).pack(side=tk.LEFT, padx=4)

        # 2. Панель динамічних налаштувань аналізу (γ, k, правило 3σ)
        cfg_frame = ttk.Frame(top_frame)
        cfg_frame.pack(fill=tk.X, padx=4, pady=2)

        ttk.Label(cfg_frame, text="Рівень довіри γ:").pack(side=tk.LEFT, padx=3)
        self.ent_gamma = ttk.Entry(cfg_frame, width=6)
        self.ent_gamma.insert(0, "0.95")
        self.ent_gamma.pack(side=tk.LEFT, padx=2)

        ttk.Label(cfg_frame, text="Класів сітки k:").pack(side=tk.LEFT, padx=5)
        self.ent_k = ttk.Entry(cfg_frame, width=5)
        self.ent_k.insert(0, "7")
        self.ent_k.pack(side=tk.LEFT, padx=2)

        self.var_filter_3sigma = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            cfg_frame, text="Фільтр аномалій 3σ (|x - x̄| > 3Sx)",
            variable=self.var_filter_3sigma, command=self.update_analysis
        ).pack(side=tk.LEFT, padx=8)

        ttk.Button(cfg_frame, text="⚡ Оновити аналіз (без перегенерації)", command=self.update_analysis).pack(side=tk.LEFT, padx=6)

        self.lbl_status = ttk.Label(cfg_frame, text="Готово", foreground="darkgreen", font=("Arial", 9, "bold"))
        self.lbl_status.pack(side=tk.LEFT, padx=10)

        # 3. Головна розділена область: Ліва (Таблиці та гіпотези) | Права (Графіки)
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        left_pane = ttk.Frame(main_paned, width=530)
        right_pane = ttk.Frame(main_paned, width=930)
        main_paned.add(left_pane, weight=1)
        main_paned.add(right_pane, weight=2)

        # --- ЛІВА ПАНЕЛЬ ---
        # Блок 1: Точкові та інтервальні оцінки (ЛР 3 + ДІ для r за Фішером)
        tbl_frame = ttk.LabelFrame(left_pane, text=" Первинні характеристики та довірчі інтервали (ЛР 3 + ДІ Фішера для r) ")
        tbl_frame.pack(fill=tk.X, padx=3, pady=2)

        cols = ("param", "lower", "point", "upper", "covered")
        self.tree_stats = ttk.Treeview(tbl_frame, columns=cols, show='headings', height=6)
        self.tree_stats.heading("param", text="Параметр")
        self.tree_stats.heading("lower", text="Нижня межа")
        self.tree_stats.heading("point", text="Оцінка θ̂")
        self.tree_stats.heading("upper", text="Верхня межа")
        self.tree_stats.heading("covered", text="Накриття")

        self.tree_stats.column("param", width=105, anchor=tk.W)
        self.tree_stats.column("lower", width=80, anchor=tk.CENTER)
        self.tree_stats.column("point", width=110, anchor=tk.CENTER)
        self.tree_stats.column("upper", width=80, anchor=tk.CENTER)
        self.tree_stats.column("covered", width=105, anchor=tk.CENTER)
        self.tree_stats.pack(fill=tk.X, padx=3, pady=2)

        # Блок 2: Оцінювання сили та значущості зв'язку за Пірсоном (r) і Спірменом (rho) (ЛР 4, п. 4.1)
        corr_frame = ttk.LabelFrame(left_pane, text=" Оцінювання зв'язку: коефіцієнти Пірсона (r) та Спірмена (ρ) (ЛР 4, п. 4.1) ")
        corr_frame.pack(fill=tk.X, padx=3, pady=2)
        self.lbl_corr_test = ttk.Label(corr_frame, text="Очікування даних...", font=("Consolas", 8), justify=tk.LEFT)
        self.lbl_corr_test.pack(anchor=tk.W, padx=4, pady=2)

        # Блок 3: Таблиця «Лінійна регресія» (ЛР 4, п. 4.2, 4.3, 6)
        reg_frame = ttk.LabelFrame(left_pane, text=" Лінійна регресія Ŷ = â + b̂·X (МНК, довірчі інтервали, t-тест, R²) ")
        reg_frame.pack(fill=tk.X, padx=3, pady=2)

        reg_cols = ("param", "est", "se", "ci", "t_stat", "p_val", "sig")
        self.tree_reg = ttk.Treeview(reg_frame, columns=reg_cols, show='headings', height=2)
        self.tree_reg.heading("param", text="Параметр")
        self.tree_reg.heading("est", text="Оцінка")
        self.tree_reg.heading("se", text="СКВ S")
        self.tree_reg.heading("ci", text="Довірчий інтервал")
        self.tree_reg.heading("t_stat", text="t-стат.")
        self.tree_reg.heading("p_val", text="p-value")
        self.tree_reg.heading("sig", text="Значущість")

        self.tree_reg.column("param", width=75, anchor=tk.W)
        self.tree_reg.column("est", width=65, anchor=tk.CENTER)
        self.tree_reg.column("se", width=60, anchor=tk.CENTER)
        self.tree_reg.column("ci", width=135, anchor=tk.CENTER)
        self.tree_reg.column("t_stat", width=60, anchor=tk.CENTER)
        self.tree_reg.column("p_val", width=70, anchor=tk.CENTER)
        self.tree_reg.column("sig", width=85, anchor=tk.CENTER)
        self.tree_reg.pack(fill=tk.X, padx=3, pady=2)

        self.lbl_reg_summary = ttk.Label(reg_frame, text="Рівняння регресії та R²...", font=("Consolas", 8), justify=tk.LEFT)
        self.lbl_reg_summary.pack(anchor=tk.W, padx=4, pady=2)

        # Блок 4: Критерій Хі-квадрат Пірсона (ЛР 3, п. 1.6)
        chi_frame = ttk.LabelFrame(left_pane, text=" Критерій узгодженості χ² Пірсона (ЛР 3, п. 1.6) ")
        chi_frame.pack(fill=tk.X, padx=3, pady=2)
        self.lbl_chi_test = ttk.Label(chi_frame, text="Очікування даних...", font=("Consolas", 8), justify=tk.LEFT)
        self.lbl_chi_test.pack(anchor=tk.W, padx=4, pady=2)

        # Блок 5: Таблиця перегляду вибірки
        self.data_preview_frame = ttk.LabelFrame(left_pane, text=" Перегляд даних вибірки (усі пари xi, yi) ")
        self.data_preview_frame.pack(fill=tk.BOTH, expand=True, padx=3, pady=2)

        d_cols = ("idx", "x_val", "y_val")
        self.tree_data = ttk.Treeview(self.data_preview_frame, columns=d_cols, show='headings', height=6)
        self.tree_data.heading("idx", text="№")
        self.tree_data.heading("x_val", text="X (xi)")
        self.tree_data.heading("y_val", text="Y (yi)")
        self.tree_data.column("idx", width=45, anchor=tk.CENTER)
        self.tree_data.column("x_val", width=130, anchor=tk.CENTER)
        self.tree_data.column("y_val", width=130, anchor=tk.CENTER)

        d_scroll = ttk.Scrollbar(self.data_preview_frame, orient=tk.VERTICAL, command=self.tree_data.yview)
        self.tree_data.configure(yscrollcommand=d_scroll.set)
        self.tree_data.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        d_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # --- ПРАВА ПАНЕЛЬ: Вкладки Notebook для графіків ---
        self.notebook = ttk.Notebook(right_pane)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_scatter = ttk.Frame(self.notebook)
        self.tab_heatmap = ttk.Frame(self.notebook)
        self.tab_3d = ttk.Frame(self.notebook)
        self.tab_3d_dens = ttk.Frame(self.notebook)
        self.tab_matrix = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_scatter, text="1. Кореляційне поле + Регресія")
        self.notebook.add(self.tab_heatmap, text="2. 2D-гістограма (теплова карта)")
        self.notebook.add(self.tab_3d, text="3. 3D-гістограма частот")
        self.notebook.add(self.tab_3d_dens, text="4. 3D-щільність f(x, y)")
        self.notebook.add(self.tab_matrix, text="5. Таблиці частот (n_ij та p_ij*)")

        # 1. Scatter + Ellipses + Regression Line
        self.fig_sc, self.ax_sc = plt.subplots(figsize=(6, 4.5))
        self.canvas_sc = FigureCanvasTkAgg(self.fig_sc, master=self.tab_scatter)
        self.canvas_sc.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 2. 2D Heatmap
        self.fig_hm, self.ax_hm = plt.subplots(figsize=(6, 4.5))
        self.canvas_hm = FigureCanvasTkAgg(self.fig_hm, master=self.tab_heatmap)
        self.canvas_hm.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 3. 3D Bar Plot
        self.fig_3d = plt.figure(figsize=(6, 4.5))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=self.tab_3d)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 4. 3D Density Surface f(x, y) (ЛР 4, п. 4)
        self.fig_dens = plt.figure(figsize=(6, 4.5))
        self.ax_dens = self.fig_dens.add_subplot(111, projection='3d')
        self.canvas_dens = FigureCanvasTkAgg(self.fig_dens, master=self.tab_3d_dens)
        self.canvas_dens.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 5. Текстові матриці спряженості
        self.txt_matrix = tk.Text(self.tab_matrix, font=("Consolas", 9), wrap=tk.NONE)
        m_scroll_y = ttk.Scrollbar(self.tab_matrix, orient=tk.VERTICAL, command=self.txt_matrix.yview)
        m_scroll_x = ttk.Scrollbar(self.tab_matrix, orient=tk.HORIZONTAL, command=self.txt_matrix.xview)
        self.txt_matrix.configure(yscrollcommand=m_scroll_y.set, xscrollcommand=m_scroll_x.set)
        self.txt_matrix.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        m_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        m_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

    def _validate_inputs(self):
        try:
            mx = float(self.ent_mx.get().replace(',', '.'))
            my = float(self.ent_my.get().replace(',', '.'))
            sx = float(self.ent_sx.get().replace(',', '.'))
            sy = float(self.ent_sy.get().replace(',', '.'))
            r = float(self.ent_r.get().replace(',', '.'))
            n = int(self.ent_n.get())
            seed_val = self.ent_seed.get().strip()
            seed = int(seed_val) if seed_val else None
        except ValueError:
            messagebox.showerror("Помилка", "Параметри генерації мають бути дійсними числами, а N та Seed — цілими!")
            return None

        if sx <= 0 or sy <= 0:
            messagebox.showerror("Помилка", "Середньоквадратичні відхилення σx та σy повинні бути строго > 0!")
            return None
        if not (-1.0 < r < 1.0):
            messagebox.showerror("Помилка", "Коефіцієнт кореляції rxy повинен належати інтервалу (-1; 1)!")
            return None
        if n < 10:
            messagebox.showerror("Помилка", "Обсяг вибірки N повинен бути щонайменше 10!")
            return None

        return mx, my, sx, sy, r, n, seed

    def handle_generate(self):
        params = self._validate_inputs()
        if params is None:
            return

        mx, my, sx, sy, r, n, seed = params
        try:
            self.raw_x, self.raw_y = generate_2d_normal(mx, my, sx, sy, r, n, seed=seed)
            self.is_generated = True
            self.current_filepath = "Згенерована вибірка"
            self.update_analysis()
        except Exception as e:
            messagebox.showerror("Помилка генерації", str(e))

    def handle_load_file(self):
        fpath = filedialog.askopenfilename(filetypes=[("Data files", "*.csv *.txt *.xlsx *.xls")])
        if not fpath:
            return
        try:
            x, y = load_2d_data(fpath)
            self.raw_x, self.raw_y = x, y
            self.is_generated = False
            self.current_filepath = fpath
            self.ent_n.delete(0, tk.END)
            self.ent_n.insert(0, str(len(x)))
            self.update_analysis()
        except Exception as e:
            messagebox.showerror("Помилка читання файлу", str(e))

    def handle_save_file(self):
        if self.x is None or self.y is None:
            messagebox.showwarning("Увага", "Немає даних для збереження!")
            return
        fpath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV file", "*.csv"), ("Text file", "*.txt"), ("Excel", "*.xlsx")]
        )
        if not fpath:
            return
        try:
            save_2d_data(fpath, self.x, self.y)
            messagebox.showinfo("Успіх", f"Дані успішно збережено у:\n{fpath}")
        except Exception as e:
            messagebox.showerror("Помилка збереження", str(e))

    @staticmethod
    def _describe_corr(val: float):
        direction = "прямий (+)" if val > 0 else ("зворотний (-)" if val < 0 else "відсутній")
        abs_v = abs(val)
        if abs_v < 0.3:
            strength = "слабкий"
        elif abs_v < 0.5:
            strength = "помірний"
        elif abs_v < 0.7:
            strength = "помітний"
        else:
            strength = "сильний (тісний)"
        return direction, strength

    def update_analysis(self):
        if self.raw_x is None or self.raw_y is None:
            return

        try:
            gamma = float(self.ent_gamma.get().replace(',', '.'))
            if not (0.0 < gamma < 1.0):
                messagebox.showerror("Помилка", "Рівень довіри γ повинен бути в межах (0; 1)!")
                return
        except ValueError:
            messagebox.showerror("Помилка", "Рівень довіри γ має бути дійсним числом!")
            return

        try:
            k = int(self.ent_k.get())
            if k < 3 or k > 25:
                messagebox.showerror("Помилка", "Кількість класів k має бути цілим числом від 3 до 25!")
                return
        except ValueError:
            messagebox.showerror("Помилка", "Кількість класів k має бути цілим числом!")
            return

        # 0. Попередня обробка за правилом 3-сигма (ЛР 4, п. 3)
        if self.var_filter_3sigma.get():
            self.x, self.y, removed_cnt = filter_outliers_3sigma(self.raw_x, self.raw_y, check_y=False)
            self.lbl_status.config(
                text=f"N_поч={len(self.raw_x)} → N={len(self.x)} (відкинуто за 3σ: {removed_cnt})",
                foreground="darkgreen"
            )
        else:
            self.x, self.y = self.raw_x.copy(), self.raw_y.copy()
            self.lbl_status.config(
                text=f"N={len(self.x)} (фільтр 3σ вимкнено)",
                foreground="blue"
            )

        alpha = 1.0 - gamma

        # 1. Розрахунок усіх статистик ЛР 3 та ЛР 4
        pts = compute_point_estimates(self.x, self.y)
        ci = compute_confidence_intervals(pts, gamma=gamma)
        t_pearson = test_correlation_significance(pts["r_xy"], pts["N"], alpha=alpha)
        spearman = compute_spearman_correlation(self.x, self.y, alpha=alpha)
        reg = compute_linear_regression(self.x, self.y, gamma=gamma)
        chi_res = test_chi2_goodness_of_fit(self.x, self.y, k=k, alpha=alpha)

        # Теоретичні параметри для перевірки накриття
        th_mx = th_my = th_s2x = th_s2y = th_r = None
        if self.is_generated:
            try:
                th_mx = float(self.ent_mx.get().replace(',', '.'))
                th_my = float(self.ent_my.get().replace(',', '.'))
                th_s2x = float(self.ent_sx.get().replace(',', '.')) ** 2
                th_s2y = float(self.ent_sy.get().replace(',', '.')) ** 2
                th_r = float(self.ent_r.get().replace(',', '.'))
            except Exception:
                pass

        # 2. Оновлення таблиці характеристик
        for row in self.tree_stats.get_children():
            self.tree_stats.delete(row)

        def check_cov(val, low, high):
            if val is None:
                return "— (з файлу)"
            return f"ТАК ({val:g})" if (low <= val <= high) else f"НІ ({val:g})"

        self.tree_stats.insert("", tk.END, values=("mx (середнє X)", f"{ci['mx_ci'][0]:.4f}", f"{pts['x_bar']:.4f}", f"{ci['mx_ci'][1]:.4f}", check_cov(th_mx, ci['mx_ci'][0], ci['mx_ci'][1])))
        self.tree_stats.insert("", tk.END, values=("my (середнє Y)", f"{ci['my_ci'][0]:.4f}", f"{pts['y_bar']:.4f}", f"{ci['my_ci'][1]:.4f}", check_cov(th_my, ci['my_ci'][0], ci['my_ci'][1])))
        self.tree_stats.insert("", tk.END, values=("σx² (дисп. X)", f"{ci['s2x_ci'][0]:.4f}", f"{pts['s2_x']:.4f} (Sx={pts['s_x']:.2f})", f"{ci['s2x_ci'][1]:.4f}", check_cov(th_s2x, ci['s2x_ci'][0], ci['s2x_ci'][1])))
        self.tree_stats.insert("", tk.END, values=("σy² (дисп. Y)", f"{ci['s2y_ci'][0]:.4f}", f"{pts['s2_y']:.4f} (Sy={pts['s_y']:.2f})", f"{ci['s2y_ci'][1]:.4f}", check_cov(th_s2y, ci['s2y_ci'][0], ci['s2y_ci'][1])))
        self.tree_stats.insert("", tk.END, values=("Kxy (коваріація)", "—", f"{pts['k_xy']:.4f}", "—", "—"))
        self.tree_stats.insert("", tk.END, values=("rxy (Пірсон, Фішер)", f"{ci['r_ci'][0]:.4f}", f"{pts['r_xy']:.4f}", f"{ci['r_ci'][1]:.4f}", check_cov(th_r, ci['r_ci'][0], ci['r_ci'][1])))

        # 3. Блок кореляції Пірсона (r) та Спірмена (rho) (ЛР 4, п. 4.1)
        r_dir, r_str = self._describe_corr(pts["r_xy"])
        rho_dir, rho_str = self._describe_corr(spearman["rho"])
        r_sig_txt = "ЗНАЧУЩИЙ" if t_pearson["is_significant"] else "незначущий"
        rho_sig_txt = "ЗНАЧУЩИЙ" if spearman["is_significant"] else "незначущий"

        corr_info = (
            f"• Пірсон  r = {pts['r_xy']:.4f}, ДІ Фішера: [{ci['r_ci'][0]:.4f}; {ci['r_ci'][1]:.4f}] (z={ci['z_fisher']:.4f})\n"
            f"  t_r = {t_pearson['t_calc']:.4f} (t_кр = ±{t_pearson['t_crit']:.4f}, p = {t_pearson['p_value']:.2e}) → {r_dir}, {r_str}, {r_sig_txt}\n"
            f"• Спірмен ρ = {spearman['rho']:.4f} (рангова кореляція монотонного зв'язку)\n"
            f"  t_ρ = {spearman['t_calc']:.4f} (t_кр = ±{spearman['t_crit']:.4f}, p = {spearman['p_value']:.2e}) → {rho_dir}, {rho_str}, {rho_sig_txt}"
        )
        self.lbl_corr_test.config(text=corr_info)

        # 4. Таблиця «Лінійна регресія» (ЛР 4, п. 4.2, 4.3, 6)
        for row in self.tree_reg.get_children():
            self.tree_reg.delete(row)

        sig_a_str = "Значущий" if reg["sig_a"] else "Незначущий"
        sig_b_str = "Значущий" if reg["sig_b"] else "Незначущий"

        self.tree_reg.insert("", tk.END, values=(
            "a (вільний)", f"{reg['a_hat']:.4f}", f"{reg['s_a']:.4f}",
            f"[{reg['a_ci'][0]:.4f}; {reg['a_ci'][1]:.4f}]",
            f"{reg['t_a']:.4f}", f"{reg['p_val_a']:.3e}", sig_a_str
        ))
        self.tree_reg.insert("", tk.END, values=(
            "b (нахил X)", f"{reg['b_hat']:.4f}", f"{reg['s_b']:.4f}",
            f"[{reg['b_ci'][0]:.4f}; {reg['b_ci'][1]:.4f}]",
            f"{reg['t_b']:.4f}", f"{reg['p_val_b']:.3e}", sig_b_str
        ))

        sign_b = "+" if reg["b_hat"] >= 0 else "-"
        reg_summary = (
            f"• Рівняння регресії: Ŷ = {reg['a_hat']:.4f} {sign_b} {abs(reg['b_hat']):.4f}·X "
            f"(S_зал² = {reg['s2_res']:.4f}, S_зал = {reg['s_res']:.4f}, t_кр = ±{reg['t_crit']:.4f}, ν = {reg['nu']})\n"
            f"• Коефіцієнт детермінації: R² = {reg['r2']:.4f} ({reg['r2']*100:.2f}% варіації Y пояснюється лінійною регресією на X)"
        )
        self.lbl_reg_summary.config(text=reg_summary)

        # 5. Блок критерію Хі-квадрат
        chi_verdict = "ДОСТОВІРНЕ (χ² ≤ χ²_кр)" if chi_res["is_valid"] else "ВІДХИЛЯЄТЬСЯ (χ² > χ²_кр)"
        chi_info = (
            f"• χ² = {chi_res['chi2_calc']:.4f} | χ²_кр(α={alpha:.2f}; ν={chi_res['df']}) = {chi_res['chi2_crit']:.4f} "
            f"(класів: {chi_res['k_initial']}→{chi_res['k_pooled']}) | p={chi_res['p_value']:.4f} → {chi_verdict}"
        )
        self.lbl_chi_test.config(text=chi_info)

        # 6. Таблиця даних вибірки
        self.data_preview_frame.config(text=f" Перегляд даних вибірки (усього N = {len(self.x)} пар) ")
        for row in self.tree_data.get_children():
            self.tree_data.delete(row)
        for i in range(len(self.x)):
            self.tree_data.insert("", tk.END, values=(i + 1, f"{self.x[i]:.4f}", f"{self.y[i]:.4f}"))

        # 7. Оновлення графіків
        plot_scatter_and_ellipses(self.x, self.y, gamma=gamma, ax=self.ax_sc)
        self.fig_sc.tight_layout()
        self.canvas_sc.draw()

        plot_2d_heatmap(self.x, self.y, k=k, ax=self.ax_hm)
        self.fig_hm.tight_layout()
        self.canvas_hm.draw()

        plot_3d_histogram(self.x, self.y, k=k, ax=self.ax_3d)
        self.fig_3d.tight_layout()
        self.canvas_3d.draw()

        plot_3d_density_surface(self.x, self.y, ax=self.ax_dens)
        self.fig_dens.tight_layout()
        self.canvas_dens.draw()

        # 8. Текстові таблиці частот
        grp = compute_2d_grouping(self.x, self.y, k=k)
        self.txt_matrix.delete("1.0", tk.END)
        m_lines = [
            f"Крокові інтервали: hx = {grp['hx']:.4f}, hy = {grp['hy']:.4f} (k = {k}, N = {len(self.x)})\n",
            f"1) ТАБЛИЦЯ АБСОЛЮТНИХ ЧАСТОТ (n_ij):"
        ]
        header = "  Класи Y \\ X  |" + "".join([f" [{grp['x_edges'][i]:6.2f};{grp['x_edges'][i+1]:6.2f}) |" for i in range(k)]) + " Сума n_y"
        m_lines.append(header)
        m_lines.append("-" * len(header))

        for j in reversed(range(k)):
            row_str = f" [{grp['y_edges'][j]:6.2f};{grp['y_edges'][j+1]:6.2f}) |"
            row_sum = 0
            for i in range(k):
                cnt = int(grp["counts"][i, j])
                row_sum += cnt
                row_str += f" {cnt:^15} |"
            row_str += f" {row_sum:^8}"
            m_lines.append(row_str)

        m_lines.append("-" * len(header))
        col_sum_str = "   Сума n_x    |"
        for i in range(k):
            c_sum = int(np.sum(grp["counts"][i, :]))
            col_sum_str += f" {c_sum:^15} |"
        col_sum_str += f" {len(self.x):^8}\n"
        m_lines.append(col_sum_str)

        m_lines.append(f"2) ТАБЛИЦЯ ВІДНОСНИХ ЧАСТОТ (p_ij* = n_ij / N):")
        header_rel = "  Класи Y \\ X  |" + "".join([f" [{grp['x_edges'][i]:6.2f};{grp['x_edges'][i+1]:6.2f}) |" for i in range(k)]) + " Сума p_y*"
        m_lines.append(header_rel)
        m_lines.append("-" * len(header_rel))

        for j in reversed(range(k)):
            row_str = f" [{grp['y_edges'][j]:6.2f};{grp['y_edges'][j+1]:6.2f}) |"
            row_sum_rel = 0.0
            for i in range(k):
                rf = grp["rel_freqs"][i, j]
                row_sum_rel += rf
                row_str += f" {rf:^15.4f} |"
            row_str += f" {row_sum_rel:^8.4f}"
            m_lines.append(row_str)

        m_lines.append("-" * len(header_rel))
        col_sum_rel_str = "   Сума p_x*   |"
        for i in range(k):
            c_sum_rel = float(np.sum(grp["rel_freqs"][i, :]))
            col_sum_rel_str += f" {c_sum_rel:^15.4f} |"
        col_sum_rel_str += f" {float(np.sum(grp['rel_freqs'])):^8.4f}"
        m_lines.append(col_sum_rel_str)

        self.txt_matrix.insert(tk.END, "\n".join(m_lines))


def main():
    root = tk.Tk()
    app = Lab3App(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (plt.close('all'), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()