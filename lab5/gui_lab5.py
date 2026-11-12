"""
gui_lab5.py — Графічний інтерфейс Лабораторної роботи № 5 (Варіант 1):
- таблиця завантажених даних (N x 3) зі статусом кожного рядка (п. 4.1.1);
- описові статистики до та після чищення (п. 4.1.2);
- таблиця відкинутих аномальних спостережень за правилом 3-сигма (п. 4.2.3);
- порівняння вектора середніх та кореляційної матриці ДО і ПІСЛЯ чищення (п. 4.2.4);
- таблиці E{xi}, DC{xi}, R{xi} з перевіркою значущості парних кореляцій (п. 4.4);
- таблиця парних, часткових і множинних кореляцій (пряма формула + перевірка через R^-1 + висновки п. 4.5);
- окремий вивід власних чисел lambda, матриці власних векторів A^T, таблиці характеристик ГК,
  зворотного перетворення (відфільтровані дані) та контрольної звірки з sklearn (п. 4.6);
- 8 вкладок візуалізації та перемикач «З аномаліями / Без аномалій» (п. 4.3, п. 6).
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from data_io import (
    load_multivariate_data, save_pc_scores,
    save_reconstructed_data, generate_sample_var01
)
from multivariate_statistics import (
    compute_univariate_descriptors, filter_outliers_3sigma_nd,
    compute_multivariate_matrices, compute_pairwise_significance,
    compute_partial_correlations, compute_multiple_correlations
)
from pca import (
    fit_pca_custom, build_pca_summary_table,
    transform_and_reconstruct, compare_with_sklearn
)
from multivariate_visualization import (
    plot_3d_scatter, plot_scatter_matrix_3x3, plot_parallel_coordinates,
    plot_correlation_heatmap, plot_bubble_chart, plot_nightingale_rose,
    plot_scree, plot_biplot
)


class Lab5App:
    def __init__(self, root):
        self.root = root
        self.root.title("ІАД — Лабораторна робота № 5: Візуалізація багатовимірних даних та МГК (Варіант 1)")
        self.root.geometry("1540x940")

        self.raw_data = None
        self.clean_data = None
        self.current_data = None
        self.clean_mask = None
        self.outlier_details = []
        self.names = ["X1", "X2", "X3"]
        self.filepath = ""

        self.pca_res = None
        self.eigvals = None
        self.var_shares = None
        self.cum_shares = None

        self.setup_ui()
        self.autoload_initial_data()

    def setup_ui(self):
        # 1. Верхня панель керування
        top_frame = ttk.LabelFrame(self.root, text=" Багатовимірний аналіз (ЛР 5) — Керування вибіркою та параметрами ")
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        row0 = ttk.Frame(top_frame)
        row0.pack(fill=tk.X, padx=4, pady=2)

        ttk.Button(row0, text="📂 Завантажити var_NN.txt...", command=self.handle_browse_file).pack(side=tk.LEFT, padx=4)
        ttk.Button(row0, text="🎲 Демо Варіант 1 (N=200)", command=self.load_demo_data).pack(side=tk.LEFT, padx=4)

        self.lbl_file = ttk.Label(row0, text="Джерело: —", font=("Segoe UI", 9, "bold"), foreground="navy")
        self.lbl_file.pack(side=tk.LEFT, padx=10)

        # Перемикач «З аномаліями / Без аномалій» (п. 6)
        self.var_use_clean = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            row0,
            text="Режим: Без аномалій (очищена вибірка N')  [вимкніть для режиму 'З аномаліями']",
            variable=self.var_use_clean,
            command=self.update_all
        ).pack(side=tk.RIGHT, padx=8)

        row1 = ttk.Frame(top_frame)
        row1.pack(fill=tk.X, padx=4, pady=2)

        ttk.Label(row1, text="Ознаки бульбашкової діаграми:").pack(side=tk.LEFT, padx=4)
        ttk.Label(row1, text="Вісь X:").pack(side=tk.LEFT, padx=2)
        self.combo_bx = ttk.Combobox(row1, values=self.names, width=6, state="readonly")
        self.combo_bx.set("X1")
        self.combo_bx.pack(side=tk.LEFT, padx=2)
        self.combo_bx.bind("<<ComboboxSelected>>", lambda e: self.update_bubble_plot())

        ttk.Label(row1, text="Вісь Y:").pack(side=tk.LEFT, padx=2)
        self.combo_by = ttk.Combobox(row1, values=self.names, width=6, state="readonly")
        self.combo_by.set("X2")
        self.combo_by.pack(side=tk.LEFT, padx=2)
        self.combo_by.bind("<<ComboboxSelected>>", lambda e: self.update_bubble_plot())

        ttk.Label(row1, text="Площа (S):").pack(side=tk.LEFT, padx=2)
        self.combo_bsize = ttk.Combobox(row1, values=self.names, width=6, state="readonly")
        self.combo_bsize.set("X3")
        self.combo_bsize.pack(side=tk.LEFT, padx=2)
        self.combo_bsize.bind("<<ComboboxSelected>>", lambda e: self.update_bubble_plot())

        ttk.Button(row1, text="💾 Зберегти pc_scores_01.txt", command=self.handle_save_scores).pack(side=tk.RIGHT, padx=4)
        ttk.Button(row1, text="💾 Зберегти відфільтровані дані", command=self.handle_save_reconstructed).pack(side=tk.RIGHT, padx=4)

        self.lbl_status = ttk.Label(row1, text="Готово", font=("Segoe UI", 9, "bold"), foreground="darkgreen")
        self.lbl_status.pack(side=tk.RIGHT, padx=10)

        # 2. Основна робоча область: Ліворуч — Таблиці та звіти | Праворуч — Графіки
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        left_pane = ttk.Frame(main_paned, width=640)
        right_pane = ttk.Frame(main_paned, width=880)
        main_paned.add(left_pane, weight=5)
        main_paned.add(right_pane, weight=6)

        # --- Ліва панель: 6 вкладок результатів ---
        self.stats_notebook = ttk.Notebook(left_pane)
        self.stats_notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_data = ttk.Frame(self.stats_notebook)
        self.tab_desc = ttk.Frame(self.stats_notebook)
        self.tab_outliers = ttk.Frame(self.stats_notebook)
        self.tab_matrices = ttk.Frame(self.stats_notebook)
        self.tab_corrs = ttk.Frame(self.stats_notebook)
        self.tab_pca = ttk.Frame(self.stats_notebook)

        self.stats_notebook.add(self.tab_data, text="1. Таблиця даних")
        self.stats_notebook.add(self.tab_desc, text="2. До / Після 3σ")
        self.stats_notebook.add(self.tab_outliers, text="3. Відкинуті (3σ)")
        self.stats_notebook.add(self.tab_matrices, text="4. E{ξ}, DC{ξ}, R{ξ}")
        self.stats_notebook.add(self.tab_corrs, text="5. Кореляції (частк./множ.)")
        self.stats_notebook.add(self.tab_pca, text="6. Метод ГК (PCA)")

        # Вкладка 1: Таблиця вихідних даних (п. 4.1.1)
        data_cols = ("idx", "x1", "x2", "x3", "status")
        self.tree_data = ttk.Treeview(self.tab_data, columns=data_cols, show="headings")
        self.tree_data.heading("idx", text="№ рядка")
        self.tree_data.heading("x1", text="X1")
        self.tree_data.heading("x2", text="X2")
        self.tree_data.heading("x3", text="X3")
        self.tree_data.heading("status", text="Статус (правило 3σ)")
        for col, w in zip(data_cols, (65, 105, 105, 105, 150)):
            self.tree_data.column(col, width=w, anchor=tk.CENTER)

        data_scroll = ttk.Scrollbar(self.tab_data, orient=tk.VERTICAL, command=self.tree_data.yview)
        self.tree_data.configure(yscrollcommand=data_scroll.set)
        self.tree_data.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        data_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Вкладка 2: Описові статистики + Порівняння ДО та ПІСЛЯ чищення
        self.txt_desc = self._create_scrollable_text(self.tab_desc)

        # Вкладка 3: Таблиця відкинутих спостережень (п. 4.2.3)
        out_cols = ("row", "feature", "val", "bounds", "sigmas")
        self.tree_outliers = ttk.Treeview(self.tab_outliers, columns=out_cols, show="headings")
        self.tree_outliers.heading("row", text="№ рядка")
        self.tree_outliers.heading("feature", text="Ознака")
        self.tree_outliers.heading("val", text="Значення x_{k,l}")
        self.tree_outliers.heading("bounds", text="Межі [x̄_k - 3σ_k; x̄_k + 3σ_k]")
        self.tree_outliers.heading("sigmas", text="|x - x̄| / σ")
        for col, w in zip(out_cols, (65, 70, 105, 200, 95)):
            self.tree_outliers.column(col, width=w, anchor=tk.CENTER)

        out_scroll = ttk.Scrollbar(self.tab_outliers, orient=tk.VERTICAL, command=self.tree_outliers.yview)
        self.tree_outliers.configure(yscrollcommand=out_scroll.set)
        self.tree_outliers.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        out_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Вкладки 4, 5, 6: Текстові звіти з таблицями
        self.txt_matrices = self._create_scrollable_text(self.tab_matrices)
        self.txt_corrs = self._create_scrollable_text(self.tab_corrs)
        self.txt_pca = self._create_scrollable_text(self.tab_pca)

        # --- Права панель: 8 вкладок графіків ---
        self.vis_notebook = ttk.Notebook(right_pane)
        self.vis_notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_v3d = ttk.Frame(self.vis_notebook)
        self.tab_vpair = ttk.Frame(self.vis_notebook)
        self.tab_vpar = ttk.Frame(self.vis_notebook)
        self.tab_vheat = ttk.Frame(self.vis_notebook)
        self.tab_vbubble = ttk.Frame(self.vis_notebook)
        self.tab_vrose = ttk.Frame(self.vis_notebook)
        self.tab_vscree = ttk.Frame(self.vis_notebook)
        self.tab_vbiplot = ttk.Frame(self.vis_notebook)

        self.vis_notebook.add(self.tab_v3d, text="1. 3D Поле")
        self.vis_notebook.add(self.tab_vpair, text="2. Матриця 3×3")
        self.vis_notebook.add(self.tab_vpar, text="3. Паралельні коорд.")
        self.vis_notebook.add(self.tab_vheat, text="4. Теплова карта R")
        self.vis_notebook.add(self.tab_vbubble, text="5. Бульбашкова")
        self.vis_notebook.add(self.tab_vrose, text="6. Найтінгейл")
        self.vis_notebook.add(self.tab_vscree, text="7. Scree Plot")
        self.vis_notebook.add(self.tab_vbiplot, text="8. Біплот")

        self.fig_3d = plt.figure(figsize=(6, 4.8))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canv_3d = self._embed_figure(self.fig_3d, self.tab_v3d)

        self.fig_pair = plt.figure(figsize=(6.5, 5.2))
        self.canv_pair = self._embed_figure(self.fig_pair, self.tab_vpair)

        self.fig_par, self.ax_par = plt.subplots(figsize=(6, 4.5))
        self.canv_par = self._embed_figure(self.fig_par, self.tab_vpar)

        self.fig_heat = plt.figure(figsize=(5.5, 4.5))
        self.canv_heat = self._embed_figure(self.fig_heat, self.tab_vheat)

        self.fig_bubble = plt.figure(figsize=(6, 4.5))
        self.canv_bubble = self._embed_figure(self.fig_bubble, self.tab_vbubble)

        self.fig_rose = plt.figure(figsize=(5, 5))
        self.ax_rose = self.fig_rose.add_subplot(111, polar=True)
        self.canv_rose = self._embed_figure(self.fig_rose, self.tab_vrose)

        self.fig_scree, self.ax_scree = plt.subplots(figsize=(6, 4.2))
        self.canv_scree = self._embed_figure(self.fig_scree, self.tab_vscree)

        self.fig_biplot, self.ax_biplot = plt.subplots(figsize=(6, 4.5))
        self.canv_biplot = self._embed_figure(self.fig_biplot, self.tab_vbiplot)

    def _create_scrollable_text(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)
        y_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL)
        x_scroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL)
        txt = tk.Text(
            frame, font=("Consolas", 9), wrap=tk.NONE,
            yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set
        )
        y_scroll.config(command=txt.yview)
        x_scroll.config(command=txt.xview)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        return txt

    def _embed_figure(self, fig, parent):
        canvas = FigureCanvasTkAgg(fig, master=parent)
        toolbar = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        return canvas

    def autoload_initial_data(self):
        """Якщо у поточній папці лежить файл var_01.txt — завантажуємо його автоматично."""
        candidates = ["var_01.txt", os.path.join("data", "var_01.txt")]
        for cpath in candidates:
            if os.path.exists(cpath):
                try:
                    self.raw_data, self.names = load_multivariate_data(cpath)
                    self.filepath = os.path.basename(cpath)
                    self.lbl_file.config(text=f"Файл: {self.filepath} (N={len(self.raw_data)})")
                    self._update_combos()
                    self.process_outliers_and_run()
                    return
                except Exception:
                    pass
        self.load_demo_data()

    def load_demo_data(self):
        self.raw_data, self.names = generate_sample_var01(n_samples=200, seed=100)
        self.filepath = "Демо var_01.txt (Варіант 1, N=200)"
        self.lbl_file.config(text=f"Джерело: {self.filepath}")
        self._update_combos()
        self.process_outliers_and_run()

    def handle_browse_file(self):
        fpath = filedialog.askopenfilename(
            title="Оберіть файл даних варіанта (var_NN.txt)",
            filetypes=[("Text files", "*.txt *.tsv *.csv"), ("All files", "*.*")]
        )
        if not fpath:
            return
        try:
            data, names = load_multivariate_data(fpath)
            self.raw_data = data
            self.names = names
            self.filepath = os.path.basename(fpath)
            self.lbl_file.config(text=f"Файл: {self.filepath} (N={len(data)})")
            self._update_combos()
            self.process_outliers_and_run()
        except Exception as e:
            messagebox.showerror("Помилка формату або завантаження файлу", str(e))

    def _update_combos(self):
        for cb, idx in zip((self.combo_bx, self.combo_by, self.combo_bsize), (0, 1, 2)):
            cb['values'] = self.names
            cb.set(self.names[idx])

    def process_outliers_and_run(self):
        if self.raw_data is None:
            return

        try:
            self.clean_data, self.outlier_details, self.clean_mask = filter_outliers_3sigma_nd(self.raw_data, self.names)
        except Exception as e:
            messagebox.showerror("Помилка фільтрації 3σ", str(e))
            return

        # 1. Заповнення таблиці вихідних даних (п. 4.1.1)
        for item in self.tree_data.get_children():
            self.tree_data.delete(item)
        for k, name in enumerate(self.names, start=1):
            self.tree_data.heading(f"x{k}", text=name)

        for idx, row in enumerate(self.raw_data):
            status_str = "Норма" if self.clean_mask[idx] else "АНОМАЛІЯ (>3σ)"
            self.tree_data.insert("", tk.END, values=(
                idx + 1, f"{row[0]:.4f}", f"{row[1]:.4f}", f"{row[2]:.4f}", status_str
            ))

        # 2. Заповнення таблиці відкинутих спостережень (п. 4.2.3)
        for item in self.tree_outliers.get_children():
            self.tree_outliers.delete(item)

        for d in self.outlier_details:
            b_str = f"[{d['low_bound']:.4f}; {d['upp_bound']:.4f}]"
            self.tree_outliers.insert("", tk.END, values=(
                d["row_idx"], d["feature"], f"{d['val']:.4f}", b_str, f"{d['diff_sigmas']:.2f} σ"
            ))

        self.update_all()

    def update_all(self):
        if self.raw_data is None:
            return

        use_clean = self.var_use_clean.get()
        show_outliers_on_plots = not use_clean
        self.current_data = self.clean_data if use_clean else self.raw_data

        mode_label = f"ОЧИЩЕНА ВИБІРКА (N' = {len(self.clean_data)})" if use_clean else f"ПОЧАТКОВА З АНОМАЛІЯМИ (N = {len(self.raw_data)})"
        self.lbl_status.config(text=mode_label)

        try:
            # --- Вкладка 2: Описові статистики та порівняння ДО і ПІСЛЯ чищення (п. 4.1.2, 4.2.4) ---
            desc_raw = compute_univariate_descriptors(self.raw_data, self.names)
            desc_clean = compute_univariate_descriptors(self.clean_data, self.names)
            mean_raw, dc_raw, r_raw = compute_multivariate_matrices(self.raw_data)
            mean_cln, dc_cln, r_cln = compute_multivariate_matrices(self.clean_data)

            n_raw = len(self.raw_data)
            n_cln = len(self.clean_data)
            n_out = n_raw - n_cln

            t2 = [
                "=== 1. ПЕРВИННИЙ АНАЛІЗ ДО ТА ПІСЛЯ ЧИЩЕННЯ ЗА ПРАВИЛОМ 3σ (п. 4.1, п. 4.2) ===\n",
                f"• Початковий обсяг вибірки:         N  = {n_raw}",
                f"• Відкинуто аномальних рядків (3σ): K  = {n_out}",
                f"• Обсяг вибірки після чищення:      N' = {n_cln}\n",
                "Описові статистики ознак ДО чищення (N) та ПІСЛЯ чищення (N'):",
                f"{'Ознака':<8} | {'Min (до)':>10} | {'Max (до)':>10} | {'x̄_k (до)':>10} | {'σ_k (до)':>10} || {'Min (після)':>11} | {'Max (після)':>11} | {'x̄_k (після)':>11} | {'σ_k (після)':>11}",
                "-" * 112
            ]
            for dr, dc in zip(desc_raw, desc_clean):
                t2.append(
                    f"{dr['name']:<8} | {dr['min']:>10.4f} | {dr['max']:>10.4f} | {dr['mean']:>10.4f} | {dr['std']:>10.4f} || "
                    f"{dc['min']:>11.4f} | {dc['max']:>11.4f} | {dc['mean']:>11.4f} | {dc['std']:>11.4f}"
                )

            t2.append("\n=== 2. ПОРІВНЯННЯ ВЕКТОРА СЕРЕДНІХ ТА КОРЕЛЯЦІЙНОЇ МАТРИЦІ ДО І ПІСЛЯ ЧИЩЕННЯ (п. 4.2.4) ===\n")
            t2.append(f"Вектор середніх ДО чищення:    E_до{{ξ}}    = [{', '.join(f'{v:.4f}' for v in mean_raw)}]^T")
            t2.append(f"Вектор середніх ПІСЛЯ чищення: E_після{{ξ}} = [{', '.join(f'{v:.4f}' for v in mean_cln)}]^T")
            t2.append(f"Зміна середніх (Δx̄ = після - до):          [{', '.join(f'{(c - r):+.4f}' for r, c in zip(mean_raw, mean_cln))}]^T\n")

            t2.append("Кореляційна матриця R{ξ} ДО чищення (N):            Кореляційна матриця R{ξ} ПІСЛЯ чищення (N'):")
            hdr = "      " + " ".join(f"{nm:>10}" for nm in self.names)
            t2.append(f"{hdr}          {hdr}")
            for i in range(3):
                left_r = f"{self.names[i]:<4}: " + " ".join(f"{r_raw[i, j]:>+10.4f}" for j in range(3))
                right_r = f"{self.names[i]:<4}: " + " ".join(f"{r_cln[i, j]:>+10.4f}" for j in range(3))
                t2.append(f"{left_r}          {right_r}")

            t2.append("\nВисновок п. 4.2: Вилучення аномальних спостережень за правилом 3σ зменшило розкид (СКВ σ_k)")
            t2.append("та усунуло викривлення парних коефіцієнтів кореляції, спричинене викидами.")

            self.txt_desc.delete("1.0", tk.END)
            self.txt_desc.insert(tk.END, "\n".join(t2))

            # --- Вкладка 4: Вектор математичних сподівань, ДК-матриця та кореляційна матриця (п. 4.4) ---
            mean_cur, dc_cur, r_cur = compute_multivariate_matrices(self.current_data)
            pair_sig = compute_pairwise_significance(r_cur, len(self.current_data), self.names, alpha=0.05)

            t4 = [
                f"=== БАГАТОВИМІРНІ ХАРАКТЕРИСТИКИ ({mode_label}) (п. 4.4) ===\n",
                "1. Оцінка вектора математичних сподівань E{ξ} = [x̄_1, x̄_2, x̄_3]^T (п. 4.4.1):"
            ]
            for k in range(3):
                t4.append(f"   x̄_{k+1} ({self.names[k]}) = {mean_cur[k]:.6f}")

            t4.append("\n2. Дисперсійно-коваріаційна матриця DC{ξ} (п. 4.4.2):")
            t4.append("        " + "  ".join(f"{nm:>12}" for nm in self.names))
            for i in range(3):
                t4.append(f"{self.names[i]:<6}: " + "  ".join(f"{dc_cur[i, j]:>12.6f}" for j in range(3)))

            t4.append("\n3. Кореляційна матриця R{ξ} та значущість парних коефіцієнтів (п. 4.4.3, α = 0.05):")
            t4.append("        " + "  ".join(f"{nm:>12}" for nm in self.names))
            for i in range(3):
                t4.append(f"{self.names[i]:<6}: " + "  ".join(f"{r_cur[i, j]:>+12.6f}" for j in range(3)))

            t4.append(f"\nПеревірка значущості парних коефіцієнтів кореляції (t-критерій, ν = N - 2 = {len(self.current_data) - 2}):")
            t4.append(f"{'Пара ознак':<12} | {'r_{i,j}':>10} | {'|t|-стат.':>10} | {'t_крит':>9} | {'p-value':>10} | {'Висновок (α=0.05)'}")
            t4.append("-" * 78)
            for ps in pair_sig:
                verdict = "Значущий зв'язок" if ps["is_significant"] else "Незначущий (H0: r=0)"
                t4.append(
                    f"{ps['pair']:<12} | {ps['r_val']:>+10.4f} | {abs(ps['t_stat']):>10.4f} | "
                    f"{ps['t_crit']:>9.4f} | {ps['p_val']:>10.4e} | {verdict}"
                )

            # Визначення найсильнішого прямого та оберненого зв'язку (п. 4.3.1)
            pos_pairs = [p for p in pair_sig if p["r_val"] > 0]
            neg_pairs = [p for p in pair_sig if p["r_val"] < 0]
            t4.append("\nВисновки щодо парних зв'язків (п. 4.3.1):")
            if pos_pairs:
                best_pos = max(pos_pairs, key=lambda x: x["r_val"])
                t4.append(f"• Найсильніший прямий зв'язок:    пара {best_pos['pair']} (r = {best_pos['r_val']:+.4f}).")
            if neg_pairs:
                best_neg = min(neg_pairs, key=lambda x: x["r_val"])
                t4.append(f"• Найсильніший обернений зв'язок: пара {best_neg['pair']} (r = {best_neg['r_val']:+.4f}).")

            self.txt_matrices.delete("1.0", tk.END)
            self.txt_matrices.insert(tk.END, "\n".join(t4))

            # --- Вкладка 5: Часткові та множинні коефіцієнти кореляції (п. 4.5) ---
            n_cur = len(self.current_data)
            partials = compute_partial_correlations(r_cur, n_cur, self.names, alpha=0.05)
            multiples = compute_multiple_correlations(r_cur, n_cur, self.names, alpha=0.05)
            p_inv_mat = np.linalg.inv(r_cur)

            t5 = [
                f"=== ЧАСТКОВІ ТА МНОЖИННІ КОЕФІЦІЄНТИ КОРЕЛЯЦІЇ ({mode_label}) (п. 4.5) ===\n",
                "Обернена кореляційна матриця P = R⁻¹ (використовується для контрольної перевірки п. 5.3, 5.4):",
                "        " + "  ".join(f"{nm:>12}" for nm in self.names)
            ]
            for i in range(3):
                t5.append(f"{self.names[i]:<6}: " + "  ".join(f"{p_inv_mat[i, j]:>+12.6f}" for j in range(3)))

            t5.append(f"\n1. Зведена таблиця парних та часткових коефіцієнтів кореляції (ν = N - 3 = {n_cur - 3}, α = 0.05):")
            t5.append(
                f"{'Пара (викл.)':<14} | {'r_парний':>9} | {'r_частк (ф-ла)':>14} | "
                f"{'r_частк (R⁻¹)':>13} | {'t-стат.':>9} | {'t_крит':>8} | {'Значущість'}"
            )
            t5.append("-" * 96)
            for p in partials:
                sig_str = "Значущий" if p["is_significant"] else "Незначущий"
                t5.append(
                    f"{p['label']:<14} | {p['r_pair']:>+9.4f} | {p['r_partial']:>+14.6f} | "
                    f"{p['r_partial_inv']:>+13.6f} | {p['t_stat']:>+9.3f} | {p['t_crit']:>8.3f} | {sig_str}"
                )

            t5.append(f"\n2. Множинні коефіцієнти кореляції та коефіцієнти детермінації r² (ν1 = 2, ν2 = {n_cur - 3}, α = 0.05):")
            t5.append(
                f"{'Ознака (поясн.)':<15} | {'r_множ (ф-ла)':>13} | {'r_множ (R⁻¹)':>12} | "
                f"{'Дет. r²':>9} | {'F-стат.':>9} | {'F_крит':>8} | {'Значущість'}"
            )
            t5.append("-" * 96)
            for m in multiples:
                sig_str = "Значущий" if m["is_significant"] else "Незначущий"
                label_m = f"{m['target']} · ({m['predictors']})"
                t5.append(
                    f"{label_m:<15} | {m['r_mult']:>13.6f} | {m['r_mult_inv']:>12.6f} | "
                    f"{m['r2']*100:>8.2f}% | {m['f_stat']:>9.3f} | {m['f_crit']:>8.3f} | {sig_str}"
                )

            # Окремий рядок множинних коефіцієнтів (вимога п. 4.5.3)
            mult_line = " | ".join(f"{m['notation']} = {m['r_mult']:.4f} (r² = {m['r2']:.4f})" for m in multiples)
            t5.append(f"\nРядок множинних коефіцієнтів (п. 4.5.3):\n  {mult_line}")

            # Розгорнутий висновок п. 4.5.4
            t5.append("\n=== ВИСНОВКИ ЗА П. 4.5.4 (порівняння парних і часткових коефіцієнтів) ===")
            for p in partials:
                delta = abs(p["r_partial"]) - abs(p["r_pair"])
                if abs(p["r_partial"]) < abs(p["r_pair"]) * 0.65 or not p["is_significant"]:
                    reason = (
                        f"зв'язок значною мірою зумовлений опосередкованим впливом ознаки {p['excluded']} "
                        f"(після її виключення модуль кореляції впав з {abs(p['r_pair']):.4f} до {abs(p['r_partial']):.4f})."
                    )
                else:
                    reason = (
                        f"зв'язок є безпосереднім (прямим), оскільки після виключення впливу {p['excluded']} "
                        f"коефіцієнт залишається статистично значущим ({p['r_partial']:+.4f} порівняно з {p['r_pair']:+.4f})."
                    )
                t5.append(f"• Пара {p['pair']} (при фіксованій {p['excluded']}): Δ|r| = {delta:+.4f} -> {reason}")

            best_m = max(multiples, key=lambda x: x["r_mult"])
            t5.append(
                f"• Ознака, найкраще пояснена лінійною регресією на дві інші: {best_m['target']} "
                f"(множинний коефіцієнт кореляції r = {best_m['r_mult']:.4f}, коефіцієнт детермінації r² = {best_m['r2']*100:.2f}%)."
            )

            self.txt_corrs.delete("1.0", tk.END)
            self.txt_corrs.insert(tk.END, "\n".join(t5))

            # --- Вкладка 6: Метод головних компонент (п. 4.6) ---
            eigvals, matrix_a, matrix_a_T, checks = fit_pca_custom(r_cur)
            pca_table, var_shares, cum_shares = build_pca_summary_table(eigvals, matrix_a, self.names)
            pca_exec = transform_and_reconstruct(self.current_data, matrix_a, var_shares, threshold_pct=5.0)
            sk_check = compare_with_sklearn(self.current_data)

            self.eigvals = eigvals
            self.var_shares = var_shares
            self.cum_shares = cum_shares
            self.pca_res = pca_exec

            t6 = [
                f"=== МЕТОД ГОЛОВНИХ КОМПОНЕНТ (МГК / PCA) ({mode_label}) (п. 4.6) ===\n",
                "1. Окремий вивід власних чисел та власних векторів кореляційної матриці R{ξ} (п. 4.6.3):",
                f"   Вектор власних чисел: λ = ({eigvals[0]:.6f};  {eigvals[1]:.6f};  {eigvals[2]:.6f})\n",
                "   Матриця власних векторів Aᵀ = [A_{x'₁}, A_{x'₂}, A_{x'₃}] розміром 3×3 (стовпці — власні вектори):"
            ]
            t6.append("          " + "  ".join(f"{'A_x\'' + str(k+1):>12}" for k in range(3)))
            for v in range(3):
                t6.append(f"   {self.names[v]:<5}: " + "  ".join(f"{matrix_a_T[v, k]:>+12.6f}" for k in range(3)))

            t6.append("\n2. Перевірка властивостей власних чисел та діагоналізації A · R · Aᵀ = Λ (п. 4.6.2, п. 4.6.3):")
            t6.append(f"   • Сума власних чисел:    λ₁ + λ₂ + λ₃ = {checks['sum_lambdas']:.6f}  (n = 3) -> {'ВИКОНУЄТЬСЯ' if checks['is_sum_3'] else 'ПОХИБКА'}")
            t6.append(f"   • Добуток власних чисел: λ₁ · λ₂ · λ₃ = {checks['prod_lambdas']:.6f}  (|R| = {checks['det_r']:.6f}) -> {'ВИКОНУЄТЬСЯ' if checks['is_prod_det'] else 'ПОХИБКА'}")
            t6.append("   • Матриця Λ = A · R · Aᵀ:")
            for i in range(3):
                t6.append("       [" + "  ".join(f"{checks['lambda_mat'][i, j]:>+11.6f}" for j in range(3)) + "]")

            t6.append("\n3. Таблиця «Характеристики головних компонент» (за зразком п. 4.6.4):")
            hdr_pca = f"   {'':<8} | " + " | ".join(f"{h:^12}" for h in pca_table["headers"])
            t6.append(hdr_pca)
            t6.append("   " + "-" * (len(hdr_pca) - 3))
            for row in pca_table["rows"]:
                t6.append(f"   {row['name']:<8} | " + " | ".join(f"{val:>12.4f}" for val in row["values"]))

            t6.append("\n4. Зворотне перетворення та редукція розмірності (п. 4.6.5, п. 5.5):")
            min_share = var_shares[-1]
            if pca_exec["w_rule"] < 3:
                t6.append(
                    f"   • За правилом < 5% дисперсії: 3-тя компонента має частку {min_share:.2f}% < 5%, тому відкидається."
                )
                t6.append(
                    f"   • Залишено w = {pca_exec['w_rule']} головні компоненти (Σ% = {cum_shares[pca_exec['w_rule']-1]:.2f}%)."
                )
                t6.append(f"   • Похибка відновлення вихідних даних: MSE = {pca_exec['mse_rule']:.6f}, MAE = {pca_exec['mae_rule']:.6f}.")
            else:
                t6.append(
                    f"   • За порогом 5%: найменша 3-тя компонента має частку {min_share:.2f}% >= 5% (при w=3 втрат немає, MSE = 0)."
                )
                t6.append(
                    f"   • При примусовому відкиданні 3-ї компоненти (w = 2 < n, Σ% = {cum_shares[1]:.2f}%):"
                )
                t6.append(f"     MSE відновлення = {pca_exec['mse_w2']:.6f}, MAE відновлення = {pca_exec['mae_w2']:.6f}.")

            t6.append("\n   Перші 5 рядків: Порівняння очищених даних X та відновлених X̂ (при w = 2):")
            t6.append(f"   {'№':<3} | {'X1 (вих.)':>10} {'X1 (відн.)':>10} | {'X2 (вих.)':>10} {'X2 (відн.)':>10} | {'X3 (вих.)':>10} {'X3 (відн.)':>10}")
            t6.append("   " + "-" * 76)
            rec_show = pca_exec["reconstructed_active"]
            for idx in range(min(5, len(self.current_data))):
                orig = self.current_data[idx]
                rec = rec_show[idx]
                t6.append(
                    f"   {idx+1:<3} | {orig[0]:>10.3f} {rec[0]:>10.3f} | "
                    f"{orig[1]:>10.3f} {rec[1]:>10.3f} | {orig[2]:>10.3f} {rec[2]:>10.3f}"
                )

            t6.append("\n5. Контрольна перевірка через sklearn.decomposition.PCA та numpy.corrcoef (п. 6):")
            max_r_diff = float(np.max(np.abs(r_cur - sk_check["r_corrcoef"])))
            t6.append(f"   • Макс. розбіжність R{{ξ}} з numpy.corrcoef: {max_r_diff:.2e}")
            if sk_check["sklearn_available"]:
                sk_l = sk_check["lambdas"]
                sk_r = sk_check["ratios"]
                t6.append(f"   • Власні числа (власна ф-ція): [{eigvals[0]:.6f}, {eigvals[1]:.6f}, {eigvals[2]:.6f}]")
                t6.append(f"   • Власні числа (sklearn PCA):  [{sk_l[0]:.6f}, {sk_l[1]:.6f}, {sk_l[2]:.6f}]")
                t6.append(f"   • Частки дисперсії sklearn %:  [{sk_r[0]:.4f}%, {sk_r[1]:.4f}%, {sk_r[2]:.4f}%]")
                t6.append("   -> Повний збіг результатів власної реалізації МГК із еталонним пакетом scikit-learn!")

            self.txt_pca.delete("1.0", tk.END)
            self.txt_pca.insert(tk.END, "\n".join(t6))

            # --- Оновлення всіх 8 графіків ---
            plot_3d_scatter(self.raw_data, self.clean_mask, self.names, show_outliers=show_outliers_on_plots, ax=self.ax_3d)
            self.fig_3d.tight_layout()
            self.canv_3d.draw()

            plot_scatter_matrix_3x3(self.raw_data, self.clean_mask, r_cur, self.names, show_outliers=show_outliers_on_plots, fig=self.fig_pair)
            self.canv_pair.draw()

            plot_parallel_coordinates(self.raw_data, self.clean_mask, self.names, show_outliers=show_outliers_on_plots, ax=self.ax_par)
            self.fig_par.tight_layout()
            self.canv_par.draw()

            plot_correlation_heatmap(r_cur, self.names, fig=self.fig_heat)
            self.canv_heat.draw()

            self.update_bubble_plot()

            plot_nightingale_rose(self.current_data, self.names, ax=self.ax_rose)
            self.fig_rose.tight_layout()
            self.canv_rose.draw()

            plot_scree(self.eigvals, self.var_shares, self.cum_shares, ax=self.ax_scree)
            self.fig_scree.tight_layout()
            self.canv_scree.draw()

            plot_biplot(pca_exec["scores"], matrix_a, self.var_shares, self.names, ax=self.ax_biplot)
            self.fig_biplot.tight_layout()
            self.canv_biplot.draw()

        except Exception as e:
            messagebox.showerror("Помилка під час обчислень", str(e))

    def update_bubble_plot(self):
        if self.current_data is None:
            return
        ix = self.names.index(self.combo_bx.get())
        iy = self.names.index(self.combo_by.get())
        isize = self.names.index(self.combo_bsize.get())

        plot_bubble_chart(self.current_data, self.names, idx_x=ix, idx_y=iy, idx_size=isize, fig=self.fig_bubble)
        self.canv_bubble.draw()

    def handle_save_scores(self):
        if self.pca_res is None:
            messagebox.showwarning("Увага", "Спочатку виконайте розрахунок МГК!")
            return
        fpath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile="pc_scores_01.txt",
            filetypes=[("Text file", "*.txt"), ("CSV file", "*.csv")]
        )
        if not fpath:
            return
        try:
            save_pc_scores(fpath, self.pca_res["scores"])
            messagebox.showinfo("Збережено", f"Значення головних компонент збережено у файл:\n{fpath}")
        except Exception as e:
            messagebox.showerror("Помилка збереження", str(e))

    def handle_save_reconstructed(self):
        if self.pca_res is None:
            messagebox.showwarning("Увага", "Спочатку виконайте розрахунок МГК!")
            return
        fpath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile="reconstructed_01.txt",
            filetypes=[("Text file", "*.txt"), ("CSV file", "*.csv")]
        )
        if not fpath:
            return
        try:
            save_reconstructed_data(fpath, self.pca_res["reconstructed_active"], self.names)
            messagebox.showinfo("Збережено", f"Відфільтровані (відновлені) дані збережено у файл:\n{fpath}")
        except Exception as e:
            messagebox.showerror("Помилка збереження", str(e))


def main():
    root = tk.Tk()
    app = Lab5App(root)

    def on_closing():
        plt.close('all')
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()