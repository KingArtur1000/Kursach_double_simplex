import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time


# ═══════════════════════════════════════════════════════════
#   ПАЛИТРА И НАСТРОЙКИ
# ═══════════════════════════════════════════════════════════
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

BG         = "#FAFAFA"
CARD       = "#FFFFFF"
BORDER     = "#E4E4E7"
TEXT       = "#18181B"
MUTED      = "#71717A"
ACCENT     = "#6366F1"
ACCENT_HOV = "#4F46E5"
SUCCESS    = "#10B981"
DANGER     = "#EF4444"
WARNING    = "#F59E0B"
PHASE1     = "#8B5CF6"
PHASE2     = "#0EA5E9"

FONT_MAIN  = ("Segoe UI", 11)
FONT_BOLD  = ("Segoe UI", 11, "bold")
FONT_TITLE = ("Segoe UI", 18, "bold")
FONT_MONO  = ("Consolas", 10)


# ═══════════════════════════════════════════════════════════
#   ДВУХФАЗНЫЙ СИМПЛЕКС-МЕТОД
# ═══════════════════════════════════════════════════════════
def simplex_iterations(A, b, c, basis, log, phase, max_iter=500):
    m, n = A.shape
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    basis = list(basis)

    for iteration in range(1, max_iter + 1):
        B = A[:, basis]
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            log("  ⚠ Вырожденный базис — прерывание.", "error")
            return False, basis, None

        x_B = B_inv @ b
        c_B = c[basis]
        z = c_B @ x_B
        y = c_B @ B_inv
        reduced = c - y @ A
        non_basic = [j for j in range(n) if j not in basis]

        log(f"\n  ┌── Итерация {iteration} (фаза {phase}) ──────────────", "iteration")
        log("  │ Базис: " + ", ".join(f"x{basis[i]+1}" for i in range(m)))
        log("  │ Решение: " + ", ".join(f"x{basis[i]+1}={x_B[i]:.3f}" for i in range(m)))
        log(f"  │ z = {z:.6f}")
        if non_basic:
            shown = non_basic[:12]
            rc_str = ", ".join(f"Δ{j+1}={reduced[j]:+.4f}" for j in shown)
            if len(non_basic) > 12:
                rc_str += ", ..."
            log("  │ Оценки: " + rc_str)

        entering = None
        min_rc = -1e-9
        for j in non_basic:
            if reduced[j] < min_rc:
                min_rc = reduced[j]
                entering = j

        if entering is None:
            log("  │ ✓ Все Δ ≥ 0 → ОПТИМУМ", "success")
            log("  └──────────────────────────────────────", "iteration")
            return True, basis, z

        log(f"  │ → Вводим x{entering+1} (Δ={reduced[entering]:.4f})", "info")
        d = B_inv @ A[:, entering]
        ratios = [(x_B[i] / d[i], i) for i in range(m) if d[i] > 1e-9]

        if not ratios:
            log("  │ ✗ Задача неограничена", "error")
            log("  └──────────────────────────────────────", "iteration")
            return False, basis, None

        min_ratio, leaving_idx = min(ratios)
        leaving = basis[leaving_idx]
        log(f"  │ → Выводим x{leaving+1} (min отн. = {min_ratio:.4f})", "info")
        log("  └──────────────────────────────────────", "iteration")

        basis[leaving_idx] = entering

    log(f"  ⚠ Лимит итераций ({max_iter})", "error")
    return False, basis, None


def solve_two_phase_simplex(c_orig, A_eq, b_eq, log):
    m, n = A_eq.shape
    A = A_eq.astype(float).copy()
    b = b_eq.astype(float).copy()
    for i in range(m):
        if b[i] < 0:
            A[i] = -A[i]; b[i] = -b[i]

    log("╔══════════════════════════════════════════════════════════╗", "header")
    log("║          ДВУХФАЗНЫЙ СИМПЛЕКС-МЕТОД                       ║", "header")
    log("╚══════════════════════════════════════════════════════════╝", "header")
    log("")
    log(f"Размерность: {m} ограничений, {n} переменных")
    log("Целевая функция:  min z = " +
        " + ".join(f"{c_orig[j]:.2f}·x{j+1}" for j in range(n)))
    log("")

    # ----- ФАЗА 1 -----
    log("─" * 60, "phase1")
    log("ФАЗА 1. Поиск начального допустимого базисного решения", "phase1")
    log("─" * 60, "phase1")

    A1 = np.hstack([A, np.eye(m)])
    c1 = np.concatenate([np.zeros(n), np.ones(m)])
    basis = list(range(n, n + m))

    success, basis, z1 = simplex_iterations(A1, b, c1, basis, log, phase=1)
    if not success or z1 > 1e-6:
        log(f"\n❌ Фаза 1 не сошлась (W={z1:.6f})", "error")
        return None

    log(f"\n✅ W = {z1:.6f} ⇒ допустимый базис найден.", "success")

    # Вывод искусственных из базиса
    redundant = []
    for idx in range(m):
        if basis[idx] >= n:
            B = A1[:, basis]
            try:
                B_inv = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                redundant.append(idx); continue
            row = B_inv[idx, :] @ A1
            swapped = False
            for j in range(n):
                if j not in basis and abs(row[j]) > 1e-9:
                    basis[idx] = j; swapped = True; break
            if not swapped:
                redundant.append(idx)

    keep = [i for i in range(m) if i not in redundant]
    A_red = A[keep, :]
    b_red = b[keep]
    basis_red = [basis[i] for i in keep]

    # ----- ФАЗА 2 -----
    log("")
    log("─" * 60, "phase2")
    log("ФАЗА 2. Оптимизация исходной целевой функции", "phase2")
    log("─" * 60, "phase2")

    success, basis_final, z2 = simplex_iterations(
        A_red, b_red, c_orig.astype(float).copy(), basis_red, log, phase=2
    )
    if not success:
        return None

    log(f"\n✅ ОПТИМУМ: z* = {z2:.6f}", "success")

    B = A_red[:, basis_final]
    x_B = np.linalg.solve(B, b_red)
    x = np.zeros(n)
    for i, b_idx in enumerate(basis_final):
        x[b_idx] = x_B[i]
    return x, z2


# ═══════════════════════════════════════════════════════════
#   КАРТОЧКА (скруглённый контейнер)
# ═══════════════════════════════════════════════════════════
class Card(ctk.CTkFrame):
    def __init__(self, master, title=None, **kw):
        super().__init__(master,
                         fg_color=CARD,
                         corner_radius=14,
                         border_width=1,
                         border_color=BORDER,
                         **kw)
        if title:
            ctk.CTkLabel(self, text=title,
                         font=FONT_BOLD,
                         text_color=TEXT,
                         anchor="w").pack(fill="x", padx=16, pady=(14, 6))


# ═══════════════════════════════════════════════════════════
#   ГЛАВНОЕ ПРИЛОЖЕНИЕ
# ═══════════════════════════════════════════════════════════
class LogisticsApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Логистический модуль · Микросхемы")
        self.geometry("1400x880")
        self.minsize(1100, 700)
        self.configure(fg_color=BG)

        # Данные
        self.num_suppliers = 3
        self.num_consumers = 3
        self.cost_entries = []
        self.supply_entries = []
        self.demand_entries = []
        self._saved_data = {'costs': [], 'supply': [], 'demand': []}
        self.canvas = None
        self._pulse_id = None
        self._pulse_state = False

        # Плавное появление окна
        self.attributes("-alpha", 0.0)
        self._fade_in_window()

        self._build_ui()
        self.rebuild_matrix(preserve=False)

    # ─── Fade-in окна ───
    def _fade_in_window(self, alpha=0.0):
        alpha = min(1.0, alpha + 0.08)
        self.attributes("-alpha", alpha)
        if alpha < 1.0:
            self.after(15, lambda: self._fade_in_window(alpha))

    # ─── UI ───
    def _build_ui(self):
        # ----- Заголовок -----
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=28, pady=(22, 8))

        ctk.CTkLabel(header, text="Логистика цепочек поставок",
                     font=("Segoe UI", 24, "bold"),
                     text_color=TEXT).pack(anchor="w")
        ctk.CTkLabel(header,
                     text="Проектирование логистического модуля микросхем · двухфазный симплекс-метод",
                     font=("Segoe UI", 12),
                     text_color=MUTED).pack(anchor="w", pady=(2, 0))

        # ----- Основной контейнер: 2 колонки -----
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=28, pady=(10, 22))
        body.grid_columnconfigure(0, weight=0, minsize=380)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        # ═══ ЛЕВАЯ КОЛОНКА: ВВОД ═══
        left = Card(body, title="Параметры задачи")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        # Размерность
        dim = ctk.CTkFrame(left, fg_color="transparent")
        dim.pack(fill="x", padx=16, pady=(4, 12))

        ctk.CTkLabel(dim, text="Поставщиков", font=FONT_MAIN,
                     text_color=MUTED).pack(side="left")
        self.supplier_spin = ctk.CTkOptionMenu(
            dim, values=[str(i) for i in range(2, 7)], width=60,
            fg_color="#F4F4F5", button_color="#E4E4E7",
            button_hover_color=ACCENT, text_color=TEXT,
            dropdown_fg_color=CARD, dropdown_text_color=TEXT,
            dropdown_hover_color="#F4F4F5", font=FONT_MAIN,
            command=lambda _: self.rebuild_matrix())
        self.supplier_spin.set("3")
        self.supplier_spin.pack(side="left", padx=(8, 20))

        ctk.CTkLabel(dim, text="Потребителей", font=FONT_MAIN,
                     text_color=MUTED).pack(side="left")
        self.consumer_spin = ctk.CTkOptionMenu(
            dim, values=[str(i) for i in range(2, 7)], width=60,
            fg_color="#F4F4F5", button_color="#E4E4E7",
            button_hover_color=ACCENT, text_color=TEXT,
            dropdown_fg_color=CARD, dropdown_text_color=TEXT,
            dropdown_hover_color="#F4F4F5", font=FONT_MAIN,
            command=lambda _: self.rebuild_matrix())
        self.consumer_spin.set("3")
        self.consumer_spin.pack(side="left", padx=8)

        # Матрица (динамическая)
        self.matrix_container = ctk.CTkFrame(left, fg_color="transparent")
        self.matrix_container.pack(fill="x", padx=16, pady=(0, 12))

        # Кнопки
        btn_frame = ctk.CTkFrame(left, fg_color="transparent")
        btn_frame.pack(fill="x", padx=16, pady=(4, 16), side="bottom")

        self.solve_btn = ctk.CTkButton(
            btn_frame, text="Решить задачу",
            font=("Segoe UI", 13, "bold"),
            fg_color=ACCENT, hover_color=ACCENT_HOV,
            text_color="white", corner_radius=10, height=44,
            command=self.solve
        )
        self.solve_btn.pack(fill="x", pady=(0, 8))

        ctk.CTkButton(
            btn_frame, text="Сбросить",
            font=FONT_MAIN,
            fg_color="transparent", hover_color="#F4F4F5",
            text_color=MUTED, border_width=1, border_color=BORDER,
            corner_radius=10, height=38,
            command=self.reset_defaults
        ).pack(fill="x")

        # ═══ ПРАВАЯ КОЛОНКА: РЕЗУЛЬТАТ ═══
        right = ctk.CTkFrame(body, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(0, weight=3)
        right.grid_rowconfigure(1, weight=2)
        right.grid_columnconfigure(0, weight=1)

        # Лог
        log_card = Card(right, title="Журнал решения")
        log_card.grid(row=0, column=0, sticky="nsew", pady=(0, 12))

        self.log_text = tk.Text(
            log_card, font=FONT_MONO, wrap="word",
            bg=CARD, fg=TEXT, relief="flat", bd=0,
            highlightthickness=0, padx=14, pady=4,
            insertbackground=ACCENT
        )
        log_scroll = ctk.CTkScrollbar(log_card, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y", padx=(0, 10), pady=(0, 12))
        self.log_text.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=(0, 12))

        self.log_text.tag_config("header",  foreground=ACCENT,  font=("Consolas", 10, "bold"))
        self.log_text.tag_config("phase1",  foreground=PHASE1,  font=("Consolas", 10, "bold"))
        self.log_text.tag_config("phase2",  foreground=PHASE2,  font=("Consolas", 10, "bold"))
        self.log_text.tag_config("iteration", foreground=WARNING)
        self.log_text.tag_config("info",    foreground=PHASE2)
        self.log_text.tag_config("success", foreground=SUCCESS, font=("Consolas", 10, "bold"))
        self.log_text.tag_config("error",   foreground=DANGER,  font=("Consolas", 10, "bold"))

        # График
        self.chart_card = Card(right, title="Карта оптимальных поставок")
        self.chart_card.grid(row=1, column=0, sticky="nsew")
        self.chart_frame = ctk.CTkFrame(self.chart_card, fg_color="transparent")
        self.chart_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    # ─── Матрица ───
    def _capture(self):
        return {
            'costs': [[e.get() for e in row] for row in self.cost_entries],
            'supply': [e.get() for e in self.supply_entries],
            'demand': [e.get() for e in self.demand_entries],
        }

    def rebuild_matrix(self, preserve=True):
        if preserve and self.cost_entries:
            self._saved_data = self._capture()

        for w in self.matrix_container.winfo_children():
            w.destroy()

        self.num_suppliers = int(self.supplier_spin.get())
        self.num_consumers = int(self.consumer_spin.get())
        self.cost_entries, self.supply_entries, self.demand_entries = [], [], []

        # Заголовок
        ctk.CTkLabel(self.matrix_container, text="Матрица затрат",
                     font=FONT_BOLD, text_color=TEXT, anchor="w").grid(
            row=0, column=0, columnspan=self.num_consumers + 2,
            sticky="w", pady=(0, 6))

        # Шапка
        ctk.CTkLabel(self.matrix_container, text="", width=60).grid(row=1, column=0)
        for j in range(self.num_consumers):
            ctk.CTkLabel(self.matrix_container, text=f"П{j+1}",
                         font=FONT_BOLD, text_color=MUTED, width=60).grid(
                row=1, column=j + 1, padx=3, pady=(0, 4))

        # Строки
        old = self._saved_data.get('costs', [])
        for i in range(self.num_suppliers):
            ctk.CTkLabel(self.matrix_container, text=f"Пост.{i+1}",
                         font=FONT_MAIN, text_color=MUTED, width=60).grid(
                row=i + 2, column=0, sticky="w")
            row_e = []
            for j in range(self.num_consumers):
                e = ctk.CTkEntry(self.matrix_container, width=60, height=30,
                                 corner_radius=8,
                                 fg_color="#F4F4F5", border_width=0,
                                 text_color=TEXT, justify="center",
                                 font=FONT_MAIN)
                val = old[i][j] if (i < len(old) and j < len(old[i])) else str((i + 1) * 10 + j * 5)
                e.insert(0, val)
                e.grid(row=i + 2, column=j + 1, padx=3, pady=3)
                row_e.append(e)
            self.cost_entries.append(row_e)

        # Запасы
        sr = self.num_suppliers + 3
        ctk.CTkLabel(self.matrix_container, text="Запасы",
                     font=FONT_BOLD, text_color=TEXT, anchor="w").grid(
            row=sr, column=0, columnspan=self.num_consumers + 2,
            sticky="w", pady=(10, 4))
        old_s = self._saved_data.get('supply', [])
        for i in range(self.num_suppliers):
            e = ctk.CTkEntry(self.matrix_container, width=60, height=30,
                             corner_radius=8, fg_color="#F4F4F5",
                             border_width=0, text_color=TEXT,
                             justify="center", font=FONT_MAIN)
            e.insert(0, old_s[i] if i < len(old_s) else "100")
            e.grid(row=sr + 1 + i // self.num_consumers,
                   column=1 + (i % self.num_consumers),
                   padx=3, pady=3)
            self.supply_entries.append(e)

        # Спрос
        dr = sr + 2 + (self.num_suppliers - 1) // self.num_consumers
        ctk.CTkLabel(self.matrix_container, text="Спрос",
                     font=FONT_BOLD, text_color=TEXT, anchor="w").grid(
            row=dr, column=0, columnspan=self.num_consumers + 2,
            sticky="w", pady=(10, 4))
        old_d = self._saved_data.get('demand', [])
        for j in range(self.num_consumers):
            e = ctk.CTkEntry(self.matrix_container, width=60, height=30,
                             corner_radius=8, fg_color="#F4F4F5",
                             border_width=0, text_color=TEXT,
                             justify="center", font=FONT_MAIN)
            e.insert(0, old_d[j] if j < len(old_d) else "100")
            e.grid(row=dr + 1, column=j + 1, padx=3, pady=3)
            self.demand_entries.append(e)

    def reset_defaults(self):
        self._saved_data = {'costs': [], 'supply': [], 'demand': []}
        self.rebuild_matrix(preserve=False)

    # ─── Лог ───
    def clear_log(self):
        self.log_text.delete("1.0", tk.END)

    def log(self, msg, tag=None):
        if tag:
            self.log_text.insert(tk.END, msg + "\n", tag)
        else:
            self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    # ─── Решение ───
    def solve(self):
        try:
            costs = np.array([[float(self.cost_entries[i][j].get())
                                for j in range(self.num_consumers)]
                               for i in range(self.num_suppliers)])
            supply = np.array([float(e.get()) for e in self.supply_entries])
            demand = np.array([float(e.get()) for e in self.demand_entries])
        except ValueError as ex:
            messagebox.showerror("Ошибка ввода", f"Проверьте данные: {ex}")
            return

        # Анимация кнопки
        self.solve_btn.configure(text="Считаю…", state="disabled")
        self.update_idletasks()

        def run():
            n_s, n_c = len(supply), len(demand)

             # Инициализация на случай, если задача сбалансирована
            supply_, demand_, costs_ = supply, demand, costs

            # Балансировка
            if abs(np.sum(supply) - np.sum(demand)) > 1e-6:
                if np.sum(supply) > np.sum(demand):
                    diff = np.sum(supply) - np.sum(demand)
                    demand_ = np.append(demand, diff)
                    costs_ = np.hstack([costs, np.zeros((n_s, 1))])
                    n_c += 1
                else:
                    diff = np.sum(demand) - np.sum(supply)
                    supply_ = np.append(supply, diff)
                    costs_ = np.vstack([costs, np.zeros((1, n_c))])
                    n_s += 1
            else:
                supply_, demand_, costs_ = supply, demand, costs

            c = costs_.flatten()
            A_eq, b_eq = [], []
            for i in range(n_s):
                row = np.zeros(n_s * n_c)
                for j in range(n_c):
                    row[i * n_c + j] = 1
                A_eq.append(row); b_eq.append(supply_[i])
            for j in range(n_c):
                row = np.zeros(n_s * n_c)
                for i in range(n_s):
                    row[i * n_c + j] = 1
                A_eq.append(row); b_eq.append(demand_[j])
            A_eq = np.array(A_eq); b_eq = np.array(b_eq)

            self.clear_log()
            result = solve_two_phase_simplex(c, A_eq, b_eq, self.log)

            if result is None:
                self.log("\n❌ Решение не найдено.", "error")
                self.after(0, self._finish_solve)
                return

            x_opt, z_opt = result
            plan = x_opt.reshape((n_s, n_c))

            self.log("")
            self.log("╔══════════════════════════════════════════════════════════╗", "header")
            self.log("║              ОПТИМАЛЬНЫЙ ПЛАН ПОСТАВОК                   ║", "header")
            self.log("╚══════════════════════════════════════════════════════════╝", "header")
            self.log(f"z* = {z_opt:.2f} руб.", "success")
            self.log("")

            header_line = "          " + "".join(f"{'П'+str(j+1):>9}" for j in range(n_c))
            self.log(header_line, "info")
            for i in range(n_s):
                line = f"Пост.{i+1:<3} " + "".join(
                    f"{int(round(plan[i][j])):>9}" for j in range(n_c))
                self.log(line)

            self.log("")
            self.log("Проверка ограничений:", "info")
            for i in range(n_s):
                self.log(f"  Отгрузки Пост.{i+1}: {int(round(np.sum(plan[i])))} / запас {int(supply_[i])}")
            for j in range(n_c):
                self.log(f"  Поставки П{j+1}: {int(round(np.sum(plan[:, j])))} / спрос {int(demand_[j])}")

            self.after(0, lambda: self._draw_chart(plan))
            self.after(0, self._finish_solve)

        threading.Thread(target=run, daemon=True).start()

    def _finish_solve(self):
        self.solve_btn.configure(text="Решить задачу", state="normal")
        self._start_pulse()

    # ─── Pulse-анимация кнопки ───
    def _start_pulse(self):
        if self._pulse_id:
            self.after_cancel(self._pulse_id)
        self._pulse_state = False
        self._pulse_step(0)

    def _pulse_step(self, step):
        if step > 6:
            self.solve_btn.configure(fg_color=ACCENT)
            return
        # между ACCENT и ACCENT_HOV
        t = step / 6.0
        c1 = (99, 102, 241); c2 = (79, 70, 229)
        col = tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))
        hex_col = "#%02X%02X%02X" % col
        self.solve_btn.configure(fg_color=hex_col)
        self._pulse_id = self.after(70, lambda: self._pulse_step(step + 1))

    # ─── График ───
    def _draw_chart(self, plan):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None

        n_s, n_c = plan.shape
        fig = Figure(figsize=(6.5, 3.5), dpi=100, facecolor=CARD)
        ax = fig.add_subplot(111)
        ax.set_facecolor(CARD)

        vmax = plan.max() if plan.max() > 0 else 1
        im = ax.imshow(plan, cmap="Blues", aspect="auto", vmin=0, vmax=vmax)

        ax.set_xticks(range(n_c))
        ax.set_xticklabels([f"Потр.{j+1}" for j in range(n_c)], color=TEXT)
        ax.set_yticks(range(n_s))
        ax.set_yticklabels([f"Пост.{i+1}" for i in range(n_s)], color=TEXT)

        ax.set_title("Карта оптимальных поставок", color=TEXT, fontsize=11, pad=10)
        for spine in ax.spines.values():
            spine.set_color(BORDER)

        threshold = vmax * 0.5
        for i in range(n_s):
            for j in range(n_c):
                val = int(round(plan[i][j]))
                color = "white" if val > threshold else TEXT
                ax.text(j, i, str(val), ha="center", va="center",
                        color=color, fontsize=11, fontweight="bold")

        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Объём поставки", color=MUTED)
        cb.ax.tick_params(colors=MUTED)
        fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)


# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = LogisticsApp()
    app.mainloop()