import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
#      ДВУХФАЗНЫЙ СИМПЛЕКС-МЕТОД (с логированием)
# ============================================================
def simplex_iterations(A, b, c, basis, log, phase, max_iter=500):
    """Двухфазный симплекс. A: m×n, b: m, c: n, basis: m индексов."""
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

        # ⚠ КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: x_B = B⁻¹·b, именно его используем в ratio test
        x_B = B_inv @ b
        c_B = c[basis]
        z = c_B @ x_B

        y = c_B @ B_inv
        reduced = c - y @ A

        non_basic = [j for j in range(n) if j not in basis]

        # ---- Лог ----
        log(f"\n  ┌── Итерация {iteration} (фаза {phase}) ──────────────────", "iteration")
        log("  │ Базис: " + ", ".join(f"x{basis[i]+1}" for i in range(m)))
        log("  │ Текущее решение: " +
            ", ".join(f"x{basis[i]+1}={x_B[i]:.3f}" for i in range(m)))
        log(f"  │ Значение z = {z:.6f}")
        if non_basic:
            shown = non_basic[:12]
            rc_str = ", ".join(f"Δ{j+1}={reduced[j]:+.4f}" for j in shown)
            if len(non_basic) > 12:
                rc_str += ", ..."
            log("  │ Приведённые оценки: " + rc_str)

        # ---- Оптимальность ----
        entering = None
        min_rc = -1e-9
        for j in non_basic:
            if reduced[j] < min_rc:
                min_rc = reduced[j]
                entering = j

        if entering is None:
            log("  │ ✓ Все приведённые оценки ≥ 0 → ОПТИМУМ", "success")
            log("  └──────────────────────────────────────────────", "iteration")
            return True, basis, z

        log(f"  │ → Вводим в базис x{entering+1} (Δ = {reduced[entering]:.4f})", "info")

        # ---- Направление ----
        d = B_inv @ A[:, entering]

        # ⚠ Ratio test использует x_B, а не b!
        ratios = []
        for i in range(m):
            if d[i] > 1e-9:
                ratios.append((x_B[i] / d[i], i))

        if not ratios:
            log("  │ ✗ Задача неограничена", "error")
            log("  └──────────────────────────────────────────────", "iteration")
            return False, basis, None

        min_ratio, leaving_idx = min(ratios)
        leaving = basis[leaving_idx]
        log(f"  │ → Выводим из базиса x{leaving+1} (min отн. = {min_ratio:.4f})", "info")
        log("  └──────────────────────────────────────────────", "iteration")

        basis[leaving_idx] = entering

    log(f"  ⚠ Достигнут лимит итераций ({max_iter})", "error")
    return False, basis, None


def solve_two_phase_simplex(c_orig, A_eq, b_eq, log):
    """Двухфазный симплекс с корректным переходом к Фазе 2."""
    m, n = A_eq.shape
    A = A_eq.astype(float).copy()
    b = b_eq.astype(float).copy()

    # Приводим b к неотрицательным
    for i in range(m):
        if b[i] < 0:
            A[i] = -A[i]
            b[i] = -b[i]

    log("╔══════════════════════════════════════════════════════════╗", "header")
    log("║          ДВУХФАЗНЫЙ СИМПЛЕКС-МЕТОД                       ║", "header")
    log("╚══════════════════════════════════════════════════════════╝", "header")
    log("")
    log(f"Размерность: {m} ограничений, {n} переменных")
    log("Целевая функция:  min z = " +
        " + ".join(f"{c_orig[j]:.2f}·x{j+1}" for j in range(n)))
    log("")

    # ================== ФАЗА 1 ==================
    log("─" * 60, "phase")
    log("ФАЗА 1. Поиск начального допустимого базисного решения", "phase")
    log("─" * 60, "phase")
    log("Добавляем искусственные переменные a1..am (по одной на каждое")
    log("ограничение). Решаем вспомогательную задачу: min W = Σ ai")
    log("")

    A1 = np.hstack([A, np.eye(m)])
    c1 = np.concatenate([np.zeros(n), np.ones(m)])
    basis = list(range(n, n + m))

    success, basis, z1 = simplex_iterations(A1, b, c1, basis, log, phase=1)

    if not success:
        log("\n❌ Фаза 1 не сошлась", "error")
        return None

    log("")
    log(f"Минимум Фазы 1: W = {z1:.6f}", "info")
    if z1 > 1e-6:
        log("❌ W > 0 ⇒ исходная задача не имеет допустимых решений.", "error")
        return None
    log("✅ W = 0 ⇒ найдено начальное допустимое базисное решение.", "success")

    # ============================================================
    # ПРИНУДИТЕЛЬНЫЙ ВЫВОД ИСКУССТВЕННЫХ ИЗ БАЗИСА
    # ============================================================
    log("")
    log("Выводим искусственные переменные из базиса (если остались):", "info")

    redundant_rows = []
    for idx in range(m):
        if basis[idx] >= n:  # в базисе сидит искусственная
            B = A1[:, basis]
            try:
                B_inv = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                redundant_rows.append(idx)
                log(f"  ⚠ Строка {idx+1}: базис вырожден, "
                    f"искусственная x{basis[idx]+1} отброшена", "info")
                continue
            row = B_inv[idx, :] @ A1
            swapped = False
            for j in range(n):
                if j not in basis and abs(row[j]) > 1e-9:
                    basis[idx] = j
                    swapped = True
                    break
            if not swapped:
                redundant_rows.append(idx)
                log(f"  ⚠ Строка {idx+1} избыточна: искусственная "
                    f"x{basis[idx]+1} = 0, ограничение линейно-зависимо", "info")

    if not redundant_rows:
        log("  ✓ Искусственные успешно выведены — все строки независимы", "success")

    # Оставляем только независимые строки
    keep = [i for i in range(m) if i not in redundant_rows]
    A_red = A[keep, :]
    b_red = b[keep]
    basis_red = [basis[i] for i in keep]

    # ================== ФАЗА 2 ==================
    log("")
    log("─" * 60, "phase")
    log("ФАЗА 2. Оптимизация исходной целевой функции", "phase")
    log("─" * 60, "phase")
    log(f"Работаем только с исходными {n} переменными. Столбцы")
    log(f"искусственных переменных отброшены.")
    log(f"Активных ограничений: {len(keep)}  "
        f"(отброшено избыточных: {len(redundant_rows)})")
    log("")

    # ✅ Отрезаем только первые n столбцов — искусственные больше не помешают
    success, basis_final, z2 = simplex_iterations(
        A_red, b_red, c_orig.astype(float).copy(), basis_red, log, phase=2
    )

    if not success:
        log("\n❌ Фаза 2 не сошлась", "error")
        return None

    log("")
    log(f"✅ ОПТИМАЛЬНОЕ ЗНАЧЕНИЕ: z* = {z2:.6f}", "success")

    # Извлекаем решение по исходным переменным
    B = A_red[:, basis_final]
    x_B = np.linalg.solve(B, b_red)
    x = np.zeros(n)
    for i, b_idx in enumerate(basis_final):
        x[b_idx] = x_B[i]

    return x, z2


# ============================================================
#                  ГРАФИЧЕСКОЕ ПРИЛОЖЕНИЕ
# ============================================================
class LogisticsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Логистический модуль цепочек поставок микросхем")
        self.root.geometry("1280x800")
        self.root.configure(bg="#f0f0f0")

        self.num_suppliers = 3
        self.num_consumers = 3
        self.cost_entries = []
        self.supply_entries = []
        self.demand_entries = []

        # Сохранение прошлых значений
        self._saved_data = {
            'costs': [],
            'supply': [],
            'demand': []
        }

        self.build_ui()

    # ---------- UI ----------
    def build_ui(self):
        header = tk.Label(
            self.root,
            text="Проектирование логистического модуля цепочек поставок микросхем\n"
                 "(на основе двухфазного симплекс-метода)",
            font=("Arial", 14, "bold"),
            bg="#f0f0f0", fg="#2c3e50", justify="center"
        )
        header.pack(pady=10)

        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # ---- Левая часть: ввод ----
        left_frame = tk.LabelFrame(main_frame, text=" Входные данные ",
                                    font=("Arial", 11, "bold"),
                                    bg="#f0f0f0", fg="#2c3e50",
                                    padx=10, pady=10)
        left_frame.pack(side="left", fill="y", padx=5)

        dim_frame = tk.Frame(left_frame, bg="#f0f0f0")
        dim_frame.pack(fill="x", pady=5)

        tk.Label(dim_frame, text="Поставщиков:", bg="#f0f0f0").pack(side="left")
        self.supplier_spin = tk.Spinbox(dim_frame, from_=2, to=6, width=3,
                                         command=self.rebuild_matrix)
        self.supplier_spin.pack(side="left", padx=5)

        tk.Label(dim_frame, text="Потребителей:", bg="#f0f0f0").pack(side="left")
        self.consumer_spin = tk.Spinbox(dim_frame, from_=2, to=6, width=3,
                                         command=self.rebuild_matrix)
        self.consumer_spin.pack(side="left", padx=5)

        self.matrix_container = tk.Frame(left_frame, bg="#f0f0f0")
        self.matrix_container.pack(fill="both", expand=True, pady=10)

        solve_btn = tk.Button(left_frame, text="🚀 Решить задачу",
                               font=("Arial", 12, "bold"),
                               bg="#27ae60", fg="white",
                               activebackground="#229954",
                               cursor="hand2",
                               command=self.solve)
        solve_btn.pack(fill="x", pady=10)

        reset_btn = tk.Button(left_frame, text="🔄 Сбросить значения",
                               font=("Arial", 10),
                               bg="#e74c3c", fg="white",
                               activebackground="#c0392b",
                               cursor="hand2",
                               command=self.reset_defaults)
        reset_btn.pack(fill="x")

        # ---- Правая часть: результат ----
        right_frame = tk.LabelFrame(main_frame, text=" Результат ",
                                     font=("Arial", 11, "bold"),
                                     bg="#f0f0f0", fg="#2c3e50",
                                     padx=10, pady=10)
        right_frame.pack(side="right", fill="both", expand=True, padx=5)

        # Верх: лог
        log_frame = tk.Frame(right_frame)
        log_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(log_frame, height=20, width=70,
                                 font=("Consolas", 9),
                                 bg="#fbfbfb", fg="#2c3e50",
                                 insertbackground="#2c3e50",
                                 wrap="word")
        scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar.set)

        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Цветовые теги для логов
        self.log_text.tag_config("header", foreground="#1a5490",
                                  font=("Consolas", 10, "bold"))
        self.log_text.tag_config("phase", foreground="#8e44ad",
                                  font=("Consolas", 9, "bold"))
        self.log_text.tag_config("iteration", foreground="#d35400")
        self.log_text.tag_config("info", foreground="#2980b9")
        self.log_text.tag_config("success", foreground="#27ae60",
                                  font=("Consolas", 9, "bold"))
        self.log_text.tag_config("error", foreground="#c0392b",
                                  font=("Consolas", 9, "bold"))

        # Низ: график
        self.chart_frame = tk.Frame(right_frame, bg="#f0f0f0")
        self.chart_frame.pack(fill="both", expand=True, pady=5)
        self.canvas = None

        self.rebuild_matrix(preserve=False)

    # ---------- Работа с матрицей ----------
    def _capture_values(self):
        data = {
            'costs': [[e.get() for e in row] for row in self.cost_entries],
            'supply': [e.get() for e in self.supply_entries],
            'demand': [e.get() for e in self.demand_entries],
        }
        return data

    def rebuild_matrix(self, preserve=True):
        # Считываем старые значения
        if preserve and self.cost_entries:
            self._saved_data = self._capture_values()

        for w in self.matrix_container.winfo_children():
            w.destroy()

        self.num_suppliers = int(self.supplier_spin.get())
        self.num_consumers = int(self.consumer_spin.get())

        self.cost_entries = []
        self.supply_entries = []
        self.demand_entries = []

        tk.Label(self.matrix_container, text="Матрица затрат на перевозку:",
                 font=("Arial", 10, "bold"), bg="#f0f0f0").grid(
            row=0, column=0, columnspan=self.num_consumers + 2, pady=5, sticky="w")

        tk.Label(self.matrix_container, text="", bg="#f0f0f0", width=10).grid(row=1, column=0)
        for j in range(self.num_consumers):
            tk.Label(self.matrix_container, text=f"Потр.{j+1}",
                     font=("Arial", 9, "bold"), bg="#f0f0f0").grid(row=1, column=j + 1)

        old_costs = self._saved_data.get('costs', [])
        for i in range(self.num_suppliers):
            tk.Label(self.matrix_container, text=f"Пост.{i+1}",
                     font=("Arial", 9, "bold"), bg="#f0f0f0").grid(row=i + 2, column=0)
            row_entries = []
            for j in range(self.num_consumers):
                e = tk.Entry(self.matrix_container, width=6, justify="center")
                # Восстанавливаем или ставим дефолт
                if i < len(old_costs) and j < len(old_costs[i]):
                    val = old_costs[i][j]
                else:
                    val = str((i + 1) * 10 + j * 5)
                e.insert(0, val)
                e.grid(row=i + 2, column=j + 1, padx=2, pady=2)
                row_entries.append(e)
            self.cost_entries.append(row_entries)

        # Запасы
        supply_row = self.num_suppliers + 3
        tk.Label(self.matrix_container, text="Запасы:",
                 font=("Arial", 10, "bold"), bg="#f0f0f0").grid(
            row=supply_row, column=0, pady=5, sticky="w")
        old_supply = self._saved_data.get('supply', [])
        for i in range(self.num_suppliers):
            e = tk.Entry(self.matrix_container, width=6, justify="center")
            val = old_supply[i] if i < len(old_supply) else "100"
            e.insert(0, val)
            e.grid(row=supply_row, column=i + 1, padx=2, pady=2)
            self.supply_entries.append(e)

        # Спрос
        demand_row = supply_row + 1
        tk.Label(self.matrix_container, text="Спрос:",
                 font=("Arial", 10, "bold"), bg="#f0f0f0").grid(
            row=demand_row, column=0, pady=5, sticky="w")
        old_demand = self._saved_data.get('demand', [])
        for j in range(self.num_consumers):
            e = tk.Entry(self.matrix_container, width=6, justify="center")
            val = old_demand[j] if j < len(old_demand) else "100"
            e.insert(0, val)
            e.grid(row=demand_row, column=j + 1, padx=2, pady=2)
            self.demand_entries.append(e)

    def reset_defaults(self):
        # Полный сброс
        self._saved_data = {'costs': [], 'supply': [], 'demand': []}
        self.rebuild_matrix(preserve=False)

    # ---------- Логирование ----------
    def clear_log(self):
        self.log_text.delete("1.0", tk.END)

    def log(self, msg, tag=None):
        if tag:
            self.log_text.insert(tk.END, msg + "\n", tag)
        else:
            self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    # ---------- Решение ----------
    def solve(self):
        self.clear_log()
        try:
            costs = np.array([[float(self.cost_entries[i][j].get())
                                for j in range(self.num_consumers)]
                               for i in range(self.num_suppliers)])
            supply = np.array([float(e.get()) for e in self.supply_entries])
            demand = np.array([float(e.get()) for e in self.demand_entries])
        except ValueError as ex:
            messagebox.showerror("Ошибка ввода", f"Проверьте данные: {ex}")
            return

        n_s = len(supply)
        n_c = len(demand)

        # Балансировка
        if abs(np.sum(supply) - np.sum(demand)) > 1e-6:
            if not messagebox.askyesno(
                    "Несбалансированная задача",
                    f"Σ запасов ({np.sum(supply):.0f}) ≠ Σ спроса ({np.sum(demand):.0f}).\n\n"
                    f"Добавить фиктивного поставщика/потребителя?"):
                return
            if np.sum(supply) > np.sum(demand):
                diff = np.sum(supply) - np.sum(demand)
                demand = np.append(demand, diff)
                costs = np.hstack([costs, np.zeros((n_s, 1))])
                self.log(f"⚙ Добавлен фиктивный потребитель со спросом {diff:.0f}",
                         "info")
                n_c += 1
            else:
                diff = np.sum(demand) - np.sum(supply)
                supply = np.append(supply, diff)
                costs = np.vstack([costs, np.zeros((1, n_c))])
                self.log(f"⚙ Добавлен фиктивный поставщик с запасом {diff:.0f}",
                         "info")
                n_s += 1

        # Формулировка задачи ЛП
        c = costs.flatten()
        A_eq = []
        b_eq = []
        for i in range(n_s):
            row = np.zeros(n_s * n_c)
            for j in range(n_c):
                row[i * n_c + j] = 1
            A_eq.append(row)
            b_eq.append(supply[i])
        for j in range(n_c):
            row = np.zeros(n_s * n_c)
            for i in range(n_s):
                row[i * n_c + j] = 1
            A_eq.append(row)
            b_eq.append(demand[j])
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)

        # Запуск решения
        result = solve_two_phase_simplex(c, A_eq, b_eq, self.log)

        if result is None:
            self.log("\n❌ Решение не найдено.", "error")
            return

        x_opt, z_opt = result
        plan = x_opt.reshape((n_s, n_c))

        # Вывод итоговой таблицы
        self.log("")
        self.log("╔══════════════════════════════════════════════════════════╗", "header")
        self.log("║              ОПТИМАЛЬНЫЙ ПЛАН ПОСТАВОК                   ║", "header")
        self.log("╚══════════════════════════════════════════════════════════╝", "header")
        self.log(f"Минимальные суммарные затраты: z* = {z_opt:.2f} руб.", "success")
        self.log("")

        header = "          " + "".join(f"{'Потр.'+str(j+1):>9}" for j in range(n_c))
        self.log(header, "info")
        for i in range(n_s):
            line = f"Пост.{i+1:<3} " + "".join(f"{int(round(plan[i][j])):>9}"
                                                 for j in range(n_c))
            self.log(line)

        self.log("")
        self.log("Проверка ограничений:", "info")
        for i in range(n_s):
            self.log(f"  Отгрузки с Пост.{i+1}: {int(round(np.sum(plan[i])))} / запас {int(supply[i])}")
        for j in range(n_c):
            self.log(f"  Поставки в Потр.{j+1}: {int(round(np.sum(plan[:, j])))} / спрос {int(demand[j])}")

        self.draw_chart(plan)

    # ---------- График ----------
    def draw_chart(self, plan):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None

        n_s, n_c = plan.shape
        fig = Figure(figsize=(6, 3.2), dpi=90, facecolor="#f0f0f0")
        ax = fig.add_subplot(111)
        im = ax.imshow(plan, cmap="Blues", aspect="auto")

        ax.set_xticks(range(n_c))
        ax.set_xticklabels([f"Потр.{j+1}" for j in range(n_c)])
        ax.set_yticks(range(n_s))
        ax.set_yticklabels([f"Пост.{i+1}" for i in range(n_s)])
        ax.set_title("Карта оптимальных поставок", fontsize=11)

        for i in range(n_s):
            for j in range(n_c):
                val = int(round(plan[i][j]))
                color = "white" if val > plan.max() * 0.5 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        color=color, fontsize=10, fontweight="bold")

        fig.colorbar(im, ax=ax, label="Объём поставки")
        fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = LogisticsApp(root)
    root.mainloop()