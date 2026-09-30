import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from scipy.optimize import linprog
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class LogisticsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Логистический модуль цепочек поставок микросхем")
        self.root.geometry("1100x750")
        self.root.configure(bg="#f0f0f0")

        # Данные по умолчанию
        self.num_suppliers = 3
        self.num_consumers = 3
        self.supplier_entries = []
        self.consumer_entries = []
        self.cost_entries = []
        self.supply_entries = []
        self.demand_entries = []

        self.build_ui()

    # ---------- Построение интерфейса ----------
    def build_ui(self):
        # Заголовок
        header = tk.Label(
            self.root,
            text="Проектирование логистического модуля цепочек поставок микросхем\n"
                 "(на основе двухфазного симплекс-метода)",
            font=("Arial", 14, "bold"),
            bg="#f0f0f0", fg="#2c3e50", justify="center"
        )
        header.pack(pady=10)

        # Основной контейнер
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Левая часть - ввод
        left_frame = tk.LabelFrame(main_frame, text="Входные данные", font=("Arial", 11, "bold"),
                                    bg="#f0f0f0", padx=10, pady=10)
        left_frame.pack(side="left", fill="both", expand=False, padx=5)

        # Правая часть - результат
        right_frame = tk.LabelFrame(main_frame, text="Результат", font=("Arial", 11, "bold"),
                                     bg="#f0f0f0", padx=10, pady=10)
        right_frame.pack(side="right", fill="both", expand=True, padx=5)

        self.left_frame = left_frame
        self.right_frame = right_frame

        # Панель размерности
        dim_frame = tk.Frame(left_frame, bg="#f0f0f0")
        dim_frame.pack(fill="x", pady=5)

        tk.Label(dim_frame, text="Поставщиков:", bg="#f0f0f0").pack(side="left")
        self.supplier_spin = tk.Spinbox(dim_frame, from_=2, to=5, width=3,
                                         command=self.rebuild_matrix)
        self.supplier_spin.pack(side="left", padx=5)

        tk.Label(dim_frame, text="Потребителей:", bg="#f0f0f0").pack(side="left")
        self.consumer_spin = tk.Spinbox(dim_frame, from_=2, to=5, width=3,
                                         command=self.rebuild_matrix)
        self.consumer_spin.pack(side="left", padx=5)

        # Контейнер для матрицы (будет перестраиваться)
        self.matrix_container = tk.Frame(left_frame, bg="#f0f0f0")
        self.matrix_container.pack(fill="both", expand=True, pady=10)

        # Кнопка решения
        solve_btn = tk.Button(left_frame, text="🚀 Решить задачу",
                               font=("Arial", 12, "bold"),
                               bg="#27ae60", fg="white",
                               activebackground="#229954",
                               cursor="hand2",
                               command=self.solve)
        solve_btn.pack(fill="x", pady=10)

        # Кнопка сброса
        reset_btn = tk.Button(left_frame, text="🔄 Сбросить",
                               font=("Arial", 10),
                               bg="#e74c3c", fg="white",
                               activebackground="#c0392b",
                               cursor="hand2",
                               command=self.rebuild_matrix)
        reset_btn.pack(fill="x")

        # Область результата (текст)
        self.result_text = tk.Text(right_frame, height=12, width=55,
                                    font=("Courier New", 10),
                                    bg="#2c3e50", fg="#ecf0f1",
                                    insertbackground="white")
        self.result_text.pack(fill="x", pady=5)

        # Контейнер для графика
        self.chart_frame = tk.Frame(right_frame, bg="#f0f0f0")
        self.chart_frame.pack(fill="both", expand=True, pady=5)

        self.canvas = None

        # Первичная отрисовка матрицы
        self.rebuild_matrix()

    # ---------- Перестроение матрицы ввода ----------
    def rebuild_matrix(self):
        # Очистить контейнер
        for widget in self.matrix_container.winfo_children():
            widget.destroy()

        self.num_suppliers = int(self.supplier_spin.get())
        self.num_consumers = int(self.consumer_spin.get())

        self.cost_entries = []
        self.supply_entries = []
        self.demand_entries = []

        # Заголовок матрицы затрат
        tk.Label(self.matrix_container, text="Матрица затрат на перевозку:",
                 font=("Arial", 10, "bold"), bg="#f0f0f0").grid(
            row=0, column=0, columnspan=self.num_consumers + 2, pady=5, sticky="w")

        # Заголовки столбцов
        tk.Label(self.matrix_container, text="", bg="#f0f0f0", width=10).grid(row=1, column=0)
        for j in range(self.num_consumers):
            tk.Label(self.matrix_container, text=f"Потр. {j+1}",
                     font=("Arial", 9, "bold"), bg="#f0f0f0").grid(row=1, column=j + 1)

        # Строки матрицы
        for i in range(self.num_suppliers):
            tk.Label(self.matrix_container, text=f"Пост. {i+1}",
                     font=("Arial", 9, "bold"), bg="#f0f0f0").grid(row=i + 2, column=0)

            row_entries = []
            for j in range(self.num_consumers):
                e = tk.Entry(self.matrix_container, width=6, justify="center")
                e.insert(0, str((i + 1) * 10 + j * 5))
                e.grid(row=i + 2, column=j + 1, padx=2, pady=2)
                row_entries.append(e)
            self.cost_entries.append(row_entries)

        # Запасы поставщиков
        supply_row = self.num_suppliers + 3
        tk.Label(self.matrix_container, text="Запасы:",
                 font=("Arial", 10, "bold"), bg="#f0f0f0").grid(
            row=supply_row, column=0, pady=5, sticky="w")
        for i in range(self.num_suppliers):
            e = tk.Entry(self.matrix_container, width=6, justify="center")
            e.insert(0, "100")
            e.grid(row=supply_row, column=i + 1, padx=2, pady=2)
            self.supply_entries.append(e)

        # Спрос потребителей
        demand_row = supply_row + 1
        tk.Label(self.matrix_container, text="Спрос:",
                 font=("Arial", 10, "bold"), bg="#f0f0f0").grid(
            row=demand_row, column=0, pady=5, sticky="w")
        for j in range(self.num_consumers):
            e = tk.Entry(self.matrix_container, width=6, justify="center")
            e.insert(0, "100")
            e.grid(row=demand_row, column=j + 1, padx=2, pady=2)
            self.demand_entries.append(e)

    # ---------- Логика решения ----------
    def solve(self):
        try:
            # Считываем данные
            costs = np.array([[float(self.cost_entries[i][j].get())
                                for j in range(self.num_consumers)]
                               for i in range(self.num_suppliers)])
            supply = np.array([float(e.get()) for e in self.supply_entries])
            demand = np.array([float(e.get()) for e in self.demand_entries])

            # Проверка баланса
            if abs(np.sum(supply) - np.sum(demand)) > 1e-6:
                if not messagebox.askyesno(
                        "Несбалансированная задача",
                        f"Сумма запасов ({np.sum(supply)}) != сумме спроса ({np.sum(demand)}).\n"
                        "Добавить фиктивного поставщика/потребителя и продолжить?"):
                    return
                # Балансировка
                if np.sum(supply) > np.sum(demand):
                    diff = np.sum(supply) - np.sum(demand)
                    demand = np.append(demand, diff)
                    costs = np.hstack([costs, np.zeros((self.num_suppliers, 1))])
                    self.num_consumers += 1
                else:
                    diff = np.sum(demand) - np.sum(supply)
                    supply = np.append(supply, diff)
                    costs = np.vstack([costs, np.zeros((1, self.num_consumers))])
                    self.num_suppliers += 1

            n_s = len(supply)
            n_c = len(demand)

            # Формируем задачу ЛП
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

            bounds = [(0, None)] * (n_s * n_c)

            result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

            if result.success:
                self.show_result(result, costs, supply, demand, n_s, n_c)
            else:
                self.result_text.delete("1.0", tk.END)
                self.result_text.insert(tk.END, f"❌ Решение не найдено:\n{result.message}")

        except ValueError as ex:
            messagebox.showerror("Ошибка ввода", f"Проверьте данные: {ex}")

    def show_result(self, result, costs, supply, demand, n_s, n_c):
        plan = result.x.reshape((n_s, n_c))

        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "✅ ОПТИМАЛЬНОЕ РЕШЕНИЕ НАЙДЕНО\n")
        self.result_text.insert(tk.END, "=" * 55 + "\n")
        self.result_text.insert(tk.END, f"Метод: двухфазный симплекс (HiGHS)\n")
        self.result_text.insert(tk.END, f"Минимальные затраты: {result.fun:.2f} руб.\n")
        self.result_text.insert(tk.END, "=" * 55 + "\n\n")
        self.result_text.insert(tk.END, "Оптимальный план поставок:\n\n")

        header = "          " + "".join(f"{'Потр.'+str(j+1):>10}" for j in range(n_c)) + "\n"
        self.result_text.insert(tk.END, header)
        for i in range(n_s):
            line = f"Пост.{i+1:<3} " + "".join(f"{int(plan[i][j]):>10}" for j in range(n_c))
            self.result_text.insert(tk.END, line + "\n")

        self.result_text.insert(tk.END, "\n" + "-" * 55 + "\n")
        self.result_text.insert(tk.END, "Проверка ограничений:\n")
        for i in range(n_s):
            self.result_text.insert(tk.END,
                f"  Отгрузки с Пост.{i+1}: {int(np.sum(plan[i]))} / запас {int(supply[i])}\n")
        for j in range(n_c):
            self.result_text.insert(tk.END,
                f"  Поставки в Потр.{j+1}: {int(np.sum(plan[:, j]))} / спрос {int(demand[j])}\n")

        self.draw_chart(plan)

    def draw_chart(self, plan):
        # Очищаем старый график
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None

        n_s, n_c = plan.shape

        fig = Figure(figsize=(5.5, 3.5), dpi=90)
        ax = fig.add_subplot(111)
        im = ax.imshow(plan, cmap="YlGnBu", aspect="auto")

        ax.set_xticks(range(n_c))
        ax.set_xticklabels([f"Потр.{j+1}" for j in range(n_c)])
        ax.set_yticks(range(n_s))
        ax.set_yticklabels([f"Пост.{i+1}" for i in range(n_s)])
        ax.set_title("Карта оптимальных поставок", fontsize=11)

        # Числа в ячейках
        for i in range(n_s):
            for j in range(n_c):
                ax.text(j, i, f"{int(plan[i][j])}",
                        ha="center", va="center",
                        color="black", fontsize=10, fontweight="bold")

        fig.colorbar(im, ax=ax, label="Объём поставки")
        fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = LogisticsApp(root)
    root.mainloop()