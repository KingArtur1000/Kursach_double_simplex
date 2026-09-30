"""Главное окно приложения."""
import tkinter as tk
from tkinter import messagebox
import threading
import numpy as np
import customtkinter as ctk

from .theme import (
    BG, CARD, BORDER, TEXT, MUTED, ACCENT, ACCENT_HOV,
    FONT_MAIN, FONT_BOLD,
)
from .widgets import Card, LogPanel
from core.simplex import solve_two_phase_simplex
from core.logistics import balance_task, build_lp_matrices
from viz.chart import render_chart


# ═══════════════════════════════════════════════════════════
#   Скролл-контейнер с ДВУМЯ скроллбарами (tk, без CTk-магии)
# ═══════════════════════════════════════════════════════════
class ScrollFrame(tk.Frame):
    """Canvas + H/V Scrollbars. Содержимое кладём в .inner."""

    def __init__(self, master, **kw):
        super().__init__(master, bg=CARD, bd=0, highlightthickness=0, **kw)

        self.canvas = tk.Canvas(
            self, bg=CARD, highlightthickness=0, bd=0,
        )
        self.v_scroll = ctk.CTkScrollbar(
            self, command=self.canvas.yview, orientation="vertical",
            button_color="#E4E4E7", button_hover_color=ACCENT,
            fg_color=CARD, width=12,
        )
        self.h_scroll = ctk.CTkScrollbar(
            self, command=self.canvas.xview, orientation="horizontal",
            button_color="#E4E4E7", button_hover_color=ACCENT,
            fg_color=CARD, height=12,
        )
        self.canvas.configure(
            yscrollcommand=self.v_scroll.set,
            xscrollcommand=self.h_scroll.set,
        )

        # Раскладка через grid, чтобы скроллбары корректно прилипали
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.inner = tk.Frame(self.canvas, bg=CARD)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_inner_configure)

        # Колёсико мыши — вертикальный скролл
        self.canvas.bind("<Enter>", lambda e: self._bind_wheel(True))
        self.canvas.bind("<Leave>", lambda e: self._bind_wheel(False))

    def _on_inner_configure(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _bind_wheel(self, on):
        if on:
            self.canvas.bind_all("<MouseWheel>", self._on_wheel)
            self.canvas.bind_all("<Button-4>", self._on_wheel)
            self.canvas.bind_all("<Button-5>", self._on_wheel)
        else:
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")

    def _on_wheel(self, event):
        # Shift+колесо = горизонтальный скролл (бонус)
        shift = (event.state & 0x0001) != 0
        if event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        else:
            delta = -1 if event.delta > 0 else 1
        if shift:
            self.canvas.xview_scroll(delta, "units")
        else:
            self.canvas.yview_scroll(delta, "units")


class LogisticsApp(ctk.CTk):
    LEFT_WIDTH = 400

    def __init__(self):
        super().__init__()
        self.title("Логистический модуль · Микросхемы")
        self.geometry("1400x880")
        self.minsize(1100, 700)
        self.configure(fg_color=BG)

        self.num_suppliers = 3
        self.num_consumers = 3
        self.cost_entries, self.supply_entries, self.demand_entries = [], [], []
        self._saved_data = {'costs': [], 'supply': [], 'demand': []}
        self.canvas = None
        self._pulse_id = None

        self.attributes("-alpha", 0.0)
        self._fade_in_window()

        self._build_ui()
        self.rebuild_matrix(preserve=False)

    def _fade_in_window(self, alpha=0.0):
        alpha = min(1.0, alpha + 0.08)
        self.attributes("-alpha", alpha)
        if alpha < 1.0:
            self.after(15, lambda: self._fade_in_window(alpha))

    # ─── UI ───
    def _build_ui(self):
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=28, pady=(22, 8))

        ctk.CTkLabel(header, text="Логистика цепочек поставок",
                     font=("Segoe UI", 24, "bold"),
                     text_color=TEXT, fg_color=BG).pack(anchor="w")
        ctk.CTkLabel(
            header,
            text="Проектирование логистического модуля микросхем · двухфазный симплекс-метод",
            font=("Segoe UI", 12), text_color=MUTED, fg_color=BG,
        ).pack(anchor="w", pady=(2, 0))

        self.body = tk.Frame(self, bg=BG)
        self.body.pack(fill="both", expand=True, padx=28, pady=(10, 22))

        self._build_left_panel(self.body)
        self._build_right_panel(self.body)

    def _build_left_panel(self, body):
        # Фиксированная ширина — горизонтальный скролл решает всё остальное
        self.left_card = tk.Frame(
            body, bg=CARD, bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=BORDER,
            width=self.LEFT_WIDTH,
        )
        self.left_card.pack(side="left", fill="y", padx=(0, 12))
        self.left_card.pack_propagate(False)

        self.left_card.grid_columnconfigure(0, weight=1)
        self.left_card.grid_rowconfigure(2, weight=1)

        # Заголовок
        tk.Label(
            self.left_card, text="Параметры задачи",
            bg=CARD, fg=TEXT, font=FONT_BOLD, anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 6))

        # Размерности
        dim = tk.Frame(self.left_card, bg=CARD)
        dim.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 10))

        tk.Label(dim, text="Поставщиков", font=FONT_MAIN,
                 bg=CARD, fg=MUTED).pack(side="left")
        self.supplier_spin = self._make_spin(dim, "3")
        self.supplier_spin.pack(side="left", padx=(8, 16))

        tk.Label(dim, text="Потребителей", font=FONT_MAIN,
                 bg=CARD, fg=MUTED).pack(side="left")
        self.consumer_spin = self._make_spin(dim, "3")
        self.consumer_spin.pack(side="left", padx=8)

        # Скролл с двумя полосами
        self.matrix_scroll = ScrollFrame(self.left_card)
        self.matrix_scroll.grid(row=2, column=0, sticky="nsew", padx=(4, 4), pady=(0, 6))

        self.matrix_container = self.matrix_scroll.inner

        # Кнопки
        btn_frame = tk.Frame(self.left_card, bg=CARD)
        btn_frame.grid(row=3, column=0, sticky="ew", padx=16, pady=(4, 16))

        self.solve_btn = ctk.CTkButton(
            btn_frame, text="Решить задачу",
            font=("Segoe UI", 13, "bold"),
            fg_color=ACCENT, hover_color=ACCENT_HOV,
            text_color="white", corner_radius=10, height=44,
            command=self.solve,
        )
        self.solve_btn.pack(fill="x", pady=(0, 8))

        ctk.CTkButton(
            btn_frame, text="Сбросить",
            font=FONT_MAIN, fg_color="transparent", hover_color="#F4F4F5",
            text_color=MUTED, border_width=1, border_color=BORDER,
            corner_radius=10, height=38,
            command=self.reset_defaults,
        ).pack(fill="x")

    def _make_spin(self, parent, default):
        sp = ctk.CTkOptionMenu(
            parent, values=[str(i) for i in range(2, 7)], width=60,
            fg_color="#F4F4F5", button_color="#E4E4E7",
            button_hover_color=ACCENT, text_color=TEXT,
            dropdown_fg_color=CARD, dropdown_text_color=TEXT,
            dropdown_hover_color="#F4F4F5", font=FONT_MAIN,
            command=lambda _: self.rebuild_matrix(),
        )
        sp.set(default)
        return sp

    def _build_right_panel(self, body):
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)
        right.grid_rowconfigure(0, weight=3)
        right.grid_rowconfigure(1, weight=2)
        right.grid_columnconfigure(0, weight=1)

        self.log_panel = LogPanel(right, title="Журнал решения")
        self.log_panel.grid(row=0, column=0, sticky="nsew", pady=(0, 12))

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

        tk.Label(self.matrix_container, text="Матрица затрат",
                 bg=CARD, fg=TEXT, font=FONT_BOLD, anchor="w").grid(
            row=0, column=0, columnspan=self.num_consumers + 2,
            sticky="w", pady=(0, 6))

        tk.Label(self.matrix_container, text="", width=8, bg=CARD).grid(row=1, column=0)
        for j in range(self.num_consumers):
            tk.Label(self.matrix_container, text=f"П{j+1}",
                     bg=CARD, fg=MUTED, font=FONT_BOLD, width=7).grid(
                row=1, column=j + 1, padx=3, pady=(0, 4))

        old = self._saved_data.get('costs', [])
        for i in range(self.num_suppliers):
            tk.Label(self.matrix_container, text=f"Пост.{i+1}",
                     bg=CARD, fg=MUTED, font=FONT_MAIN, width=8, anchor="w").grid(
                row=i + 2, column=0, sticky="w")
            row_e = []
            for j in range(self.num_consumers):
                e = self._make_entry()
                val = old[i][j] if (i < len(old) and j < len(old[i])) \
                    else str((i + 1) * 10 + j * 5)
                e.insert(0, val)
                e.grid(row=i + 2, column=j + 1, padx=3, pady=3)
                row_e.append(e)
            self.cost_entries.append(row_e)

        sr = self.num_suppliers + 3
        tk.Label(self.matrix_container, text="Запасы",
                 bg=CARD, fg=TEXT, font=FONT_BOLD, anchor="w").grid(
            row=sr, column=0, columnspan=self.num_consumers + 2,
            sticky="w", pady=(10, 4))
        old_s = self._saved_data.get('supply', [])
        for i in range(self.num_suppliers):
            e = self._make_entry()
            e.insert(0, old_s[i] if i < len(old_s) else "100")
            e.grid(row=sr + 1 + i // self.num_consumers,
                   column=1 + (i % self.num_consumers), padx=3, pady=3)
            self.supply_entries.append(e)

        dr = sr + 2 + (self.num_suppliers - 1) // self.num_consumers
        tk.Label(self.matrix_container, text="Спрос",
                 bg=CARD, fg=TEXT, font=FONT_BOLD, anchor="w").grid(
            row=dr, column=0, columnspan=self.num_consumers + 2,
            sticky="w", pady=(10, 4))
        old_d = self._saved_data.get('demand', [])
        for j in range(self.num_consumers):
            e = self._make_entry()
            e.insert(0, old_d[j] if j < len(old_d) else "100")
            e.grid(row=dr + 1, column=j + 1, padx=3, pady=3)
            self.demand_entries.append(e)

        # Сбрасываем скролл в начало
        self.matrix_scroll.canvas.xview_moveto(0)
        self.matrix_scroll.canvas.yview_moveto(0)
        self.update_idletasks()

    def _make_entry(self):
        return ctk.CTkEntry(
            self.matrix_container, width=60, height=30,
            corner_radius=8, fg_color="#F4F4F5",
            border_width=0, text_color=TEXT,
            justify="center", font=FONT_MAIN,
        )

    def reset_defaults(self):
        self._saved_data = {'costs': [], 'supply': [], 'demand': []}
        self.rebuild_matrix(preserve=False)

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

        total_s, total_d = float(np.sum(supply)), float(np.sum(demand))
        if abs(total_s - total_d) > 1e-6:
            if total_s < total_d:
                who = f"фиктивного поставщика с запасом {total_d - total_s:.0f}"
            else:
                who = f"фиктивного потребителя со спросом {total_s - total_d:.0f}"
            ok = messagebox.askyesno(
                "Несбалансированная задача",
                f"Σ запасов = {total_s:.0f}\n"
                f"Σ спроса  = {total_d:.0f}\n\n"
                f"Задача не сбалансирована. Добавить {who}?\n\n"
                f"«Нет» — отменить решение."
            )
            if not ok:
                return

        self.solve_btn.configure(text="Считаю…", state="disabled")
        self.update_idletasks()

        def run():
            costs_b, supply_b, demand_b, balanced = balance_task(costs, supply, demand)
            n_s, n_c = costs_b.shape
            c, A_eq, b_eq = build_lp_matrices(costs_b, supply_b, demand_b)

            self.after(0, self.log_panel.clear)
            if balanced:
                self.after(0, lambda: self.log_panel.log(
                    f"⚙ Задача была несбалансирована — добавлен фиктивный узел "
                    f"(размерность: {n_s}×{n_c})", "info"))

            result = solve_two_phase_simplex(c, A_eq, b_eq, self.log_panel.log)

            if result is None:
                self.after(0, lambda: self.log_panel.log("\n❌ Решение не найдено.", "error"))
                self.after(0, self._finish_solve)
                return

            x_opt, z_opt = result
            plan = x_opt.reshape((n_s, n_c))

            self.after(0, lambda: self._show_summary(plan, z_opt, supply_b, demand_b))
            self.after(0, lambda: self._draw_chart(plan))
            self.after(0, self._finish_solve)

        threading.Thread(target=run, daemon=True).start()

    def _show_summary(self, plan, z_opt, supply_b, demand_b):
        n_s, n_c = plan.shape
        self.log_panel.log("")
        self.log_panel.log("╔══════════════════════════════════════════════════════════╗", "header")
        self.log_panel.log("║              ОПТИМАЛЬНЫЙ ПЛАН ПОСТАВОК                   ║", "header")
        self.log_panel.log("╚══════════════════════════════════════════════════════════╝", "header")
        self.log_panel.log(f"z* = {z_opt:.2f} руб.", "success")
        self.log_panel.log("")

        header_line = "          " + "".join(f"{'П'+str(j+1):>9}" for j in range(n_c))
        self.log_panel.log(header_line, "info")
        for i in range(n_s):
            line = f"Пост.{i+1:<3} " + "".join(
                f"{int(round(plan[i][j])):>9}" for j in range(n_c))
            self.log_panel.log(line)

        self.log_panel.log("")
        self.log_panel.log("Проверка ограничений:", "info")
        for i in range(n_s):
            self.log_panel.log(
                f"  Отгрузки Пост.{i+1}: {int(round(np.sum(plan[i])))} / запас {int(supply_b[i])}")
        for j in range(n_c):
            self.log_panel.log(
                f"  Поставки П{j+1}: {int(round(np.sum(plan[:, j])))} / спрос {int(demand_b[j])}")

    def _finish_solve(self):
        self.solve_btn.configure(text="Решить задачу", state="normal")
        self._start_pulse()

    def _start_pulse(self):
        if self._pulse_id:
            self.after_cancel(self._pulse_id)
        self._pulse_step(0)

    def _pulse_step(self, step):
        if step > 6:
            self.solve_btn.configure(fg_color=ACCENT)
            return
        t = step / 6.0
        c1 = (99, 102, 241); c2 = (79, 70, 229)
        col = tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))
        self.solve_btn.configure(fg_color="#%02X%02X%02X" % col)
        self._pulse_id = self.after(70, lambda: self._pulse_step(step + 1))

    def _draw_chart(self, plan):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = render_chart(self.chart_frame, plan)