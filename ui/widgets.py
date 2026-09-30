"""Вспомогательные виджеты: карточка и журнал с зумом."""
import tkinter as tk
import customtkinter as ctk

from .theme import T, FONT_BOLD


class Card(ctk.CTkFrame):
    """Скруглённая карточка с опциональным заголовком."""

    def __init__(self, master, title: str = None, **kw):
        super().__init__(
            master,
            fg_color=T.card,
            corner_radius=14,
            border_width=1,
            border_color=T.border,
            **kw,
        )
        if title:
            ctk.CTkLabel(
                self, text=title,
                font=FONT_BOLD, text_color=T.text, anchor="w",
            ).pack(fill="x", padx=16, pady=(14, 6))


class LogPanel(ctk.CTkFrame):
    """Карточка-журнал с поддержкой масштаба."""

    MIN_SIZE = 8
    MAX_SIZE = 26
    DEFAULT_SIZE = 10

    def __init__(self, master, title: str = "Журнал решения", **kw):
        super().__init__(
            master,
            fg_color=T.card,
            corner_radius=14,
            border_width=1,
            border_color=T.border,
            **kw,
        )
        self._font_size = self.DEFAULT_SIZE

        # ─── Шапка ───
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(14, 6))

        ctk.CTkLabel(
            header, text=title,
            font=FONT_BOLD, text_color=T.text, anchor="w",
        ).pack(side="left")

        controls = ctk.CTkFrame(header, fg_color="transparent")
        controls.pack(side="right")

        self.zoom_label = ctk.CTkLabel(
            controls, text="100%",
            font=("Segoe UI", 10),
            text_color=T.muted, width=42,
        )
        self.zoom_label.pack(side="left", padx=(0, 6))

        # Кнопки зума (↺ — глиф, который есть во всех системных шрифтах)
        self._make_btn(controls, "−", self._zoom_out)
        self._make_btn(controls, "+", self._zoom_in)
        self._make_btn(controls, "↺", self._zoom_reset)

        # ─── Тело ───
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=(10, 0), pady=(0, 12))

        self.text = tk.Text(
            body, font=("Consolas", self._font_size),
            wrap="word", bg=T.card, fg=T.text,
            relief="flat", bd=0, highlightthickness=0,
            padx=14, pady=4, insertbackground=T.accent,
        )
        scroll = ctk.CTkScrollbar(
            body, command=self.text.yview,
            button_color=T.entry_btn,
            button_hover_color=T.accent,
            fg_color=T.card,
        )
        self.text.configure(yscrollcommand=scroll.set)

        scroll.pack(side="right", fill="y", padx=(0, 10))
        self.text.pack(side="left", fill="both", expand=True, padx=(10, 0))

        self._apply_tags()

        # Горячие клавиши / колёсико
        self.text.bind("<Control-MouseWheel>", self._on_wheel)
        self.text.bind("<Control-Button-4>",   lambda e: self._zoom_in())
        self.text.bind("<Control-Button-5>",   lambda e: self._zoom_out())
        self.text.bind("<Control-plus>",       lambda e: self._zoom_in())
        self.text.bind("<Control-equal>",      lambda e: self._zoom_in())
        self.text.bind("<Control-minus>",      lambda e: self._zoom_out())
        self.text.bind("<Control-Key-0>",      lambda e: self._zoom_reset())

    # ─── Публичное свойство для сохранения зума при смене темы ───
    @property
    def font_size(self) -> int:
        return self._font_size

    def _make_btn(self, parent, label, cmd):
        return ctk.CTkButton(
            parent, text=label, width=28, height=24,
            fg_color="transparent", hover_color=T.hover,
            text_color=T.muted, font=("Segoe UI", 14, "bold"),
            corner_radius=6, command=cmd,
        ).pack(side="left", padx=2)

    # ─── Зум ───
    def _on_wheel(self, event):
        if event.delta > 0:
            self._zoom_in()
        else:
            self._zoom_out()
        return "break"

    def _zoom_in(self):
        self._set_size(self._font_size + 1)

    def _zoom_out(self):
        self._set_size(self._font_size - 1)

    def _zoom_reset(self):
        self._set_size(self.DEFAULT_SIZE)

    def _set_size(self, size: int):
        size = max(self.MIN_SIZE, min(self.MAX_SIZE, size))
        if size == self._font_size:
            return
        self._font_size = size
        self.text.configure(font=("Consolas", self._font_size))
        self._apply_tags()
        pct = int(round(self._font_size / self.DEFAULT_SIZE * 100))
        self.zoom_label.configure(text=f"{pct}%")

    def _apply_tags(self):
        styles = {
            "header":    (T.accent,  True),
            "phase1":    (T.phase1,  True),
            "phase2":    (T.phase2,  True),
            "iteration": (T.warning, False),
            "info":      (T.phase2,  False),
            "success":   (T.success, True),
            "error":     (T.danger,  True),
        }
        for tag, (color, bold) in styles.items():
            self.text.tag_config(
                tag,
                foreground=color,
                font=("Consolas", self._font_size, "bold") if bold
                     else ("Consolas", self._font_size),
            )

    # ─── Public API ───
    def log(self, msg: str, tag: str = None):
        if tag:
            self.text.insert(tk.END, msg + "\n", tag)
        else:
            self.text.insert(tk.END, msg + "\n")
        self.text.see(tk.END)

    def clear(self):
        self.text.delete("1.0", tk.END)