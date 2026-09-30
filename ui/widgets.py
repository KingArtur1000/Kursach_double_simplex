"""Вспомогательные виджеты: карточка и журнал с зумом."""
import tkinter as tk
import customtkinter as ctk

from .theme import (
    CARD, BORDER, TEXT, MUTED, ACCENT, FONT_MAIN, FONT_BOLD, FONT_MONO,
    SUCCESS, DANGER, WARNING, PHASE1, PHASE2,
)


class Card(ctk.CTkFrame):
    """Скруглённая карточка с опциональным заголовком."""

    def __init__(self, master, title: str = None, **kw):
        super().__init__(
            master,
            fg_color=CARD,
            corner_radius=14,
            border_width=1,
            border_color=BORDER,
            **kw
        )
        if title:
            ctk.CTkLabel(
                self, text=title,
                font=FONT_BOLD, text_color=TEXT, anchor="w"
            ).pack(fill="x", padx=16, pady=(14, 6))


class LogPanel(ctk.CTkFrame):
    """
    Карточка-журнал с поддержкой масштаба:
      • Ctrl + колёсико мыши  — плавный зум
      • Кнопки  − / + / ⟲     — в шапке карточки
      • Ctrl+= / Ctrl+- / Ctrl+0 — горячие клавиши
    """

    MIN_SIZE = 8
    MAX_SIZE = 26
    DEFAULT_SIZE = 10

    TAG_STYLES = {
        "header":    (ACCENT,   True),
        "phase1":    (PHASE1,   True),
        "phase2":    (PHASE2,   True),
        "iteration": (WARNING,  False),
        "info":      (PHASE2,   False),
        "success":   (SUCCESS,  True),
        "error":     (DANGER,   True),
    }

    def __init__(self, master, title: str = "Журнал решения", **kw):
        super().__init__(
            master,
            fg_color=CARD,
            corner_radius=14,
            border_width=1,
            border_color=BORDER,
            **kw
        )
        self._font_size = self.DEFAULT_SIZE

        # ─── Шапка с заголовком и кнопками зума ───
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(14, 6))

        ctk.CTkLabel(
            header, text=title,
            font=FONT_BOLD, text_color=TEXT, anchor="w"
        ).pack(side="left")

        controls = ctk.CTkFrame(header, fg_color="transparent")
        controls.pack(side="right")

        self.zoom_label = ctk.CTkLabel(
            controls, text="100%",
            font=("Segoe UI", 10),
            text_color=MUTED, width=42
        )
        self.zoom_label.pack(side="left", padx=(0, 6))

        self._make_btn(controls, "−", self._zoom_out, tooltip="Уменьшить")
        self._make_btn(controls, "+", self._zoom_in,  tooltip="Увеличить")
        self._make_btn(controls, "⟲", self._zoom_reset, tooltip="Сброс (Ctrl+0)")

        # ─── Тело: Text + скроллбар ───
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=(10, 0), pady=(0, 12))

        self.text = tk.Text(
            body, font=("Consolas", self._font_size),
            wrap="word", bg=CARD, fg=TEXT,
            relief="flat", bd=0, highlightthickness=0,
            padx=14, pady=4, insertbackground=ACCENT,
        )
        scroll = ctk.CTkScrollbar(body, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)

        scroll.pack(side="right", fill="y", padx=(0, 10))
        self.text.pack(side="left", fill="both", expand=True, padx=(10, 0))

        # Применяем теги для цветов
        self._apply_tags()

        # ─── Привязки клавиш и колеса мыши ───
        self.text.bind("<Control-MouseWheel>", self._on_wheel)   # Windows / macOS
        self.text.bind("<Control-Button-4>",   lambda e: self._zoom_in())   # Linux
        self.text.bind("<Control-Button-5>",   lambda e: self._zoom_out())  # Linux
        self.text.bind("<Control-plus>",       lambda e: self._zoom_in())
        self.text.bind("<Control-equal>",      lambda e: self._zoom_in())
        self.text.bind("<Control-minus>",      lambda e: self._zoom_out())
        self.text.bind("<Control-Key-0>",      lambda e: self._zoom_reset())

    # ─── Кнопки ───
    def _make_btn(self, parent, label, cmd, tooltip=""):
        b = ctk.CTkButton(
            parent, text=label, width=28, height=24,
            fg_color="transparent", hover_color="#F4F4F5",
            text_color=MUTED, font=("Segoe UI", 14, "bold"),
            corner_radius=6, command=cmd,
        )
        b.pack(side="left", padx=2)
        return b

    # ─── Логика зума ───
    def _on_wheel(self, event):
        # delta > 0 — вверх (увеличить), delta < 0 — вниз (уменьшить)
        if event.delta > 0:
            self._zoom_in()
        else:
            self._zoom_out()
        return "break"  # не пропускаем событие дальше (чтобы не скроллилось)

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
        for tag, (color, bold) in self.TAG_STYLES.items():
            self.text.tag_config(
                tag,
                foreground=color,
                font=("Consolas", self._font_size, "bold") if bold
                     else ("Consolas", self._font_size),
            )

    # ─── Публичный API ───
    def log(self, msg: str, tag: str = None):
        if tag:
            self.text.insert(tk.END, msg + "\n", tag)
        else:
            self.text.insert(tk.END, msg + "\n")
        self.text.see(tk.END)

    def clear(self):
        self.text.delete("1.0", tk.END)

    @property
    def font_size(self) -> int:
        return self._font_size