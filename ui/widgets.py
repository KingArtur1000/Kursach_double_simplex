"""Вспомогательные виджеты (Card и др.)."""
import customtkinter as ctk
from .theme import CARD, BORDER, TEXT, FONT_BOLD


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