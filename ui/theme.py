"""Палитра, шрифты, настройки внешнего вида."""
import customtkinter as ctk


LIGHT = {
    "bg":         "#FAFAFA",
    "card":       "#FFFFFF",
    "border":     "#E4E4E7",
    "text":       "#18181B",
    "muted":      "#71717A",
    "accent":     "#6366F1",
    "accent_hov": "#4F46E5",
    "success":    "#10B981",
    "danger":     "#EF4444",
    "warning":    "#F59E0B",
    "phase1":     "#8B5CF6",
    "phase2":     "#0EA5E9",
    "entry":      "#F4F4F5",
    "entry_btn":  "#E4E4E7",
    "hover":      "#F4F4F5",
}

DARK = {
    "bg":         "#18181B",
    "card":       "#27272A",
    "border":     "#3F3F46",
    "text":       "#FAFAFA",
    "muted":      "#A1A1AA",
    "accent":     "#818CF8",
    "accent_hov": "#6366F1",
    "success":    "#34D399",
    "danger":     "#F87171",
    "warning":    "#FBBF24",
    "phase1":     "#A78BFA",
    "phase2":     "#38BDF8",
    "entry":      "#3F3F46",
    "entry_btn":  "#52525B",
    "hover":      "#3F3F46",
}


class _Theme:
    """Всегда актуальная палитра. Обращение: T.card, T.text и т.д."""
    def __getattr__(self, name):
        mode = ctk.get_appearance_mode().lower()
        palette = DARK if mode == "dark" else LIGHT
        return palette[name]


T = _Theme()

FONT_MAIN = ("Segoe UI", 11)
FONT_BOLD = ("Segoe UI", 11, "bold")
FONT_MONO = ("Consolas", 10)


def setup_appearance(mode="light"):
    ctk.set_appearance_mode(mode)
    ctk.set_default_color_theme("blue")


def is_dark() -> bool:
    return ctk.get_appearance_mode().lower() == "dark"


def set_mode(mode: str):
    ctk.set_appearance_mode(mode)