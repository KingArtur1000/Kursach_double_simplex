"""Палитра, шрифты, настройки внешнего вида."""
import customtkinter as ctk


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
FONT_TITLE = ("Segoe UI", 24, "bold")
FONT_SUB   = ("Segoe UI", 12)
FONT_MONO  = ("Consolas", 10)


def setup_appearance():
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")