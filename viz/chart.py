"""Построение карты оптимальных поставок."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ui.theme import T, is_dark


def render_chart(parent_frame, plan: np.ndarray) -> FigureCanvasTkAgg:
    n_s, n_c = plan.shape

    # ═══ Полностью тёмный/светлый Figure под тему ═══
    fig = Figure(figsize=(6.5, 3.5), dpi=100, facecolor=T.card)
    ax = fig.add_subplot(111)
    ax.set_facecolor(T.card)

    # Кастомная палитра: от цвета карточки к акценту
    cmap = LinearSegmentedColormap.from_list(
        "theme_chart",
        [T.card, T.accent],
        N=256,
    )

    vmax = plan.max() if plan.max() > 0 else 1
    im = ax.imshow(plan, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)

    # ─── Оси и подписи ───
    ax.set_xticks(range(n_c))
    ax.set_xticklabels([f"Потр.{j+1}" for j in range(n_c)], color=T.text)
    ax.set_yticks(range(n_s))
    ax.set_yticklabels([f"Пост.{i+1}" for i in range(n_s)], color=T.text)

    ax.set_title("Карта оптимальных поставок", color=T.text, fontsize=11, pad=10)

    for spine in ax.spines.values():
        spine.set_color(T.border)

    ax.tick_params(axis="x", colors=T.text)
    ax.tick_params(axis="y", colors=T.text)
    ax.set_xticks(np.arange(-0.5, n_c, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_s, 1), minor=True)
    ax.grid(which="minor", color=T.border, linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    # ─── Цвет чисел в ячейках под тему ───
    # Высокие значения → яркая заливка (accent) → текст тёмный
    # Низкие значения → заливка = цвет карточки → текст цвета T.text
    threshold = vmax * 0.5
    for i in range(n_s):
        for j in range(n_c):
            val = int(round(plan[i][j]))
            if val > threshold:
                color = "#18181B" if is_dark() else "white"
            else:
                color = T.text
            ax.text(j, i, str(val), ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

    # ─── Colorbar ───
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Объём поставки", color=T.muted)
    cb.ax.tick_params(colors=T.muted)
    cb.outline.set_edgecolor(T.border)
    cb.outline.set_linewidth(1)

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return canvas