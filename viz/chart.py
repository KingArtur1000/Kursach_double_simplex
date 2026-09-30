"""Построение карты оптимальных поставок."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ui.theme import CARD, TEXT, MUTED, BORDER


def render_chart(parent_frame, plan: np.ndarray) -> FigureCanvasTkAgg:
    """Рисует heatmap поставок и встраивает в parent_frame."""
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

    canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return canvas