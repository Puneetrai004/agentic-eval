# utils/visuals.py
import math
import matplotlib.pyplot as plt

def spider_net_multi(labels, rows, title="Spider (Radar) Chart", fill_alpha=0.15, show=False):
    """
    Draws a radar/spider chart for multiple rows.
    - labels: list of category names (length N)
    - rows: list of dicts with values and optional name: {"name": "Agent 1","values":[...]}
    - returns matplotlib fig, ax
    """
    N = len(labels)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    # radial limits 0-5
    max_val = 5
    step = 1
    ax.set_yticks(range(0, int(max_val) + step, step))
    ax.set_ylim(0, max_val)
    for r in rows:
        values = r["values"] + r["values"][:1]
        ax.plot(angles, values, linewidth=1, label=r.get("name", ""))
        ax.fill(angles, values, alpha=fill_alpha)
    ax.set_title(title, va='bottom')
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    if show:
        plt.show()
    return fig, ax
