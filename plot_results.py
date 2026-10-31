"""绘制路径、剪枝过程与决策对照图。"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from cases import MAZE, WEIGHTED, TREASURE_MAP


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
BLUE, ORANGE, GRAY = "#236b8e", "#b56a35", "#7d8790"


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=190, bbox_inches="tight")
    plt.close(fig)


def draw_grid(ax, grid, path=None, title=""):
    rows, cols = len(grid), len(grid[0])
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            blocked = cell == "X"
            value = float(cell) if isinstance(cell, (float, int)) else 1
            color = "#33414c" if blocked else "#f7f9fa" if value <= 1 else "#e6c4a9" if value <= 3 else "#c98c67"
            ax.add_patch(Rectangle((c-.5, r-.5), 1, 1, facecolor=color, edgecolor="white", linewidth=1))
            if not blocked and cell != ".":
                ax.text(c, r, str(cell), ha="center", va="center", fontsize=9, zorder=4)
    if path:
        xs, ys = [p[1] for p in path], [p[0] for p in path]
        ax.plot(xs, ys, color=BLUE, lw=2, alpha=.65, zorder=2)
        for a, b in zip(path, path[1:]):
            ax.annotate("", xy=(b[1], b[0]), xytext=(a[1], a[0]),
                        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2, shrinkA=8, shrinkB=8), zorder=3)
    ax.set(xlim=(-.55, cols-.45), ylim=(rows-.45, -.55), aspect="equal")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.tick_params(length=0, labelsize=8)
    ax.set_xlabel("column", fontsize=8)
    ax.set_ylabel("row", fontsize=8)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def paths():
    traces = json.loads((RESULTS / "path_traces.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.55), constrained_layout=True)
    draw_grid(axes[0], MAZE, traces["q1_safe_larger_g"]["path"], "a  Unit-cost maze: cost 7")
    draw_grid(axes[1], WEIGHTED, traces["q2_safe_larger_g"]["path"], "b  Weighted terrain: cost 7, E = 0")
    save(fig, "given_paths")
    frame = pd.read_csv(RESULTS / "path_benchmarks.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.4), constrained_layout=True)
    for mode, tie, color, style, label in [
        ("zero", "fifo", GRAY, "-", "Dijkstra / FIFO"),
        ("safe", "fifo", ORANGE, "--", "A* / FIFO"),
        ("safe", "larger_g", BLUE, "-", "A* / larger g")
    ]:
        sub = frame[(frame.family == "open") & (frame.heuristic == mode) & (frame.tie == tie)].sort_values("size")
        axes[0].plot(sub["size"], sub.expanded, marker="o", color=color, linestyle=style, label=label, ms=4)
    axes[0].set(xlabel="Grid side length", ylabel="Expanded nodes", yscale="log", xticks=[20,40,80])
    axes[0].set_title("a  Open grids", loc="left", fontweight="bold")
    axes[0].legend(fontsize=7)
    blocked = frame[frame.name == "blocked_20"]
    groups = [("zero","fifo"),("safe","fifo"),("safe","larger_g")]
    selected = [blocked[(blocked.heuristic==h)&(blocked.tie==t)].iloc[0] for h,t in groups]
    x=np.arange(3)
    axes[1].bar(x-.16,[r.expanded for r in selected],.32,color=GRAY,label="Expanded")
    axes[1].bar(x+.16,[r.peak_open for r in selected],.32,color=BLUE,label="Peak heap")
    axes[1].set_xticks(x,["Dijkstra", "A* FIFO", "A* larger g"])
    axes[1].set_ylabel("Entries / nodes")
    axes[1].set_title("b  Unreachable goal",loc="left",fontweight="bold")
    axes[1].legend(fontsize=7)
    save(fig, "path_scaling")


def tree():
    fig, ax = plt.subplots(figsize=(7.0, 2.75), constrained_layout=True)
    nodes = {"r":(.5,3,"MAX\n5"), "a":(.25,2,"MIN\n5"), "b":(.75,2,"MIN\n2"),
             "a0":(.125,1,"MAX\n5"), "a1":(.375,1,"MAX\n>= 6"),
             "b0":(.625,1,"MAX\n9"), "b1":(.875,1,"MAX\n2")}
    edges = [("r","a"),("r","b"),("a","a0"),("a","a1"),("b","b0"),("b","b1")]
    for parent, child in edges:
        ax.plot([nodes[parent][0],nodes[child][0]],[nodes[parent][1]-.15,nodes[child][1]+.18],color=GRAY,lw=1)
    for x,y,label in nodes.values():
        ax.text(x,y,label,ha="center",va="center",fontsize=9,
                bbox=dict(boxstyle="round,pad=.25",fc="white",ec=BLUE,lw=.8),zorder=3)
    leaves = [3,5,6,9,6,9,1,2]
    for i,value in enumerate(leaves):
        x=.0625+i*.125
        parent=["a0","a1","b0","b1"][i//2]
        pruned=i==3
        ax.plot([nodes[parent][0],x],[.82,.15],color=ORANGE if pruned else GRAY,
                linestyle="--" if pruned else "-",lw=1)
        ax.text(x,0,str(value),ha="center",va="center",fontsize=11,color=ORANGE if pruned else "black")
        if pruned:
            ax.plot([x-.02,x+.02],[-.12,.12],color=ORANGE,lw=2)
            ax.plot([x-.02,x+.02],[.12,-.12],color=ORANGE,lw=2)
    ax.text(.37,2.25,"beta = 5",fontsize=8,color=ORANGE)
    ax.text(.40,.48,"6 >= beta\nprune 9",fontsize=8,color=ORANGE)
    ax.set(xlim=(0,1),ylim=(-.3,3.35))
    ax.axis("off")
    save(fig,"tree_pruning")

    data=pd.read_csv(RESULTS/"tree_benchmarks.csv")
    fig,axes=plt.subplots(1,2,figsize=(6.7,2.45),constrained_layout=True)
    for ax,seed in zip(axes,[66,88]):
        sub=data[(data.seed==seed)&(data.branching==3)&(data.depth==6)]
        configs=[("minimax","natural"),("alpha_beta","natural"),("alpha_beta","reverse"),("alpha_beta","random")]
        chosen=[sub[(sub.algorithm==a)&(sub.order==o)].iloc[0] for a,o in configs]
        bars=ax.bar(range(4),[r.visited for r in chosen],color=[GRAY,BLUE,BLUE,ORANGE],width=.65)
        for bar,r in zip(bars,chosen):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+20,str(int(r.visited)),ha="center",fontsize=8)
        ax.set_xticks(range(4),["Minimax","AB natural","AB reverse","AB random"],rotation=16,ha="right")
        ax.set_ylabel("Visited nodes")
        ax.set_ylim(0,1250)
        ax.set_title(f"Seed {seed}: branching 3, depth 6",loc="left",fontsize=9)
    save(fig,"tree_order")


def treasure():
    fig,ax=plt.subplots(figsize=(3.1,3.0),constrained_layout=True)
    draw_grid(ax,TREASURE_MAP,title="Treasure map")
    save(fig,"treasure_map")
    frame=pd.read_csv(RESULTS/"decision_benchmarks.csv")
    data=frame[frame.name!="given"].pivot(index="name",columns="method",values="regret")
    methods=["greedy","H2_current","H2_opportunity","H4_full"]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = [name for name in
        ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"] if name in available_fonts]
    fig,axes=plt.subplots(1,2,figsize=(6.7,2.45),constrained_layout=True)
    for i,method in enumerate(methods):
        y=data[method].to_numpy()
        offsets=np.linspace(-.16,.16,len(y))
        axes[0].scatter(i+offsets,y,s=12,color=BLUE if method=="H2_opportunity" else GRAY,alpha=.7)
        axes[0].plot([i-.2,i+.2],[y.mean(),y.mean()],color=ORANGE,lw=2)
    axes[0].set_xticks(range(4),["即时贪心","原始两层","机会两层","完整四层"])
    axes[0].set_ylabel("首步损失（越低越好）")
    axes[0].set_title("（a）20 张地图；橙线为平均损失",loc="left",fontsize=9)
    delta=data.H2_current-data.H2_opportunity
    counts=[int((delta>0).sum()),int((delta==0).sum()),int((delta<0).sum())]
    bars=axes[1].bar(range(3),counts,color=[BLUE,GRAY,ORANGE],width=.6)
    axes[1].set_xticks(range(3),["损失减少","损失相同","损失增加"])
    axes[1].set_ylabel("地图数量")
    axes[1].set_ylim(0,max(counts)+3)
    axes[1].set_title("（b）机会两层与原始两层逐图比较",loc="left",fontsize=9)
    for bar,n in zip(bars,counts):
        axes[1].text(bar.get_x()+bar.get_width()/2,n+.25,str(n),ha="center")
    save(fig,"decision_quality")


def main():
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "axes.linewidth":.7,"pdf.fonttype":42,"svg.fonttype":"none"})
    paths()
    tree()
    treasure()


if __name__ == "__main__":
    main()
