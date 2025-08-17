# 可视化：使用 matplotlib + networkx，将结论节点着色与其他节点不同
# 说明：在保持原有思路（networkx + 分层布局 + 文本框节点）的基础上，
#       清理未使用的 import / 代码，移除表情符号，逻辑尽量简洁。

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple


def visualize_dag(root_node, filename: str = "reasoning_dag", format: str = "png",
                  figsize: Tuple[int, int] = (12, 8), dpi: int = 300, style: str = "modern") -> None:
    """将 DAGNode 结构可视化为图片。

    参数：
        root_node: DAG 的根节点（需包含 .children / .rule / .z3_expr 等常见属性）
        filename: 输出文件名（不含扩展名）
        format: 输出格式（png/pdf/svg/jpg）
        figsize: 图尺寸
        dpi: 分辨率
        style: 风格（"modern" | "classic" | "minimal"）
    """
    G, node_info = _build_networkx_graph(root_node)
    if G.number_of_nodes() == 0:
        print("[DAG 可视化] 没有可视化的节点")
        return

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if style == "classic":
        _draw_classic_style(G, node_info, ax)
    elif style == "minimal":
        _draw_minimal_style(G, node_info, ax)
    else:
        _draw_modern_style(G, node_info, ax)

    ax.set_title("Logical Reasoning DAG", fontsize=16, fontweight="bold", pad=16)
    ax.axis("off")
    plt.tight_layout()

    out_path = f"{filename}.{format}"
    plt.savefig(out_path, format=format, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[DAG 可视化] 已保存：{out_path}")


# ------------------------------
# 构图与文本处理
# ------------------------------

def _build_networkx_graph(root_node) -> Tuple[nx.DiGraph, Dict]:
    G = nx.DiGraph()
    node_info: Dict[int, Dict] = {}
    visited = set()

    def dfs(node, depth: int = 0):
        if node is None:
            return
        nid = id(node)
        if nid in visited:
            return
        visited.add(nid)

        rule_name = getattr(node, "rule_name", None) or (
            type(node.rule).__name__ if hasattr(node, "rule") and node.rule else "Unknown"
        )
        expr_str = str(getattr(node, "z3_expr", "Unknown"))
        display_expr = _simplify_expression(expr_str)

        G.add_node(nid)
        node_info[nid] = {
            "rule": rule_name,
            "expression": display_expr,
            "depth": depth,
            "full_expr": expr_str,
        }

        if getattr(node, "children", None):
            for child in node.children:
                if child is None:
                    continue
                cid = id(child)
                dfs(child, depth + 1)
                # 用子 -> 父 表示推理方向
                G.add_edge(cid, nid)

    dfs(root_node)
    return G, node_info


def _simplify_expression(s: str, max_len: int = 28) -> str:
    if not s:
        return ""
    s = s.replace("LogicVar_", "V").replace("_General", "").replace("_Premise", "").replace("_Conclusion", "")
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


# ------------------------------
# 三种绘制风格（结论节点：out_degree==0 高亮）
# ------------------------------

def _draw_modern_style(G: nx.DiGraph, node_info: Dict, ax) -> None:
    pos = _hierarchical_layout(G, node_info)
    colors = {
        "background": "#f8f9fa",
        "node_fill": "#e3f2fd",
        "node_border": "#1976d2",
        "edge": "#666666",
        "text": "#212121",
        # 结论节点颜色（与普通节点不同）
        "conclusion_fill": "#ffe082",
        "conclusion_border": "#ef6c00",
    }
    ax.set_facecolor(colors["background"])
    _draw_curved_edges(G, pos, ax, color=colors["edge"], alpha=0.7)

    for nid, (x, y) in pos.items():
        info = node_info[nid]
        is_final = G.out_degree(nid) == 0
        facec = colors["conclusion_fill"] if is_final else colors["node_fill"]
        edgec = colors["conclusion_border"] if is_final else colors["node_border"]
        lw = 2.2 if is_final else 1.8

        bbox = dict(boxstyle="round,pad=0.3", facecolor=facec, edgecolor=edgec, linewidth=lw, alpha=0.97)
        txt = f"{info['rule']}\n{info['expression']}"
        ax.text(x, y, txt, ha="center", va="center", fontsize=9, fontweight="bold", bbox=bbox, color=colors["text"])


def _draw_classic_style(G: nx.DiGraph, node_info: Dict, ax) -> None:
    pos = _hierarchical_layout(G, node_info)
    colors = {
        "node_fill": "#ffffff",
        "node_border": "#333333",
        "edge": "#000000",
        "text": "#000000",
        "conclusion_fill": "#fff3cd",
        "conclusion_border": "#cc9a06",
    }
    _draw_straight_edges(G, pos, ax, color=colors["edge"])
    for nid, (x, y) in pos.items():
        info = node_info[nid]
        is_final = G.out_degree(nid) == 0
        facec = colors["conclusion_fill"] if is_final else colors["node_fill"]
        edgec = colors["conclusion_border"] if is_final else colors["node_border"]
        lw = 2.0 if is_final else 1.5
        bbox = dict(boxstyle="square,pad=0.3", facecolor=facec, edgecolor=edgec, linewidth=lw, alpha=0.98)
        ax.text(x, y, f"{info['rule']}\n{info['expression']}", ha="center", va="center", fontsize=9, bbox=bbox, color=colors["text"])


def _draw_minimal_style(G: nx.DiGraph, node_info: Dict, ax) -> None:
    pos = _hierarchical_layout(G, node_info)
    colors = {
        "node_fill": "#f5f5f5",
        "node_border": "#888888",
        "edge": "#999999",
        "text": "#222222",
        "conclusion_fill": "#f8d7da",  # 淡红，醒目但不刺眼
        "conclusion_border": "#c82333",
    }
    _draw_straight_edges(G, pos, ax, color=colors["edge"], alpha=0.6, width=1.2)
    for nid, (x, y) in pos.items():
        info = node_info[nid]
        is_final = G.out_degree(nid) == 0
        facec = colors["conclusion_fill"] if is_final else colors["node_fill"]
        edgec = colors["conclusion_border"] if is_final else colors["node_border"]
        bbox = dict(boxstyle="round,pad=0.25", facecolor=facec, edgecolor=edgec, linewidth=1.6 if is_final else 1.2, alpha=0.96)
        ax.text(x, y, f"{info['rule']}\n{info['expression']}", ha="center", va="center", fontsize=9, bbox=bbox, color=colors["text"])


# ------------------------------
# 辅助：布局与连线
# ------------------------------

def _hierarchical_layout(G: nx.DiGraph, node_info: Dict) -> Dict[int, Tuple[float, float]]:
    """按 depth 分层，自上而下。"""
    layers: Dict[int, List[int]] = {}
    for nid, info in node_info.items():
        layers.setdefault(info["depth"], []).append(nid)

    y_gap = 1.2
    x_gap = 1.2
    pos: Dict[int, Tuple[float, float]] = {}

    for depth in sorted(layers.keys()):
        nodes = layers[depth]
        count = max(1, len(nodes))
        xs = np.linspace(0, (count - 1) * x_gap, count)
        x_center = (count - 1) * x_gap / 2
        for i, nid in enumerate(nodes):
            pos[nid] = (xs[i] - x_center, -depth * y_gap)
    return pos


def _draw_curved_edges(G: nx.DiGraph, pos: Dict[int, Tuple[float, float]], ax, color: str = "#666666", alpha: float = 0.8, width: float = 1.6) -> None:
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        # 简单二次贝塞尔曲线（轻微弯曲，避免重叠）
        ctrl = (mx + 0.2, my)
        t = np.linspace(0, 1, 60)
        xc = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * ctrl[0] + t ** 2 * x1
        yc = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * ctrl[1] + t ** 2 * y1
        ax.plot(xc, yc, color=color, alpha=alpha, linewidth=width)
        _add_arrow(ax, (x0, y0), (x1, y1), color=color, alpha=alpha)


def _draw_straight_edges(G: nx.DiGraph, pos: Dict[int, Tuple[float, float]], ax, color: str = "#000000", alpha: float = 1.0, width: float = 1.5) -> None:
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color=color, alpha=alpha, linewidth=width)
        _add_arrow(ax, (x0, y0), (x1, y1), color=color, alpha=alpha)


def _add_arrow(ax, start: Tuple[float, float], end: Tuple[float, float], color: str = "#000000", alpha: float = 1.0, size: float = 0.15) -> None:
    dx, dy = end[0] - start[0], end[1] - start[1]
    L = (dx ** 2 + dy ** 2) ** 0.5
    if L <= 0:
        return
    offset = 0.35
    sx = start[0] + dx * offset / L
    sy = start[1] + dy * offset / L
    ex = end[0] - dx * offset / L
    ey = end[1] - dy * offset / L
    ax.annotate('', xy=(ex, ey), xytext=(sx, sy), arrowprops=dict(arrowstyle='->', color=color, alpha=alpha, lw=1.5, shrinkA=0, shrinkB=0))


# 可选：导出文本信息（若外部需要，可保留；不需要可自行删除）
def save_dag_info(root_node, filename: str = "dag_info.txt") -> None:
    lines: List[str] = []
    visited = set()

    def walk(n, depth=0):
        if n is None or id(n) in visited:
            return
        visited.add(id(n))
        indent = '  ' * depth
        rule = getattr(n, 'rule_name', 'Unknown')
        expr = str(getattr(n, 'z3_expr', 'Unknown'))
        lines.append(f"{indent}[depth {depth}] {rule}: {expr}")
        for c in getattr(n, 'children', []) or []:
            walk(c, depth + 1)

    walk(root_node)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))