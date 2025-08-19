import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import re
from typing import Dict, List, Tuple, Optional


def visualize_dag(
    root_node,
    filename: str = "reasoning_dag",
    format: str = "png",
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
    style: str = "modern",
    var_map: Optional[Dict[str, str]] = None,
) -> None:
    """将 DAGNode 结构可视化为图片。

    参数：
        root_node: DAG 的根节点（需包含 .children / .rule / .z3_expr 等常见属性）
        filename: 输出文件名（不含扩展名）
        format: 输出格式（png/pdf/svg/jpg）
        figsize: 图尺寸
        dpi: 分辨率
        style: 风格（"modern" | "classic" | "minimal"）
        var_map: 可选，V* → 语义变量名的映射（例如 {"V1":"completed_coursework", ...}）。
                 若提供，将用于把 Unknown 节点替换为可读的“派生/中间”标签。
    """
    G, node_info = _build_networkx_graph(root_node, var_map)
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

def _build_networkx_graph(root_node, var_map: Optional[Dict[str, str]] = None) -> Tuple[nx.DiGraph, Dict]:
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
            type(node.rule).__name__ if hasattr(node, "rule") and node.rule else "Derived"
        )
        # 原始表达式
        expr_str = str(getattr(node, "z3_expr", ""))

        # 清理 + Unknown 替换
        display_expr, is_derived = _prettify_expression(expr_str, var_map)

        G.add_node(nid)
        node_info[nid] = {
            "rule": rule_name if rule_name != "Unknown" else "Derived",
            "expression": display_expr,
            "depth": depth,
            "full_expr": expr_str,
            "derived": is_derived or (rule_name in {"Unknown", None}),
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


def _prettify_expression(s: str, var_map: Optional[Dict[str, str]] = None, max_len: int = 28) -> Tuple[str, bool]:
    """将表达式转为可读文本，并识别是否为“派生/中间”节点。
    返回 (显示字符串, 是否派生)
    规则：
      - "Unknown V5" → "qualified_for_degree (derived)"
      - "Unknown Implies(V2, V3)" → "Derived: Implies(passed_examinations, submitted_thesis)"
      - 其余表达式：替换其中的 V* 为语义名；如找不到映射则保留。
    """
    if not s:
        return "", True

    is_derived = False
    text = s

    # 常规清理（便于阅读）
    text = text.replace("LogicVar_", "V").replace("_General", "").replace("_Premise", "").replace("_Conclusion", "")

    # Unknown 规则 1：纯变量
    m = re.fullmatch(r"Unknown\s+(V\d+)", text)
    if m:
        v = m.group(1)
        pretty = (var_map or {}).get(v, v) + " (derived)"
        return _truncate(pretty, max_len), True

    # Unknown 规则 2：带 Implies(...)
    m = re.fullmatch(r"Unknown\s+Implies\((.+)\)", text)
    if m:
        body = m.group(1)
        pretty_body = _replace_vars(body, var_map)
        pretty = f"Derived: Implies({pretty_body})"
        return _truncate(pretty, max_len), True

    # 其他：仅做变量替换
    text2 = _replace_vars(text, var_map)
    # 若原串含 Unknown，则仍视为派生
    is_derived = ("Unknown" in s)
    return _truncate(text2, max_len), is_derived


def _replace_vars(text: str, var_map: Optional[Dict[str, str]] = None) -> str:
    if not var_map:
        return text
    return re.sub(r"V\d+", lambda mv: var_map.get(mv.group(0), mv.group(0)), text)


def _truncate(s: str, max_len: int) -> str:
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


# ------------------------------
# 三种绘制风格（结论节点：out_degree==0 高亮；派生节点虚线）
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
        # 派生节点边框（虚线）
        "derived_border": "#6c757d",
    }
    ax.set_facecolor(colors["background"])
    _draw_curved_edges(G, pos, ax, color=colors["edge"], alpha=0.7)

    for nid, (x, y) in pos.items():
        info = node_info[nid]
        is_final = G.out_degree(nid) == 0
        is_derived = bool(info.get("derived"))
        facec = colors["conclusion_fill"] if is_final else colors["node_fill"]
        edgec = colors["conclusion_border"] if is_final else (colors["derived_border"] if is_derived else colors["node_border"])
        lw = 2.2 if is_final else 1.8
        boxstyle = "round,pad=0.3"
        bbox = dict(boxstyle=boxstyle, facecolor=facec, edgecolor=edgec, linewidth=lw, alpha=0.97)
        txt = f"{info['rule']}\n{info['expression']}"
        t = ax.text(x, y, txt, ha="center", va="center", fontsize=9, fontweight="bold", bbox=bbox, color=colors["text"])
        if is_derived and not is_final:
            # 用虚线表示派生/中间节点
            t.set_path_effects([])
            # 通过在文字周围再画一个近似虚线框来表达（matplotlib 文本框不直接支持虚线）
            ax.add_patch(plt.Rectangle((x-1.2, y-0.3), 2.4, 0.6, fill=False, lw=1.4, ls="--", ec=colors["derived_border"], alpha=0.9))


def _draw_classic_style(G: nx.DiGraph, node_info: Dict, ax) -> None:
    pos = _hierarchical_layout(G, node_info)
    colors = {
        "node_fill": "#ffffff",
        "node_border": "#333333",
        "edge": "#000000",
        "text": "#000000",
        "conclusion_fill": "#fff3cd",
        "conclusion_border": "#cc9a06",
        "derived_border": "#555555",
    }
    _draw_straight_edges(G, pos, ax, color=colors["edge"])
    for nid, (x, y) in pos.items():
        info = node_info[nid]
        is_final = G.out_degree(nid) == 0
        is_derived = bool(info.get("derived"))
        facec = colors["conclusion_fill"] if is_final else colors["node_fill"]
        edgec = colors["conclusion_border"] if is_final else (colors["derived_border"] if is_derived else colors["node_border"])
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
        "conclusion_fill": "#f8d7da",
        "conclusion_border": "#c82333",
        "derived_border": "#666666",
    }
    _draw_straight_edges(G, pos, ax, color=colors["edge"], alpha=0.6, width=1.2)
    for nid, (x, y) in pos.items():
        info = node_info[nid]
        is_final = G.out_degree(nid) == 0
        is_derived = bool(info.get("derived"))
        facec = colors["conclusion_fill"] if is_final else colors["node_fill"]
        edgec = colors["conclusion_border"] if is_final else (colors["derived_border"] if is_derived else colors["node_border"])
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


# 可选：导出文本信息
def save_dag_info(root_node, filename: str = "dag_info.txt", var_map: Optional[Dict[str, str]] = None) -> None:
    lines: List[str] = []
    visited = set()

    def walk(n, depth=0):
        if n is None or id(n) in visited:
            return
        visited.add(id(n))
        indent = '  ' * depth
        rule = getattr(n, 'rule_name', 'Derived')
        expr = str(getattr(n, 'z3_expr', ''))
        expr_pretty, _ = _prettify_expression(expr, var_map)
        lines.append(f"{indent}[depth {depth}] {rule}: {expr_pretty}")
        for c in getattr(n, 'children', []) or []:
            walk(c, depth + 1)

    walk(root_node)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))