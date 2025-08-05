# 文件：dag/visualizer.py
# 功能：将 DAGNode 结构可视化，使用 matplotlib + networkx 替代 Graphviz
# 优势：安装简单，兼容性好，图形美观，支持多种输出格式

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path


def visualize_dag(root_node, filename="reasoning_dag", format="png",
                  figsize=(12, 8), dpi=300, style="modern"):
    """
    将 DAGNode 结构输出为可视化图片

    参数：
        root_node: DAGNode，DAG 的根节点
        filename: str，输出文件名（不含扩展名）
        format: str，输出格式 (png, pdf, svg, jpg)
        figsize: tuple，图片尺寸
        dpi: int，图片分辨率
        style: str，可视化风格 ("modern", "classic", "minimal")
    """
    try:
        # 构建 NetworkX 图
        G, node_info = _build_networkx_graph(root_node)

        if len(G.nodes()) == 0:
            print(f"[⚠️  DAG 可视化警告] 没有找到有效节点")
            return

        # 设置图形样式
        plt.style.use('default')  # 确保使用默认样式
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # 根据风格选择布局和颜色
        if style == "modern":
            _draw_modern_style(G, node_info, ax)
        elif style == "classic":
            _draw_classic_style(G, node_info, ax)
        else:  # minimal
            _draw_minimal_style(G, node_info, ax)

        # 设置标题和布局
        ax.set_title("Logical Reasoning DAG", fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')  # 隐藏坐标轴

        # 紧凑布局
        plt.tight_layout()

        # 保存文件
        output_path = f"{filename}.{format}"
        plt.savefig(output_path, format=format, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()  # 释放内存

        print(f"[✅ DAG 可视化输出] 已保存为 {output_path}")

    except Exception as e:
        print(f"[❌ DAG 可视化失败] {e}")
        # 确保释放matplotlib资源
        plt.close('all')


def _build_networkx_graph(root_node) -> Tuple[nx.DiGraph, Dict]:
    """构建 NetworkX 有向图"""
    G = nx.DiGraph()
    node_info = {}
    visited = set()

    def traverse_node(node, depth=0):
        if node is None:
            return

        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        # 准备节点信息
        rule_name = getattr(node, 'rule_name', None) or \
                    (type(node.rule).__name__ if hasattr(node, 'rule') and node.rule else "Unknown")

        expr_str = str(getattr(node, 'z3_expr', 'Unknown'))

        # 简化表达式显示
        display_expr = _simplify_expression(expr_str)

        # 添加节点到图中
        G.add_node(node_id)
        node_info[node_id] = {
            'rule': rule_name,
            'expression': display_expr,
            'depth': depth,
            'full_expr': expr_str
        }

        # 处理子节点
        if hasattr(node, 'children') and node.children:
            for child in node.children:
                if child is not None:
                    child_id = id(child)
                    traverse_node(child, depth + 1)
                    # 在 DAG 中，边从子节点指向父节点（表示推理方向）
                    G.add_edge(child_id, node_id)

    traverse_node(root_node)
    return G, node_info


def _simplify_expression(expr_str: str, max_length: int = 25) -> str:
    """简化表达式显示"""
    if len(expr_str) <= max_length:
        return expr_str

    # 常见的简化规则
    expr_str = expr_str.replace('LogicVar_', 'V')
    expr_str = expr_str.replace('_General', '')
    expr_str = expr_str.replace('_Premise', '')
    expr_str = expr_str.replace('_Conclusion', '')

    # 如果还是太长，截断并添加省略号
    if len(expr_str) > max_length:
        return expr_str[:max_length - 3] + "..."

    return expr_str


def _draw_modern_style(G: nx.DiGraph, node_info: Dict, ax):
    """现代风格绘制"""
    # 使用层次化布局
    pos = _create_hierarchical_layout(G, node_info)

    # 现代配色方案
    colors = {
        'background': '#f8f9fa',
        'node_fill': '#e3f2fd',
        'node_border': '#1976d2',
        'edge': '#666666',
        'text': '#212121',
        'rule_text': '#1565c0'
    }

    # 设置背景色
    ax.set_facecolor(colors['background'])

    # 绘制边（使用曲线）
    _draw_curved_edges(G, pos, ax, color=colors['edge'], alpha=0.7)

    # 绘制节点
    for node_id, (x, y) in pos.items():
        info = node_info[node_id]

        # 节点形状（圆角矩形）
        bbox = dict(boxstyle="round,pad=0.3",
                    facecolor=colors['node_fill'],
                    edgecolor=colors['node_border'],
                    linewidth=2, alpha=0.9)

        # 节点文本
        rule_text = info['rule']
        expr_text = info['expression']

        # 分行显示
        display_text = f"{rule_text}\n{expr_text}"

        ax.text(x, y, display_text, ha='center', va='center',
                fontsize=9, fontweight='bold', bbox=bbox,
                color=colors['text'])


def _draw_classic_style(G: nx.DiGraph, node_info: Dict, ax):
    """经典风格绘制"""
    pos = _create_hierarchical_layout(G, node_info)

    # 经典配色
    colors = {
        'node_fill': '#ffffff',
        'node_border': '#333333',
        'edge': '#000000',
        'text': '#000000'
    }

    # 绘制边（直线）
    _draw_straight_edges(G, pos, ax, color=colors['edge'])

    # 绘制节点
    for node_id, (x, y) in pos.items():
        info = node_info[node_id]

        # 节点形状（矩形）
        bbox = dict(boxstyle="square,pad=0.3",
                    facecolor=colors['node_fill'],
                    edgecolor=colors['node_border'],
                    linewidth=1.5)

        display_text = f"{info['rule']}\n{info['expression']}"

        ax.text(x, y, display_text, ha='center', va='center',
                fontsize=9, bbox=bbox, color=colors['text'])


def _draw_minimal_style(G: nx.DiGraph, node_info: Dict, ax):
    """极简风格绘制"""
    pos = _create_hierarchical_layout(G, node_info)

    # 极简配色
    colors = {
        'node_fill': '#f5f5f5',
        'node_border': '#888888',
        'edge': '#cccccc',
        'text': '#333333'
    }

    # 绘制边（细线）
    _draw_straight_edges(G, pos, ax, color=colors['edge'], width=1, alpha=0.6)

    # 绘制节点（只显示规则名）
    for node_id, (x, y) in pos.items():
        info = node_info[node_id]

        # 圆形节点
        circle = plt.Circle((x, y), 0.3, facecolor=colors['node_fill'],
                            edgecolor=colors['node_border'], linewidth=1)
        ax.add_patch(circle)

        # 只显示规则名
        ax.text(x, y, info['rule'], ha='center', va='center',
                fontsize=8, color=colors['text'], fontweight='bold')


def _create_hierarchical_layout(G: nx.DiGraph, node_info: Dict) -> Dict[int, Tuple[float, float]]:
    """创建层次化布局"""
    # 按深度分组节点
    depth_groups = {}
    for node_id, info in node_info.items():
        depth = info['depth']
        if depth not in depth_groups:
            depth_groups[depth] = []
        depth_groups[depth].append(node_id)

    pos = {}
    max_depth = max(depth_groups.keys()) if depth_groups else 0

    for depth, nodes in depth_groups.items():
        y = max_depth - depth  # 根节点在顶部

        if len(nodes) == 1:
            # 单个节点居中
            pos[nodes[0]] = (0, y)
        else:
            # 多个节点均匀分布
            x_positions = np.linspace(-len(nodes) / 2, len(nodes) / 2, len(nodes))
            for i, node_id in enumerate(nodes):
                pos[node_id] = (x_positions[i], y)

    return pos


def _draw_curved_edges(G: nx.DiGraph, pos: Dict, ax, color='gray', alpha=0.7, width=2):
    """绘制曲线边"""
    for edge in G.edges():
        start_pos = pos[edge[0]]
        end_pos = pos[edge[1]]

        # 计算控制点创建曲线
        mid_x = (start_pos[0] + end_pos[0]) / 2
        mid_y = (start_pos[1] + end_pos[1]) / 2

        # 添加一些曲率
        control_offset = 0.2
        control_x = mid_x + control_offset
        control_y = mid_y

        # 使用贝塞尔曲线
        t = np.linspace(0, 1, 100)
        x_curve = (1 - t) ** 2 * start_pos[0] + 2 * (1 - t) * t * control_x + t ** 2 * end_pos[0]
        y_curve = (1 - t) ** 2 * start_pos[1] + 2 * (1 - t) * t * control_y + t ** 2 * end_pos[1]

        ax.plot(x_curve, y_curve, color=color, alpha=alpha, linewidth=width)

        # 添加箭头
        _add_arrow(ax, start_pos, end_pos, color=color, alpha=alpha)


def _draw_straight_edges(G: nx.DiGraph, pos: Dict, ax, color='black', alpha=1.0, width=1.5):
    """绘制直线边"""
    for edge in G.edges():
        start_pos = pos[edge[0]]
        end_pos = pos[edge[1]]

        ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]],
                color=color, alpha=alpha, linewidth=width)

        # 添加箭头
        _add_arrow(ax, start_pos, end_pos, color=color, alpha=alpha)


def _add_arrow(ax, start_pos: Tuple[float, float], end_pos: Tuple[float, float],
               color='black', alpha=1.0, size=0.15):
    """添加箭头"""
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]

    # 缩短箭头，避免与节点重叠
    length = np.sqrt(dx ** 2 + dy ** 2)
    if length > 0:
        # 箭头起点稍微偏移
        offset = 0.35
        arrow_start_x = start_pos[0] + dx * offset / length
        arrow_start_y = start_pos[1] + dy * offset / length
        arrow_end_x = end_pos[0] - dx * offset / length
        arrow_end_y = end_pos[1] - dy * offset / length

        ax.annotate('', xy=(arrow_end_x, arrow_end_y),
                    xytext=(arrow_start_x, arrow_start_y),
                    arrowprops=dict(arrowstyle='->', color=color, alpha=alpha,
                                    lw=1.5, shrinkA=0, shrinkB=0))


def create_comparison_visualization(dag_list: List, labels: List[str],
                                    filename="dag_comparison", format="png"):
    """
    创建多个DAG的对比可视化

    参数：
        dag_list: DAG根节点列表
        labels: 对应的标签列表
        filename: 输出文件名
        format: 输出格式
    """
    try:
        n_dags = len(dag_list)
        if n_dags == 0:
            print("[⚠️  对比可视化警告] 没有提供DAG")
            return

        # 创建子图
        cols = min(3, n_dags)
        rows = (n_dags + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), dpi=300)
        if n_dags == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        for i, (dag, label) in enumerate(zip(dag_list, labels)):
            ax = axes[i]

            G, node_info = _build_networkx_graph(dag)
            if len(G.nodes()) > 0:
                _draw_minimal_style(G, node_info, ax)

            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.axis('off')

        # 隐藏多余的子图
        for i in range(n_dags, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        output_path = f"{filename}.{format}"
        plt.savefig(output_path, format=format, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[✅ 对比可视化输出] 已保存为 {output_path}")

    except Exception as e:
        print(f"[❌ 对比可视化失败] {e}")
        plt.close('all')


def save_dag_info(root_node, filename="dag_info.txt"):
    """
    保存DAG的文本信息（作为可视化的补充）
    """
    try:
        info_lines = []
        visited = set()

        def collect_info(node, depth=0):
            if node is None or id(node) in visited:
                return
            visited.add(id(node))

            indent = "  " * depth
            rule_name = getattr(node, 'rule_name', 'Unknown')
            expr = str(getattr(node, 'z3_expr', 'Unknown'))

            info_lines.append(f"{indent}[深度 {depth}] {rule_name}: {expr}")

            if hasattr(node, 'children') and node.children:
                for child in node.children:
                    collect_info(child, depth + 1)

        collect_info(root_node)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("DAG 结构信息\n")
            f.write("=" * 50 + "\n")
            f.write("\n".join(info_lines))

        print(f"[✅ DAG 信息输出] 已保存为 {filename}")

    except Exception as e:
        print(f"[❌ DAG 信息保存失败] {e}")


# 为了保持向后兼容，提供一个简化的接口
def visualize_dag_simple(root_node, output_dir="output", sample_id="sample"):
    """
    简化的可视化接口（与原始接口兼容）
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = os.path.join(output_dir, f"dag_{sample_id}")

    # 生成多种格式和风格
    visualize_dag(root_node, filename=filename, format="png", style="modern")
    visualize_dag(root_node, filename=f"{filename}_minimal", format="pdf", style="minimal")

    # 同时保存文本信息
    save_dag_info(root_node, filename=f"{filename}_info.txt")


if __name__ == "__main__":
    # 测试代码
    print("DAG 可视化器测试")
    print("需要提供 DAGNode 实例进行测试")
    print("依赖库: matplotlib, networkx, numpy")
    print("安装命令: pip install matplotlib networkx numpy")