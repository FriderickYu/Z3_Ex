# 文件：dag/visualizer.py
# 功能：将 DAGNode 结构可视化为 Graphviz 图

from graphviz import Digraph

def visualize_dag(root_node, filename="reasoning_dag", format="pdf"):
    """
    将 DAGNode 结构输出为 Graphviz 图
    参数：
        root_node: DAGNode，DAG 的根节点
        filename: str，输出文件名（不含扩展名）
        format: str，输出格式，默认为 pdf（可选 svg/png/dot 等）
    """
    dot = Digraph(comment="Reasoning DAG")
    visited = set()

    def add_node(node):
        node_id = str(id(node))
        if node_id in visited:
            return
        visited.add(node_id)

        # ✅ 使用纯文本 label，避免 HTML 语法错误
        label = f"{type(node.rule).__name__}\n{str(node.z3_expr)}"
        dot.node(node_id, label=label, shape="box", style="filled", fillcolor="lightgrey")

        for child in node.children:
            child_id = str(id(child))
            add_node(child)
            dot.edge(child_id, node_id)

    add_node(root_node)

    dot.render(filename=filename, format=format, cleanup=True)
    print(f"[✅ DAG 可视化输出] 已保存为 {filename}.{format}")