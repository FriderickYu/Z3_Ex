# 文件说明：定义逻辑推理图中的节点结构
# 每个节点代表一个 z3 表达式，记录其父子关系和应用的规则

class Node:
    def __init__(self, z3_expr, depth, rule=None):
        """
        初始化一个图节点

        参数:
        - z3_expr: Z3 表达式（结论）
        - depth: 当前节点深度（根为0）
        - rule: 当前节点由哪个规则推导出（可选）
        """
        self.id = str(z3_expr)
        self.z3_expr = z3_expr
        self.depth = depth
        self.rule = rule
        # 被推导
        self.parents = []
        # 前提
        self.children = []

    def add_child(self, child_node):
        """将一个子节点添加为当前节点的前提"""
        self.children.append(child_node)
        child_node.parents.append(self)

    def __repr__(self):
        return f"Node(id={self.id}, depth={self.depth}, rule={type(self.rule).__name__ if self.rule else None})"