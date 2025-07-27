# 文件：dag/dag_builder.py
# 说明：构建支持多规则组合的逻辑推理 DAG，并提取结构化推理路径

import random
from copy import deepcopy
import z3
from rules.rules_pool import rule_pool


class DAGNode:
    def __init__(self, z3_expr, rule=None):
        self.z3_expr = z3_expr  # 当前节点对应的 Z3 表达式
        self.rule = rule        # 当前节点是由哪个规则生成的
        self.children = []      # 子节点（前提）

    def add_child(self, child_node):
        self.children.append(child_node)


class DAGBuilder:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.node_counter = 0

    def build(self):
        # 1. 随机采样一个规则用于构造最终目标表达式
        final_rule = rule_pool.sample_rule()
        num = final_rule.num_premises()
        premises = [z3.Bool(f"Var_{self.node_counter + i}") for i in range(num)]
        self.node_counter += num

        goal_expr = final_rule.construct_conclusion(premises)

        # 2. 构造 DAG 根节点
        root = DAGNode(goal_expr, final_rule)
        for premise_expr in premises:
            child_node = DAGNode(premise_expr)
            root.add_child(child_node)
            self._expand_node(child_node, current_depth=2)

        return root

    def _expand_node(self, node, current_depth):
        if current_depth >= self.max_depth:
            return

        # 3. 为当前节点递归构造子前提
        rule = rule_pool.sample_rule()
        num = rule.num_premises()
        premises = [z3.Bool(f"Var_{self.node_counter + i}") for i in range(num)]
        self.node_counter += num

        conclusion = rule.construct_conclusion(premises)

        # 替换当前节点的表达式为新结论，并添加前提为子节点
        node.z3_expr = conclusion
        node.rule = rule

        for premise_expr in premises:
            child_node = DAGNode(premise_expr)
            node.add_child(child_node)
            self._expand_node(child_node, current_depth + 1)


# === 🧠 提取逻辑步骤（结构化形式） ===
def extract_logical_steps(root_node):
    steps = []

    def dfs(node):
        for child in node.children:
            dfs(child)

        if not node.children:
            return  # 叶子节点不构成推理步骤

        # 构造 antecedent 字符串
        if len(node.children) == 1:
            antecedent_str = str(node.children[0].z3_expr)
        else:
            antecedent_str = f"And({', '.join(str(c.z3_expr) for c in node.children)})"

        step = {
            "rule": type(node.rule).__name__,
            "conclusion": str(node.z3_expr),
            "conclusion_expr": node.z3_expr,  # ✅ 保留 Z3 对象
            "premises": [str(c.z3_expr) for c in node.children],
            "premises_expr": [c.z3_expr for c in node.children],  # ✅ 保留 Z3 对象
            "antecedent": antecedent_str,
            "description": f"{type(node.rule).__name__} 推理得到 {str(node.z3_expr)}"
        }
        steps.append(step)

    dfs(root_node)
    return steps


# ✅ 对外统一封装接口
def build_reasoning_dag(max_depth=3):
    """
    快速构建推理 DAG 并提取结构化推理步骤
    返回：
        root_node: DAG 根节点
        steps: List[Dict] 推理步骤
    """
    builder = DAGBuilder(max_depth=max_depth)
    root_node = builder.build()
    steps = extract_logical_steps(root_node)
    return root_node, steps