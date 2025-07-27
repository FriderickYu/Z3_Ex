# 文件路径: test/test_structured_goal.py
# 说明: 构建结构化目标表达式并展开为 DAG，验证 target_depth / max_branching 控制逻辑

from dag.dag_builder import DAGBuilder
from rules.rules_pool import rule_pool
from rules.balanced_and_rule import BalancedAndRule
import z3


def print_dag(node, indent=0):
    print("  " * indent + f"{node.z3_expr} (depth={node.depth}, rule={type(node.rule).__name__ if node.rule else 'None'})")
    for child in node.children:
        print_dag(child, indent + 1)


def main():
    # ✅ 用 BalancedAndRule 构造一个结构性目标表达式
    premises = [z3.Bool(f"Var_{i}") for i in range(32)]  # 可调节为 16/32 等更深层结构
    goal_expr = BalancedAndRule().construct_conclusion(premises)

    # ✅ 初始化 DAG 构建器，控制目标深度和分支数
    builder = DAGBuilder(5)

    # ✅ 构建推理图
    dag_root = builder.build()

    print("\n🧠 构造的推理图结构如下（结构化目标表达式构建）：\n")
    print_dag(dag_root)


if __name__ == "__main__":
    main()