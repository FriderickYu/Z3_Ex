from dag.dag_builder import DAGBuilder, extract_logical_steps
from rules.rules_pool import rule_pool
from rules.balanced_and_rule import BalancedAndRule
import z3

def print_dag(node, indent=0):
    print("  " * indent + f"{node.z3_expr} (depth={node.depth}, rule={type(node.rule).__name__ if node.rule else None})")
    for child in node.children:
        print_dag(child, indent + 1)

def main():
    # 🧱 构造目标合取表达式（可以换成 8、16、32 来观察深度）
    premises = [z3.Bool(f"Var_{i}") for i in range(32)]
    goal_expr = BalancedAndRule().construct_conclusion(premises)

    # 🏗️ 构建 DAG（你可调整参数查看结构复杂度）
    builder = DAGBuilder(rule_pool, target_depth=5, max_branching=3)
    dag_root = builder.build(goal_expr, root_rule=BalancedAndRule())

    # 🌳 打印图结构
    print("🧠 构造的推理图结构如下（自动生成）：\n")
    print_dag(dag_root)

    # 📜 打印逻辑推理链
    steps = extract_logical_steps(dag_root)
    print(f"Steps are : {steps}")
    print("\n📜 推理路径如下（logical_steps）：")
    for i, step in enumerate(steps, 1):
        print(f"Step {i}: {step['conclusion']} ← {step['rule']}({', '.join(step['premises'])})")

if __name__ == "__main__":
    main()
