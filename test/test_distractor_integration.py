# 文件：test/test_distractor_integration.py
# 说明：集成测试推理 DAG 构建 + 验证 + 干扰项生成流程

from dag.dag_builder import build_reasoning_dag
from dag.validator import validate_logical_steps
from dag.visualizer import visualize_dag
from distractor.generator import DistractorGenerator


def test_distractor_integration():
    print("=== 🔍 集成测试：推理验证 + 干扰项生成 ===")

    # 1. 构建推理 DAG
    root, steps = build_reasoning_dag(max_depth=3)

    # 2. 打印推理路径
    print("\n📜 推理链（logical steps）:")
    for i, step in enumerate(steps, 1):
        print(f"Step {i}: {step['conclusion']} ← {step['rule']}({', '.join(step['premises'])})")

    # 3. 验证逻辑
    valid_steps, failed_steps = validate_logical_steps(steps)
    print(f"\n🧪 验证通过：{len(valid_steps)} 步")
    print(f"❌ 验证失败：{len(failed_steps)} 步")

    # 4. 为每个逻辑步骤生成干扰项（使用 DistractorGenerator）
    # 构造 DistractorGenerator，变量池来自 DAG 根节点
    all_vars = list(root.symbol_table.values())
    generator = DistractorGenerator(available_vars=all_vars)

    for i, step in enumerate(valid_steps, 1):
        print(f"\n--- 🔄 Step {i}: {step['rule']} ---")
        # 仅对单步 step 封装为列表，传入 generator
        distractors = generator.generate_all([step], num_per_strategy=2)
        for j, d in enumerate(distractors, 1):
            print(f"  Distractor {j} [{d['strategy']}]: {d['natural_language']}")
            print(f"    Z3: {d['z3_expr']}")

    # 5. 可视化 DAG（可选）
    visualize_dag(root, filename="output/reasoning_dag", format="pdf")

    print("\n=== ✅ 集成测试完成 ===")


if __name__ == "__main__":
    test_distractor_integration()
