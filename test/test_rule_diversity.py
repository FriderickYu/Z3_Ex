# 文件：test/test_rule_diversity.py
# 说明：测试规则多样性与结构合理性，并验证每步推理的逻辑有效性

from collections import Counter
from dag.dag_builder import build_reasoning_dag
from dag.validator import validate_logical_steps
from dag.visualizer import visualize_dag


def test_rule_diversity():
    print("=== 🧪 多规则组合验证（test_rule_diversity） ===")

    # 构建 DAG 并提取推理步骤
    root, steps = build_reasoning_dag(max_depth=3)


    # 打印每一步推理
    for idx, step in enumerate(steps, 1):
        print(f"Step {idx}: {step['conclusion']} ← {step['rule']}({step['antecedent']})")

    # 统计规则使用频率
    rule_counter = Counter([step['rule'] for step in steps])
    print("\n📊 规则使用频次统计：")
    for rule, count in rule_counter.items():
        print(f"- {rule}: {count} 次")

    # 验证每步推理逻辑有效性
    print("\n🧪 验证 logical_steps 合理性...")
    valid_steps, failed_steps = validate_logical_steps(steps)

    print(f"\n✅ 通过验证的推理步骤：{len(valid_steps)} / {len(steps)}")

    if failed_steps:
        print("\n❌ 存在验证失败的步骤：")
        for step in failed_steps:
            print(f"- ❗ {step['rule']}({step['antecedent']}) ⟶ {step['conclusion']}")
            print(f"  错误原因: {step.get('error', '未知错误')}")

    visualize_dag(root, filename="output/reasoning_dag", format="pdf")

    print("=== ✅ 多规则组合验证完成 ===")




if __name__ == "__main__":
    test_rule_diversity()
