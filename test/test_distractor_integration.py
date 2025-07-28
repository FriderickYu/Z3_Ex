# 文件：test/test_distractor_integration.py
# 说明：集成测试推理 DAG 构建 + 验证 + 干扰项生成流程

from dag.dag_builder import build_reasoning_dag
from dag.validator import validate_logical_steps
from dag.visualizer import visualize_dag
from distractor.generator import DistractorGenerator
import z3


def extract_simple_bool_vars_from_dag(root_node):
    """
    从 DAG 中提取所有简单的布尔变量（排除复杂表达式）
    """
    all_vars = set()

    def is_simple_bool_var(expr):
        """检查是否为简单布尔变量"""
        try:
            return (z3.is_const(expr) and
                    z3.is_bool(expr) and
                    expr.decl().kind() == z3.Z3_OP_UNINTERPRETED)
        except:
            return False

    def extract_vars_from_expr(expr):
        """从表达式中递归提取简单变量"""
        if is_simple_bool_var(expr):
            all_vars.add(expr)
        elif hasattr(expr, 'children'):
            try:
                for child in expr.children():
                    extract_vars_from_expr(child)
            except:
                pass

    def traverse(node):
        # 提取当前节点表达式中的变量
        try:
            extract_vars_from_expr(node.z3_expr)
        except:
            pass

        # 递归遍历子节点
        for child in node.children:
            traverse(child)

    traverse(root_node)
    return list(all_vars)


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

    if failed_steps:
        print("\n失败步骤详情：")
        for step in failed_steps:
            print(f"  - {step['rule']}: {step.get('error', '未知错误')}")

    # 4. 提取 DAG 中的简单布尔变量
    all_vars = extract_simple_bool_vars_from_dag(root)
    print(f"\n🔧 从 DAG 中提取到 {len(all_vars)} 个简单布尔变量")

    # 如果没有提取到变量，创建一些默认变量
    if not all_vars:
        print("⚠️  未提取到变量，创建默认变量池")
        all_vars = [z3.Bool(f"DefaultVar_{i}") for i in range(5)]

    # 5. 为每个逻辑步骤生成干扰项
    if valid_steps:
        generator = DistractorGenerator(available_vars=all_vars)

        print("\n🔄 生成干扰项：")
        for i, step in enumerate(valid_steps[:3], 1):  # 只处理前3步，避免输出过多
            print(f"\n--- Step {i}: {step['rule']} ---")
            print(f"原始结论: {step['conclusion']}")

            # 为单步生成干扰项
            distractors = generator.generate_all([step], num_per_strategy=1)

            if not distractors:
                print("  ⚠️  未生成任何干扰项")
            else:
                for j, d in enumerate(distractors, 1):
                    print(f"  Distractor {j} [{d['strategy']}]: {d['description']}")
                    if 'premises_expr' in d and d['premises_expr']:
                        print(f"    前提: {', '.join(str(p) for p in d['premises_expr'])}")
    else:
        print("⚠️  没有有效的推理步骤，跳过干扰项生成")

    # 6. 可视化 DAG（可选）
    try:
        visualize_dag(root, filename="output/reasoning_dag", format="pdf")
        print("\n📊 DAG 可视化已保存")
    except Exception as e:
        print(f"\n⚠️  可视化失败: {e}")

    print("\n=== ✅ 集成测试完成 ===")


if __name__ == "__main__":
    test_distractor_integration()