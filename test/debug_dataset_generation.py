# 文件：test/debug_dataset_generation.py
# 说明：调试版数据集生成测试，逐步验证各个组件

import sys
import os
from pathlib import Path

# 添加根目录到Python路径
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import logging
import json
import z3
from dag.dag_builder import build_reasoning_dag
from dag.validator import validate_logical_steps
from distractor.generator import DistractorGenerator


class DebugVariableExtractor:
    """调试版变量提取器"""

    def __init__(self):
        self.logger = logging.getLogger("debug_extractor")

    def extract_from_dag_debug(self, root_node):
        """逐步调试DAG变量提取过程"""
        print("🔍 开始调试DAG变量提取...")
        variables = set()
        node_count = 0

        def safe_traverse(node, depth=0):
            nonlocal node_count
            node_count += 1

            print(f"{'  ' * depth}📍 处理节点 {node_count}")

            try:
                # 检查节点属性
                if not hasattr(node, 'z3_expr'):
                    print(f"{'  ' * depth}⚠️  节点无 z3_expr 属性")
                    return

                expr = node.z3_expr
                if expr is None:
                    print(f"{'  ' * depth}⚠️  z3_expr 为 None")
                    return

                # 检查表达式类型
                print(f"{'  ' * depth}📝 表达式类型: {type(expr)}")

                try:
                    # 尝试转换为字符串
                    expr_str = str(expr)
                    print(f"{'  ' * depth}📄 表达式字符串: {expr_str[:50]}...")

                    # 提取变量
                    import re
                    vars_in_expr = re.findall(r'Var_\d+', expr_str)
                    if vars_in_expr:
                        print(f"{'  ' * depth}✅ 找到变量: {vars_in_expr}")
                        variables.update(vars_in_expr)
                    else:
                        print(f"{'  ' * depth}❌ 未找到变量")

                except Exception as e:
                    print(f"{'  ' * depth}❌ 字符串转换失败: {e}")

                # 遍历子节点
                if hasattr(node, 'children') and node.children:
                    print(f"{'  ' * depth}🌳 有 {len(node.children)} 个子节点")
                    for i, child in enumerate(node.children):
                        if child is not None:
                            print(f"{'  ' * depth}├─ 子节点 {i + 1}:")
                            safe_traverse(child, depth + 1)
                        else:
                            print(f"{'  ' * depth}├─ 子节点 {i + 1}: None")
                else:
                    print(f"{'  ' * depth}🍃 叶子节点")

            except Exception as e:
                print(f"{'  ' * depth}💥 节点处理异常: {e}")

        safe_traverse(root_node)

        result = sorted(list(variables))
        print(f"\n✅ 总计处理 {node_count} 个节点")
        print(f"🎯 提取到变量: {result}")
        return result

    def test_variable_creation(self, var_names):
        """测试变量创建过程"""
        print(f"\n🧪 测试创建 {len(var_names)} 个变量...")

        safe_vars = []
        for i, var_name in enumerate(var_names):
            try:
                print(f"  创建变量 {i + 1}: {var_name}")
                bool_var = z3.Bool(var_name)

                # 测试变量是否可用
                test_expr = z3.And(bool_var, z3.Bool('test'))
                print(f"    ✅ 变量创建成功，测试表达式: {test_expr}")

                safe_vars.append(bool_var)

            except Exception as e:
                print(f"    ❌ 变量创建失败: {e}")
                continue

        print(f"✅ 成功创建 {len(safe_vars)} 个变量")
        return safe_vars


def test_dag_building():
    """测试DAG构建过程"""
    print("=== 🏗️ 测试DAG构建 ===")

    try:
        root, logical_steps = build_reasoning_dag(max_depth=2)  # 减少深度降低复杂度
        print(f"✅ DAG构建成功")
        print(f"📊 生成了 {len(logical_steps)} 个推理步骤")

        # 打印步骤概要
        for i, step in enumerate(logical_steps[:3], 1):
            print(f"  Step {i}: {step.get('rule')} -> {step.get('conclusion', '')[:30]}...")

        return root, logical_steps

    except Exception as e:
        print(f"❌ DAG构建失败: {e}")
        import traceback
        traceback.print_exc()
        return None, []


def test_validation(logical_steps):
    """测试逻辑验证"""
    print("\n=== 🔍 测试逻辑验证 ===")

    try:
        valid_steps, failed_steps = validate_logical_steps(logical_steps)
        print(f"✅ 验证完成")
        print(f"📈 有效步骤: {len(valid_steps)}")
        print(f"📉 失败步骤: {len(failed_steps)}")

        return valid_steps, failed_steps

    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return [], []


def test_variable_extraction(root):
    """测试变量提取"""
    print("\n=== 🔧 测试变量提取 ===")

    extractor = DebugVariableExtractor()
    variables = extractor.extract_from_dag_debug(root)

    if variables:
        print(f"✅ 变量提取成功: {variables}")

        # 测试变量创建
        safe_vars = extractor.test_variable_creation(variables[:5])
        return variables, safe_vars
    else:
        print("❌ 未提取到任何变量")
        return [], []


def test_distractor_generation(valid_steps, safe_vars):
    """测试干扰项生成"""
    print("\n=== 🎯 测试干扰项生成 ===")

    if not safe_vars:
        print("⚠️ 没有可用变量，跳过干扰项生成")
        return []

    try:
        generator = DistractorGenerator(available_vars=safe_vars)
        distractors = generator.generate_all(valid_steps[:2], num_per_strategy=1)

        print(f"✅ 干扰项生成成功")
        print(f"📊 生成了 {len(distractors)} 个干扰项")

        for i, d in enumerate(distractors, 1):
            strategy = d.get('strategy', 'unknown')
            desc = d.get('description', 'No description')
            print(f"  Distractor {i}: [{strategy}] {desc[:50]}...")

        return distractors

    except Exception as e:
        print(f"❌ 干扰项生成失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_debug_test():
    """运行完整的调试测试"""
    print("🔬 开始调试测试")
    print("=" * 60)

    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 1. 测试DAG构建
    root, logical_steps = test_dag_building()
    if not logical_steps:
        print("💥 DAG构建失败，测试终止")
        return

    # 2. 测试逻辑验证
    valid_steps, failed_steps = test_validation(logical_steps)
    if not valid_steps:
        print("💥 没有有效步骤，测试终止")
        return

    # 3. 测试变量提取
    variables, safe_vars = test_variable_extraction(root)
    if not variables:
        print("💥 变量提取失败，测试终止")
        return

    # 4. 测试干扰项生成
    distractors = test_distractor_generation(valid_steps, safe_vars)

    # 5. 保存调试结果
    debug_output = {
        "dag_steps": len(logical_steps),
        "valid_steps": len(valid_steps),
        "failed_steps": len(failed_steps),
        "variables_extracted": variables,
        "distractors_generated": len(distractors),
        "status": "SUCCESS" if distractors else "PARTIAL_SUCCESS"
    }

    # 保存到test/output
    test_output_dir = Path(__file__).parent / "output"
    test_output_dir.mkdir(exist_ok=True)

    with open(test_output_dir / "debug_results.json", 'w', encoding='utf-8') as f:
        json.dump(debug_output, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("✅ 调试测试完成")
    print(f"📊 结果摘要: {debug_output}")
    print(f"💾 详细结果已保存到: {test_output_dir / 'debug_results.json'}")


if __name__ == "__main__":
    run_debug_test()
