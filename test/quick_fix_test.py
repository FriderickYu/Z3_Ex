# 文件：test/quick_fix_test.py
# 说明：快速测试修复后的组件

import sys
from pathlib import Path

# 添加根目录到Python路径
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import logging
import z3


def test_basic_imports():
    """测试基本导入是否正常"""
    print("🧪 测试基本导入...")

    try:
        from dag.dag_builder import build_reasoning_dag
        print("✅ dag_builder 导入成功")

        from dag.validator import validate_logical_steps
        print("✅ validator 导入成功")

        from distractor.generator import DistractorGenerator
        print("✅ distractor.generator 导入成功")

        print("✅ z3 导入成功")
        return True

    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_z3_variable_creation():
    """测试Z3变量创建"""
    print("\n🔧 测试Z3变量创建...")

    try:
        # 创建简单变量
        var1 = z3.Bool("Var_1")
        var2 = z3.Bool("Var_2")

        # 测试表达式创建
        expr = z3.And(var1, var2)
        print(f"✅ 创建表达式成功: {expr}")

        # 测试字符串转换
        expr_str = str(expr)
        print(f"✅ 字符串转换成功: {expr_str}")

        # 测试变量提取
        import re
        variables = re.findall(r'Var_\d+', expr_str)
        print(f"✅ 变量提取成功: {variables}")

        return True

    except Exception as e:
        print(f"❌ Z3变量测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dag_minimal():
    """最小化DAG测试"""
    print("\n🏗️ 测试最小化DAG构建...")

    try:
        from dag.dag_builder import build_reasoning_dag

        # 构建最小DAG
        root, steps = build_reasoning_dag(max_depth=1)
        print(f"✅ DAG构建成功，步骤数: {len(steps)}")

        # 检查根节点
        if hasattr(root, 'z3_expr'):
            print(f"✅ 根节点有z3_expr属性")
            try:
                expr_str = str(root.z3_expr)
                print(f"✅ 根节点表达式: {expr_str[:50]}...")
            except Exception as e:
                print(f"⚠️ 根节点表达式转换失败: {e}")
        else:
            print("⚠️ 根节点无z3_expr属性")

        return root, steps

    except Exception as e:
        print(f"❌ DAG构建失败: {e}")
        import traceback
        traceback.print_exc()
        return None, []


def test_distractor_minimal():
    """最小化干扰项测试"""
    print("\n🎯 测试最小化干扰项生成...")

    try:
        from distractor.generator import DistractorGenerator

        # 创建简单变量
        vars = [z3.Bool(f"Var_{i}") for i in range(3)]
        print(f"✅ 创建了 {len(vars)} 个变量")

        # 创建简单测试步骤
        test_step = {
            'premises_expr': [vars[0], vars[1]],
            'conclusion_expr': z3.And(vars[0], vars[1]),
            'rule': 'TestRule',
            'premises': ['Var_0', 'Var_1'],
            'conclusion': 'And(Var_0, Var_1)'
        }

        # 测试生成器
        generator = DistractorGenerator(available_vars=vars)
        distractors = generator.generate_all([test_step], num_per_strategy=1)

        print(f"✅ 干扰项生成成功，数量: {len(distractors)}")
        for d in distractors:
            strategy = d.get('strategy', 'unknown')
            print(f"  - 策略: {strategy}")

        return True

    except Exception as e:
        print(f"❌ 干扰项测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_test():
    """运行快速测试"""
    print("🚀 快速修复验证测试")
    print("=" * 40)

    # 1. 测试导入
    if not test_basic_imports():
        return

    # 2. 测试Z3基础功能
    if not test_z3_variable_creation():
        return

    # 3. 测试DAG构建
    root, steps = test_dag_minimal()
    if not steps:
        return

    # 4. 测试干扰项生成
    if not test_distractor_minimal():
        return

    print("\n" + "=" * 40)
    print("✅ 所有基础测试通过")
    print("💡 可以尝试运行完整的数据集生成测试了")


if __name__ == "__main__":
    # 配置简单日志
    logging.basicConfig(level=logging.INFO)

    run_quick_test()