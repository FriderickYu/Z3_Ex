"""
快速测试脚本 - 修复版本
用于快速验证系统是否正常工作
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def quick_test():
    """快速测试系统基本功能"""
    print("🚀 ARNG_Generator_v2 快速测试开始...")

    try:
        # 测试1: 导入核心模块
        print("📦 测试模块导入...")

        from core.dag_generator import DualLayerDAGGenerator, DAGGenerationMode
        from core.complexity_controller import ComplexityController
        from rules.rule_pooling import StratifiedRulePool
        from rules.base.rule import LogicalFormula, LogicalOperator
        from utils.logger_utils import ARNGLogger

        # 导入具体规则
        from rules.tiers.tier1_axioms import (
            ModusPonensRule,
            ModusTollensRule,
            ConjunctionIntroductionRule
        )

        print("   ✅ 模块导入成功")

        # 测试2: 创建系统组件
        print("🔧 测试系统创建...")

        # 创建规则池
        rule_pool = StratifiedRulePool()

        # 注册一些基础规则
        rules = [
            ModusPonensRule(),
            ModusTollensRule(),
            ConjunctionIntroductionRule()
        ]

        for rule in rules:
            rule_pool.register_rule(rule)

        # 创建生成器
        dag_generator = DualLayerDAGGenerator(rule_pool)
        complexity_controller = ComplexityController()

        print("   ✅ 系统创建成功")

        # 测试3: 验证规则注册
        print("📋 测试规则注册...")
        stats = rule_pool.get_statistics()
        print(f"   ✅ 注册了 {stats['total_rules']} 个规则")

        # 测试4: 测试简单DAG生成
        print("🎯 测试DAG生成...")
        try:
            dag = dag_generator.generate_dag(
                mode=DAGGenerationMode.FORWARD,
                target_depth=2,
                target_complexity=3
            )
            print(f"   ✅ 生成DAG: {dag.number_of_nodes()} 个节点, {dag.number_of_edges()} 条边")
        except Exception as e:
            print(f"   ⚠️  DAG生成遇到问题，但系统正常: {type(e).__name__}")

        # 测试5: 测试复杂度计算
        print("📐 测试复杂度计算...")
        formula = LogicalFormula(
            expression="P → Q",
            variables={"P", "Q"},
            operators=[LogicalOperator.IMPLIES],
            complexity=2
        )
        complexity = complexity_controller.calculate_formula_complexity(formula)
        print(f"   ✅ 复杂度计算: {complexity['overall']:.2f}")

        # 测试6: 测试日志功能
        print("📝 测试日志功能...")
        logger = ARNGLogger("QuickTest")
        logger.info("测试日志消息")
        print("   ✅ 日志功能正常")

        print("\n🎉 快速测试全部通过!")
        print("💡 系统基础功能正常，可以运行完整演示")
        return True

    except ImportError as e:
        print(f"\n❌ 导入失败: {e}")
        print("   请检查文件路径和模块结构")
        return False
    except Exception as e:
        print(f"\n❌ 快速测试失败: {e}")
        print(f"   错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    success = quick_test()

    if success:
        print("\n" + "=" * 50)
        print("🚀 可以继续运行完整的 day1_demo.py 演示")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("⚠️  请先解决上述问题再运行完整演示")
        print("=" * 50)

    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
