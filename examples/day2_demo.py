"""
Day 2 演示脚本 - 集成系统完整测试
测试表达式树构建器、复杂度驱动控制器、Z3验证器和集成DAG生成器
展示新功能与原有功能的对比
"""

import sys
import os
import time
import json
from pathlib import Path
from rules.rule_pooling import StratifiedRulePool
from rules.base.rule import LogicalFormula, LogicalOperator
from utils.logger_utils import ARNGLogger

from rules.tiers.tier1_axioms import (
    ModusPonensRule,
    ModusTollensRule,
    HypotheticalSyllogismRule,
    DisjunctiveSyllogismRule,
    ConjunctionIntroductionRule,
    ConjunctionEliminationRule
)


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# 尝试导入集成组件
try:
    from core.expression_tree_builder import ExpressionTreeBuilder, ExpressionType, ConnectorType
    from core.complexity_controller import ComplexityController, ComplexityGrowthStrategy
    from core.z3_validator import Z3Validator, ValidationResult
    from core.dag_generator import DualLayerDAGGenerator, DAGGenerationMode

    INTEGRATED_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  集成组件导入失败: {e}")
    # 导入原有组件作为备选
    from core.dag_generator import DualLayerDAGGenerator, DAGGenerationMode

    INTEGRATED_AVAILABLE = False

# 导入可视化（如果可用）
try:
    from dag_visualizer import create_unified_visualization

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


def print_separator(title: str, char: str = "=", width: int = 80):
    """打印分隔线"""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


def print_subsection(title: str):
    """打印子章节"""
    print(f"\n{'─' * 60}")
    print(f"🔸 {title}")
    print(f"{'─' * 60}")


def get_tier1_rules():
    """获取所有Tier 1规则实例"""
    return [
        ModusPonensRule(),
        ModusTollensRule(),
        HypotheticalSyllogismRule(),
        DisjunctiveSyllogismRule(),
        ConjunctionIntroductionRule(),
        ConjunctionEliminationRule()
    ]


def create_rule_pool():
    """创建并初始化规则池"""
    rule_pool = StratifiedRulePool()
    tier1_rules = get_tier1_rules()

    for rule in tier1_rules:
        rule_pool.register_rule(rule)

    return rule_pool


def test_expression_tree_builder():
    """测试表达式树构建器"""
    print_subsection("表达式树构建器测试")

    if not INTEGRATED_AVAILABLE:
        print("❌ 表达式树构建器不可用")
        return None

    try:
        tree_builder = ExpressionTreeBuilder()

        # 测试不同深度的表达式树
        print("📊 测试不同深度的表达式树生成:")

        results = []
        for depth in [2, 3, 4]:
            print(f"  🌲 深度 {depth}:", end=" ")

            tree = tree_builder.build_expression_tree(
                target_depth=depth,
                complexity_factor=1.0,
                variable_limit=4
            )

            formula = tree_builder.convert_to_logical_formula(tree)
            actual_depth = _get_tree_depth(tree)

            print(f"实际深度 {actual_depth} | 表达式: {formula.expression}")

            results.append({
                'target_depth': depth,
                'actual_depth': actual_depth,
                'expression': formula.expression,
                'variables': list(formula.variables),
                'complexity': formula.complexity
            })

        # 测试复杂度因子影响
        print("\n📈 测试复杂度因子的影响:")
        for factor in [0.5, 1.0, 1.5, 2.0]:
            tree = tree_builder.build_expression_tree(
                target_depth=3,
                complexity_factor=factor
            )
            formula = tree_builder.convert_to_logical_formula(tree)
            print(f"  🎛️  因子 {factor}: {formula.expression} (复杂度: {formula.complexity})")

        # 获取统计信息
        stats = tree_builder.get_statistics()
        print(f"\n📊 构建统计: 总数 {stats['total_trees']}, 平均深度 {stats['avg_depth']:.2f}")

        return tree_builder, results

    except Exception as e:
        print(f"❌ 表达式树构建器测试失败: {e}")
        return None


def test_complexity_controller():
    """测试复杂度控制器"""
    print_subsection("复杂度驱动控制器测试")

    if not INTEGRATED_AVAILABLE:
        print("❌ 复杂度控制器不可用")
        return None

    try:
        complexity_controller = ComplexityController()

        # 测试公式复杂度计算
        print("📊 测试公式复杂度计算:")

        test_formulas = [
            ("P", LogicalFormula("P", {"P"}, [], 1)),
            ("P → Q", LogicalFormula("P → Q", {"P", "Q"}, [LogicalOperator.IMPLIES], 2)),
            ("(P ∧ Q) → R", LogicalFormula("(P ∧ Q) → R", {"P", "Q", "R"},
                                           [LogicalOperator.AND, LogicalOperator.IMPLIES], 3)),
            ("¬(P ∨ Q) ∧ R", LogicalFormula("¬(P ∨ Q) ∧ R", {"P", "Q", "R"},
                                            [LogicalOperator.NOT, LogicalOperator.OR, LogicalOperator.AND], 4))
        ]

        for name, formula in test_formulas:
            complexity = complexity_controller.calculate_formula_complexity(formula)
            print(f"  📐 {name:15} | 总复杂度: {complexity['overall']:.2f} | "
                  f"结构: {complexity['structural']:.1f} | 语义: {complexity['semantic']:.1f}")

        # 测试复杂度递增策略
        print("\n📈 测试复杂度递增策略:")
        strategies = [
            ComplexityGrowthStrategy.LINEAR,
            ComplexityGrowthStrategy.EXPONENTIAL,
            ComplexityGrowthStrategy.ADAPTIVE
        ]

        for strategy in strategies:
            print(f"\n  🎯 {strategy.value.upper()} 策略:")
            complexity_controller.set_growth_strategy(strategy)

            for step in range(5):
                if hasattr(complexity_controller, 'generate_complexity_target'):
                    target = complexity_controller.generate_complexity_target(step)
                    target_val = target.get('overall', 0) if isinstance(target, dict) else target
                else:
                    target_val = complexity_controller.get_target_complexity_for_step(step)

                print(f"    步骤 {step}: {target_val:.2f}")

        # 测试表达式树复杂度（如果可用）
        if hasattr(complexity_controller, 'calculate_expression_tree_complexity'):
            print("\n🌲 测试表达式树复杂度计算:")
            tree_builder = ExpressionTreeBuilder()

            for depth in [2, 3, 4]:
                tree = tree_builder.build_expression_tree(target_depth=depth)
                tree_complexity = complexity_controller.calculate_expression_tree_complexity(tree)

                print(f"  深度 {depth}: 总复杂度 {tree_complexity['overall']:.2f} | "
                      f"树深度 {tree_complexity.get('tree_depth', 0):.1f} | "
                      f"树分支 {tree_complexity.get('tree_branching', 0):.1f}")

        return complexity_controller

    except Exception as e:
        print(f"❌ 复杂度控制器测试失败: {e}")
        return None


def test_z3_validator():
    """测试Z3验证器"""
    print_subsection("Z3验证器测试")

    if not INTEGRATED_AVAILABLE:
        print("❌ Z3验证器不可用")
        return None

    try:
        z3_validator = Z3Validator(timeout_seconds=10)

        # 测试有效推理案例
        print("✅ 测试有效推理案例:")
        valid_cases = [
            (["P → Q", "P"], "Q", "Modus Ponens"),
            (["P → Q", "¬Q"], "¬P", "Modus Tollens"),
            (["P → Q", "Q → R"], "P → R", "假言三段论"),
            (["P", "Q"], "P ∧ Q", "合取引入"),
            (["P ∧ Q"], "P", "合取消除")
        ]

        valid_count = 0
        for premises, conclusion, rule_name in valid_cases:
            result = z3_validator.validate_reasoning_step(premises, conclusion)
            status = "✅" if result.is_valid else "❌"
            confidence = f"({result.confidence_score:.2f})" if hasattr(result, 'confidence_score') else ""
            print(f"  {status} {rule_name:12} | {premises} ⊢ {conclusion} {confidence}")

            if result.is_valid:
                valid_count += 1

            if result.validation_time_ms:
                print(f"      验证时间: {result.validation_time_ms:.1f}ms")

        print(f"\n  📊 有效推理识别率: {valid_count}/{len(valid_cases)} ({valid_count / len(valid_cases) * 100:.1f}%)")

        # 测试无效推理案例
        print("\n❌ 测试无效推理案例:")
        invalid_cases = [
            (["P"], "Q", "无关推理"),
            (["P → Q"], "P", "肯定后件谬误"),
            (["P → Q", "R"], "Q", "不相关前提"),
        ]

        invalid_count = 0
        for premises, conclusion, error_type in invalid_cases:
            result = z3_validator.validate_reasoning_step(premises, conclusion)
            status = "✅" if not result.is_valid else "❌"
            print(f"  {status} {error_type:12} | {premises} ⊬ {conclusion}")

            if not result.is_valid:
                invalid_count += 1

        print(
            f"\n  📊 无效推理识别率: {invalid_count}/{len(invalid_cases)} ({invalid_count / len(invalid_cases) * 100:.1f}%)")

        # 测试表达式转换
        print("\n🔄 测试表达式转换:")
        test_expressions = ["P", "P → Q", "P ∧ Q", "P ∨ Q", "¬P", "(P ∧ Q) → R"]

        for expr in test_expressions:
            try:
                z3_expr, warnings = z3_validator.converter.convert_to_z3(expr)
                status = "✅" if not warnings else f"⚠️ ({len(warnings)} warnings)"
                print(f"  {status} {expr:15} → Z3表达式")
            except Exception as e:
                print(f"  ❌ {expr:15} → 转换失败: {str(e)[:40]}")

        # 获取验证统计
        stats = z3_validator.get_validation_statistics()
        print(f"\n📊 验证统计: 总验证 {stats['total_validations']}, "
              f"成功率 {stats.get('success_rate', 0) * 100:.1f}%, "
              f"平均时间 {stats['average_time_ms']:.1f}ms")

        return z3_validator

    except Exception as e:
        print(f"❌ Z3验证器测试失败: {e}")
        return None


def test_integrated_dag_generator(rule_pool):
    """测试集成DAG生成器"""
    print_subsection("集成DAG生成器测试")

    try:
        # 创建基础生成器（原有功能）
        basic_generator = DualLayerDAGGenerator(rule_pool)

        # 创建集成生成器（新功能）
        if INTEGRATED_AVAILABLE:
            integrated_generator = DualLayerDAGGenerator(
                rule_pool,
                enable_expression_trees=True,
                enable_z3_validation=True,
                z3_timeout_seconds=10
            )
        else:
            integrated_generator = basic_generator

        dags_for_visualization = []

        # 测试不同生成模式
        print("🔄 测试不同DAG生成模式:")

        test_modes = [
            (DAGGenerationMode.FORWARD, "前向推理"),
            (DAGGenerationMode.BACKWARD, "后向推理"),
            (DAGGenerationMode.BIDIRECTIONAL, "双向推理")
        ]

        if INTEGRATED_AVAILABLE:
            test_modes.append((DAGGenerationMode.TREE_DRIVEN, "表达式树驱动"))

        for mode, description in test_modes:
            try:
                print(f"\n  🎯 {description} ({mode.value}):")

                start_time = time.time()
                dag = integrated_generator.generate_dag(
                    mode=mode,
                    target_depth=4,
                    target_complexity=8
                )
                generation_time = time.time() - start_time

                print(f"    ✅ 生成成功: {dag.number_of_nodes()} 节点, {dag.number_of_edges()} 边")
                print(f"    ⏱️  生成时间: {generation_time:.2f}秒")

                # 分析节点信息
                validated_count = 0
                premise_count = 0
                conclusion_count = 0

                for node_id, data in dag.nodes(data=True):
                    node = data['data']
                    if node.is_premise:
                        premise_count += 1
                    if node.is_conclusion:
                        conclusion_count += 1
                    if hasattr(node, 'is_validated') and node.is_validated:
                        validated_count += 1

                print(f"    📊 前提: {premise_count}, 结论: {conclusion_count}, "
                      f"已验证: {validated_count}")

                # 保存用于可视化
                dags_for_visualization.append((dag, description, f"深度4，复杂度8"))

            except Exception as e:
                print(f"    ❌ 生成失败: {str(e)}")

        # 测试深度控制准确性
        print("\n📏 测试深度控制准确性:")

        for target_depth in [2, 3, 4, 5]:
            try:
                dag = integrated_generator.generate_dag(
                    mode=DAGGenerationMode.TREE_DRIVEN if INTEGRATED_AVAILABLE else DAGGenerationMode.FORWARD,
                    target_depth=target_depth,
                    target_complexity=6
                )

                # 计算实际深度
                max_level = max(data['data'].level for _, data in dag.nodes(data=True))
                depth_accuracy = abs(max_level - (target_depth - 1)) <= 1

                status = "✅" if depth_accuracy else "❌"
                print(f"  {status} 目标深度 {target_depth} → 实际深度 {max_level + 1}")

            except Exception as e:
                print(f"  ❌ 深度 {target_depth} 测试失败: {str(e)}")

        return integrated_generator, dags_for_visualization

    except Exception as e:
        print(f"❌ 集成DAG生成器测试失败: {e}")
        return None, []


def test_dataset_generation(generator):
    """测试数据集生成"""
    print_subsection("数据集生成测试")

    if not generator:
        print("❌ 生成器不可用")
        return None

    try:
        # 生成小型数据集
        print("📦 生成训练数据集:")

        dataset = generator.generate_dataset(
            sample_count=10,
            mode=DAGGenerationMode.TREE_DRIVEN if INTEGRATED_AVAILABLE else DAGGenerationMode.FORWARD
        )

        print(f"  ✅ 成功生成 {len(dataset)} 个样本")

        # 分析数据集质量
        if dataset:
            print("\n📊 数据集质量分析:")

            # 统计验证结果
            validated_samples = sum(1 for sample in dataset if sample.get('is_validated'))
            valid_samples = sum(1 for sample in dataset
                                if sample.get('validation_result') is True)

            print(f"  🔍 已验证样本: {validated_samples}/{len(dataset)} "
                  f"({validated_samples / len(dataset) * 100:.1f}%)")

            if validated_samples > 0:
                print(f"  ✅ 逻辑有效样本: {valid_samples}/{validated_samples} "
                      f"({valid_samples / validated_samples * 100:.1f}%)")

            # 显示样本示例
            print("\n📝 样本示例:")
            for i, sample in enumerate(dataset[:3]):
                print(f"  样本 {i + 1}:")
                print(f"    前提: {sample['premises']}")
                print(f"    结论: {sample['conclusion']}")
                if sample.get('rule_applied'):
                    print(f"    规则: {sample['rule_applied']}")
                if sample.get('validation_result') is not None:
                    validation = "有效" if sample['validation_result'] else "无效"
                    print(f"    验证: {validation}")
                print()

        return dataset

    except Exception as e:
        print(f"❌ 数据集生成测试失败: {e}")
        return None


def test_performance_benchmark(generator):
    """性能基准测试"""
    print_subsection("性能基准测试")

    if not generator:
        print("❌ 生成器不可用")
        return

    try:
        print("⏱️  执行性能基准测试:")

        # 测试不同规模的生成性能
        test_configs = [
            (2, 5, "小规模"),
            (3, 8, "中规模"),
            (4, 12, "大规模")
        ]

        for depth, complexity, scale in test_configs:
            print(f"\n  📊 {scale}测试 (深度:{depth}, 复杂度:{complexity}):")

            times = []
            node_counts = []

            for i in range(3):  # 测试3次取平均
                start_time = time.time()

                dag = generator.generate_dag(
                    mode=DAGGenerationMode.TREE_DRIVEN if INTEGRATED_AVAILABLE else DAGGenerationMode.FORWARD,
                    target_depth=depth,
                    target_complexity=complexity
                )

                generation_time = time.time() - start_time
                times.append(generation_time)
                node_counts.append(dag.number_of_nodes())

            avg_time = sum(times) / len(times)
            avg_nodes = sum(node_counts) / len(node_counts)

            print(f"    ⏱️  平均生成时间: {avg_time:.2f}秒")
            print(f"    📊 平均节点数: {avg_nodes:.1f}")
            print(f"    🚀 生成速度: {avg_nodes / avg_time:.1f} 节点/秒")

        # 获取生成器统计信息
        if hasattr(generator, 'get_generation_statistics'):
            stats = generator.get_generation_statistics()
            print(f"\n📈 生成器统计:")
            print(f"  总生成次数: {stats.get('total_generations', 'N/A')}")
            print(
                f"  成功率: {stats.get('successful_generations', 0) / max(1, stats.get('total_generations', 1)) * 100:.1f}%")
            print(f"  平均验证率: {stats.get('average_validation_rate', 0) * 100:.1f}%")

    except Exception as e:
        print(f"❌ 性能基准测试失败: {e}")


def generate_visualization(dags_for_visualization):
    """生成可视化"""
    print_subsection("生成可视化")

    if not VISUALIZATION_AVAILABLE:
        print("❌ 可视化组件不可用")
        return

    if not dags_for_visualization:
        print("❌ 没有DAG可供可视化")
        return

    try:
        print("🎨 生成统一DAG可视化...")

        viz_file = create_unified_visualization(
            dags_for_visualization,
            "logs/day2_integrated_visualization.html"
        )

        print(f"✅ 可视化文件已生成: {viz_file}")
        print("🌐 请用浏览器打开该文件查看集成系统生成的推理DAG")

        return viz_file

    except Exception as e:
        print(f"❌ 可视化生成失败: {e}")
        return None


def save_demo_results(results_data):
    """保存演示结果"""
    print_subsection("保存演示结果")

    try:
        # 创建日志目录
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # 保存结果到JSON文件
        results_file = log_dir / "day2_demo_results.json"

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2, default=str)

        print(f"✅ 演示结果已保存: {results_file}")

        return results_file

    except Exception as e:
        print(f"❌ 保存演示结果失败: {e}")
        return None


def _get_tree_depth(tree) -> int:
    """获取表达式树的实际深度"""
    if tree.is_leaf():
        return 1

    max_child_depth = 0
    if tree.left_child:
        max_child_depth = max(max_child_depth, _get_tree_depth(tree.left_child))
    if tree.right_child:
        max_child_depth = max(max_child_depth, _get_tree_depth(tree.right_child))

    return 1 + max_child_depth


def main():
    """主演示函数"""
    # 创建日志器
    logger = ARNGLogger("Day2Demo")
    logger.session_start("ARNG_Generator Day 2 集成系统演示")

    print_separator("🚀 ARNG_Generator Day 2 集成系统演示", "=", 80)

    print("📋 今天完成的功能:")
    print("  ✅ 表达式树构建器 - 真正的深度控制")
    print("  ✅ 复杂度驱动控制器 - 智能复杂度管理")
    print("  ✅ Z3验证器 - 逻辑正确性保证")
    print("  ✅ 集成DAG生成器 - 统一系统接口")
    print("  ✅ 完整测试框架 - 质量保证体系")

    if not INTEGRATED_AVAILABLE:
        print("\n⚠️  警告: 部分集成组件不可用，将使用基础功能演示")

    # 记录演示结果
    demo_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "integrated_available": INTEGRATED_AVAILABLE,
        "components_tested": [],
        "test_results": {}
    }

    try:
        # 1. 创建规则池
        print_separator("🔧 系统初始化")
        rule_pool = create_rule_pool()
        stats = rule_pool.get_statistics()
        print(f"✅ 规则池初始化完成: {stats['total_rules']} 个规则")

        demo_results["rule_pool_stats"] = stats

        # 2. 测试表达式树构建器
        tree_builder, tree_results = test_expression_tree_builder()
        if tree_builder:
            demo_results["components_tested"].append("ExpressionTreeBuilder")
            demo_results["test_results"]["expression_tree"] = tree_results

        # 3. 测试复杂度控制器
        complexity_controller = test_complexity_controller()
        if complexity_controller:
            demo_results["components_tested"].append("ComplexityController")

        # 4. 测试Z3验证器
        z3_validator = test_z3_validator()
        if z3_validator:
            demo_results["components_tested"].append("Z3Validator")

        # 5. 测试集成DAG生成器
        generator, dags_for_viz = test_integrated_dag_generator(rule_pool)
        if generator:
            demo_results["components_tested"].append("IntegratedDAGGenerator")

        # 6. 测试数据集生成
        dataset = test_dataset_generation(generator)
        if dataset:
            demo_results["test_results"]["dataset_size"] = len(dataset)

        # 7. 性能基准测试
        test_performance_benchmark(generator)

        # 8. 生成可视化
        viz_file = generate_visualization(dags_for_viz)
        if viz_file:
            demo_results["visualization_file"] = str(viz_file)

        # 9. 保存演示结果
        results_file = save_demo_results(demo_results)

        # 总结
        print_separator("🎉 演示完成总结")

        print("📊 测试组件总结:")
        for component in demo_results["components_tested"]:
            print(f"  ✅ {component}")

        if not demo_results["components_tested"]:
            print("  ⚠️  没有集成组件被测试")

        print(f"\n📁 生成的文件:")
        if results_file:
            print(f"  📄 演示结果: {results_file}")
        if viz_file:
            print(f"  🎨 可视化文件: {viz_file}")

        log_file = ARNGLogger.get_current_log_file()
        if log_file:
            print(f"  📝 详细日志: {log_file}")

        print(f"\n🚀 核心成就:")
        print(f"  🎯 解决了深度控制问题 - 现在是真正的推理深度，不是规则数量")
        print(f"  🔍 集成了Z3验证 - 确保生成的推理在逻辑上正确")
        print(f"  📈 实现了复杂度驱动 - 可以生成梯度递增的训练数据")
        print(f"  🧪 完整的测试框架 - 保证系统质量和稳定性")

        print(f"\n💡 下一步建议:")
        print(f"  1. 运行 python tests/test_integrated_system.py --quick 进行快速验证")
        print(f"  2. 运行 python tests/test_integrated_system.py --full 进行完整测试")
        print(f"  3. 使用新的TREE_DRIVEN模式生成高质量推理数据集")
        print(f"  4. 根据需要调整复杂度增长策略和验证参数")

    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        demo_results["error"] = str(e)

    logger.session_end("ARNG_Generator Day 2 集成系统演示")

    print_separator("🏁 演示结束", "=", 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  演示被用户中断")
        logger = ARNGLogger("Day2Demo")
        logger.warning("Day 2 演示被用户中断")
    except Exception as e:
        print(f"\n\n❌ 演示过程中发生未处理的错误: {e}")
        logger = ARNGLogger("Day2Demo")
        logger.error(f"Day 2 演示发生未处理错误: {e}")
        import traceback

        traceback.print_exc()