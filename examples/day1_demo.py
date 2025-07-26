"""
Day 1 演示脚本 - 统一可视化版本
生成一个HTML文件展示所有推理模式的DAG
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# 导入核心模块
from core.dag_generator import DualLayerDAGGenerator, DAGGenerationMode
from core.complexity_controller import ComplexityController
from rules.rule_pooling import StratifiedRulePool
from rules.base.rule import LogicalFormula, LogicalOperator
from utils.logger_utils import ARNGLogger

# 导入规则类
from rules.tiers.tier1_axioms import (
    ModusPonensRule,
    ModusTollensRule,
    HypotheticalSyllogismRule,
    DisjunctiveSyllogismRule,
    ConjunctionIntroductionRule,
    ConjunctionEliminationRule
)

# 导入统一可视化模块
from dag_visualizer import create_unified_visualization


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


def create_default_generator():
    """创建默认生成器"""
    rule_pool = StratifiedRulePool()
    tier1_rules = get_tier1_rules()
    for rule in tier1_rules:
        rule_pool.register_rule(rule)

    dag_generator = DualLayerDAGGenerator(rule_pool)
    complexity_controller = ComplexityController()

    return dag_generator, complexity_controller, rule_pool


def print_separator(title):
    """打印分隔线"""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print('=' * 60)


def main():
    """主演示函数"""
    # 创建日志器
    logger = ARNGLogger("Day1Demo")
    logger.session_start("ARNG_Generator_v2 统一可视化演示")

    print_separator("ARNG_Generator_v2 统一可视化演示")
    logger.info("演示系统正在启动...")

    # 系统初始化
    try:
        dag_generator, complexity_controller, rule_pool = create_default_generator()
        stats = rule_pool.get_statistics()
        print(f"✅ 系统初始化成功: {stats['total_rules']} 个规则")
        logger.info(f"系统初始化完成，注册了 {stats['total_rules']} 个规则")
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        logger.error(f"系统初始化失败: {e}")
        return

    # 收集所有DAG
    dags_to_visualize = []

    print("\n🔄 正在生成不同模式的推理DAG...")

    # 1. 基础DAG生成
    try:
        print("  📊 生成基础DAG...")
        basic_dag = dag_generator.generate_dag(
            mode=DAGGenerationMode.FORWARD,
            target_depth=3,
            target_complexity=6
        )
        dags_to_visualize.append((
            basic_dag,
            "📊 基础DAG生成",
            f"前向推理，深度3，复杂度6 - 节点: {basic_dag.number_of_nodes()}, 边: {basic_dag.number_of_edges()}"
        ))
        print(f"     ✅ 完成: {basic_dag.number_of_nodes()} 节点, {basic_dag.number_of_edges()} 边")
        logger.info(f"基础DAG生成完成: {basic_dag.number_of_nodes()} 节点, {basic_dag.number_of_edges()} 边")
    except Exception as e:
        print(f"     ❌ 失败: {e}")
        logger.error(f"基础DAG生成失败: {e}")

    # 2. 不同推理模式
    modes = [
        (DAGGenerationMode.FORWARD, "🔄 前向推理", "从前提出发，逐步推导结论"),
        (DAGGenerationMode.BACKWARD, "🔙 后向推理", "从目标出发，反向寻找前提"),
        (DAGGenerationMode.BIDIRECTIONAL, "↔️ 双向推理", "结合前向和后向推理")
    ]

    for mode, title, description in modes:
        try:
            print(f"  {title}...")
            dag = dag_generator.generate_dag(
                mode=mode,
                target_depth=2,
                target_complexity=4
            )
            full_description = f"{description} - 节点: {dag.number_of_nodes()}, 边: {dag.number_of_edges()}"
            dags_to_visualize.append((dag, title, full_description))
            print(f"     ✅ 完成: {dag.number_of_nodes()} 节点, {dag.number_of_edges()} 边")
            logger.info(f"{title} 完成: {dag.number_of_nodes()} 节点, {dag.number_of_edges()} 边")
        except Exception as e:
            print(f"     ❌ 失败: {e}")
            logger.warning(f"{title} 失败: {e}")

    # 3. 生成统一可视化
    if dags_to_visualize:
        print(f"\n🎨 正在生成统一可视化文件...")
        try:
            viz_file = create_unified_visualization(
                dags_to_visualize,
                "logs/dag_unified_visualization.html"
            )
            print(f"✅ 统一可视化文件已生成: {viz_file}")
            print(f"🌐 请用浏览器打开该文件查看所有推理DAG")
            logger.info(f"统一可视化文件生成成功: {viz_file}")
        except Exception as e:
            print(f"❌ 可视化生成失败: {e}")
            logger.error(f"可视化生成失败: {e}")
    else:
        print("⚠️  没有成功的DAG可供可视化")

    # 4. 复杂度分析演示
    print(f"\n📐 复杂度分析演示...")
    test_formulas = [
        ("P", LogicalFormula("P", {"P"}, [], 1)),
        ("P → Q", LogicalFormula("P → Q", {"P", "Q"}, [LogicalOperator.IMPLIES], 2)),
        ("(P ∧ Q) → R", LogicalFormula("(P ∧ Q) → R", {"P", "Q", "R"},
                                     [LogicalOperator.AND, LogicalOperator.IMPLIES], 3))
    ]

    for name, formula in test_formulas:
        try:
            complexity = complexity_controller.calculate_formula_complexity(formula)
            print(f"  📊 {name}: 复杂度 {complexity['overall']:.2f}")
            logger.debug(f"复杂度分析: {name} = {complexity['overall']:.2f}")
        except Exception as e:
            print(f"  ❌ {name}: 分析失败 {e}")

    # 5. 复杂度递增演示
    print(f"\n📈 复杂度递增演示...")
    initial_complexity = 2.0
    for step in range(5):
        try:
            target_complexity = complexity_controller.get_target_complexity_for_step(
                step, initial_complexity
            )
            print(f"  步骤 {step}: 目标复杂度 {target_complexity:.2f}")

            # 模拟记录复杂度
            mock_complexity = {
                'structural': target_complexity * 0.3,
                'semantic': target_complexity * 0.25,
                'computational': target_complexity * 0.25,
                'cognitive': target_complexity * 0.2,
                'overall': target_complexity
            }
            complexity_controller.record_complexity(step, mock_complexity)
            logger.debug(f"步骤 {step} 复杂度记录: {target_complexity:.2f}")

        except Exception as e:
            print(f"  ❌ 步骤 {step} 失败: {e}")
            logger.error(f"步骤 {step} 复杂度计算失败: {e}")

    # 显示复杂度统计
    try:
        stats = complexity_controller.get_complexity_statistics()
        print(f"\n📊 复杂度统计: 总步骤 {stats['total_steps']}, 平均复杂度 {stats['overall']['mean']:.2f}")
        logger.info(f"复杂度统计完成: {stats['total_steps']} 步骤")
    except Exception as e:
        print(f"❌ 复杂度统计失败: {e}")
        logger.error(f"复杂度统计失败: {e}")

    # 6. 显示生成的文件
    print_separator("生成的文件")
    log_file = ARNGLogger.get_current_log_file()
    print(f"📝 系统日志: {log_file}")
    print(f"🎨 统一可视化: logs/dag_unified_visualization.html")
    print(f"\n💡 使用浏览器打开HTML文件查看完整的推理DAG可视化")

    # 结束演示
    print_separator("演示完成")
    print("🎉 ARNG_Generator_v2 统一可视化演示完成!")
    print("📂 查看生成的文件:")
    print("   • 详细日志 → 用文本编辑器打开 .log 文件")
    print("   • 推理图谱 → 用浏览器打开 .html 文件")
    print("\n🔍 推理模式对比:")
    print("   • 前向推理: 从已知前提推导结论")
    print("   • 后向推理: 从目标反推所需前提")
    print("   • 双向推理: 结合两种方法的优势")

    logger.session_end("ARNG_Generator_v2 统一可视化演示")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger = ARNGLogger("Day1Demo")
        logger.warning("演示被用户中断")
        print("\n\n⚠️  演示被用户中断")
    except Exception as e:
        logger = ARNGLogger("Day1Demo")
        logger.error(f"演示过程中发生错误: {e}")
        print(f"\n\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()