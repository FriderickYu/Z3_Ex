"""
在原有DualLayerDAGGenerator基础上的最小化修改
只添加必要的功能，保持代码简洁
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import random
import networkx as nx
from enum import Enum

from rules.base.rule import BaseRule, LogicalFormula, RuleInstance, LogicalOperator
from rules.rule_pooling import StratifiedRulePool

from utils.logger_utils import ARNGLogger

# 新增：导入集成组件（可选）
try:
    from core.expression_tree_builder import ExpressionTreeBuilder, ExpressionNode
    from core.complexity_controller import ComplexityController, ComplexityGrowthStrategy
    from core.z3_validator import Z3Validator, Z3ValidationResult, ValidationResult

    INTEGRATED_COMPONENTS_AVAILABLE = True
except ImportError:
    INTEGRATED_COMPONENTS_AVAILABLE = False


class DAGGenerationMode(Enum):
    """DAG生成模式"""
    FORWARD = "forward"  # 前向推理
    BACKWARD = "backward"  # 后向推理
    BIDIRECTIONAL = "bidirectional"  # 双向推理
    TREE_DRIVEN = "tree_driven"  # 新增：表达式树驱动


@dataclass
class DAGNode:
    """DAG节点 - 在原有基础上添加可选字段"""
    node_id: str
    formula: LogicalFormula
    level: int  # 推理层级
    is_premise: bool = False
    is_conclusion: bool = False
    applied_rule: Optional[str] = None
    children: List[str] = None
    parents: List[str] = None

    # 新增：集成功能相关字段（可选）
    expression_tree: Optional[Any] = None  # ExpressionNode，避免循环导入
    tree_complexity: Dict[str, float] = field(default_factory=dict)
    validation_result: Optional[Any] = None  # Z3ValidationResult
    is_validated: bool = False

    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.parents is None:
            self.parents = []


@dataclass
class DAGEdge:
    """DAG边"""
    source: str
    target: str
    rule_id: str
    confidence: float = 1.0


class DualLayerDAGGenerator:
    """
    双层DAG生成器 - 核心架构组件
    支持前向/后向/双向推理的DAG构建
    新增：表达式树驱动模式和Z3验证
    """

    def __init__(self, rule_pool: StratifiedRulePool,
                 enable_expression_trees: bool = False,
                 enable_z3_validation: bool = False,
                 **config_kwargs):
        self.rule_pool = rule_pool
        self.logger = ARNGLogger("DAGGenerator")

        # 原有生成配置
        self.max_depth = 8
        self.max_width = 5
        self.min_premises = 2
        self.max_premises = 6

        # 原有复杂度控制
        self.complexity_growth_rate = 1.2
        self.tier_progression_prob = 0.7

        # 新增：集成组件配置
        self.enable_expression_trees = enable_expression_trees and INTEGRATED_COMPONENTS_AVAILABLE
        self.enable_z3_validation = enable_z3_validation and INTEGRATED_COMPONENTS_AVAILABLE

        # 新增：集成组件初始化
        self._init_integrated_components(**config_kwargs)

        self.logger.info(
            f"双层DAG生成器初始化完成 - 表达式树: {self.enable_expression_trees}, Z3验证: {self.enable_z3_validation}")

    def _init_integrated_components(self, **kwargs):
        """初始化集成组件"""
        self.tree_builder = None
        self.complexity_controller = None
        self.z3_validator = None

        if not INTEGRATED_COMPONENTS_AVAILABLE:
            if self.enable_expression_trees or self.enable_z3_validation:
                self.logger.warning("集成组件不可用，禁用相关功能")
            return

        # 初始化表达式树构建器
        if self.enable_expression_trees:
            self.tree_builder = ExpressionTreeBuilder()

        # 初始化复杂度控制器（总是初始化，因为很有用）
        self.complexity_controller = ComplexityController()

        # 初始化Z3验证器
        if self.enable_z3_validation:
            z3_timeout = kwargs.get('z3_timeout_seconds', 30)
            z3_strict = kwargs.get('z3_strict_mode', True)
            self.z3_validator = Z3Validator(
                timeout_seconds=z3_timeout,
                require_strict_validity=z3_strict
            )

    def generate_dag(self,
                     mode: DAGGenerationMode = DAGGenerationMode.BIDIRECTIONAL,
                     target_depth: int = 5,
                     target_complexity: int = 10,
                     seed_formulas: Optional[List[LogicalFormula]] = None) -> nx.DiGraph:
        """
        生成推理DAG

        Args:
            mode: 生成模式
            target_depth: 目标深度
            target_complexity: 目标复杂度
            seed_formulas: 种子公式（起始点）

        Returns:
            nx.DiGraph: 生成的DAG
        """
        self.logger.info(f"开始生成DAG - 模式: {mode.value}, 深度: {target_depth}, 复杂度: {target_complexity}")

        # 创建新的DAG
        dag = nx.DiGraph()

        # 初始化种子节点
        if seed_formulas is None:
            seed_formulas = self._generate_seed_formulas()

        # 根据模式生成DAG
        if mode == DAGGenerationMode.FORWARD:
            dag = self._generate_forward_dag(seed_formulas, target_depth, target_complexity)
        elif mode == DAGGenerationMode.BACKWARD:
            dag = self._generate_backward_dag(seed_formulas, target_depth, target_complexity)
        elif mode == DAGGenerationMode.TREE_DRIVEN:
            # 新增：表达式树驱动模式
            dag = self._generate_tree_driven_dag(seed_formulas, target_depth, target_complexity)
        else:  # BIDIRECTIONAL
            dag = self._generate_bidirectional_dag(seed_formulas, target_depth, target_complexity)

        # 验证和优化DAG
        self._validate_dag(dag)
        self._optimize_dag(dag)

        self.logger.info(f"DAG生成完成 - 节点数: {dag.number_of_nodes()}, 边数: {dag.number_of_edges()}")
        return dag

    def _generate_tree_driven_dag(self, seed_formulas: List[LogicalFormula],
                                  target_depth: int, target_complexity: int) -> nx.DiGraph:
        """
        新增：表达式树驱动的DAG生成
        解决深度控制问题的关键方法
        """
        if not self.enable_expression_trees or not self.tree_builder:
            # 如果没有表达式树功能，回退到前向推理
            self.logger.warning("表达式树功能未启用，使用前向推理")
            return self._generate_forward_dag(seed_formulas, target_depth, target_complexity)

        self.logger.info("使用表达式树驱动模式生成DAG")
        dag = nx.DiGraph()
        current_level = 0

        # 第一层：添加种子节点（前提）
        active_nodes = []
        for i, formula in enumerate(seed_formulas):
            node = DAGNode(
                node_id=f"premise_{i}",
                formula=formula,
                level=current_level,
                is_premise=True
            )
            dag.add_node(node.node_id, data=node)
            active_nodes.append(node)

        # 逐层构建，每层基于表达式树
        while current_level < target_depth - 1 and active_nodes:
            current_level += 1

            # 生成当前层的复杂度目标
            if self.complexity_controller:
                complexity_target = self.complexity_controller.generate_complexity_target(
                    step=current_level
                )
                target_overall = complexity_target.get('overall', target_complexity)
            else:
                target_overall = target_complexity * (1.2 ** current_level)

            self.logger.debug(f"生成第 {current_level} 层 - 复杂度目标: {target_overall:.2f}")

            # 生成当前层的节点
            new_nodes = self._generate_tree_level_nodes(
                active_nodes, current_level, target_overall, dag
            )

            if not new_nodes:
                self.logger.warning(f"第 {current_level} 层没有生成新节点，提前结束")
                break

            active_nodes = new_nodes

        # 标记最终结论
        if active_nodes:
            # 选择复杂度最高的节点作为结论
            conclusion_node = max(active_nodes,
                                  key=lambda n: n.tree_complexity.get('overall', 0) if n.tree_complexity else 0)
            conclusion_node.is_conclusion = True
            dag.nodes[conclusion_node.node_id]['data'] = conclusion_node

        return dag

    def _generate_tree_level_nodes(self, parent_nodes: List[DAGNode], level: int,
                                   target_complexity: float, dag: nx.DiGraph) -> List[DAGNode]:
        """为指定层级生成基于表达式树的节点"""
        new_nodes = []
        target_count = min(self.max_width, max(2, len(parent_nodes)))

        for i in range(target_count):
            try:
                # 生成表达式树
                tree_depth = min(2 + level // 2, 5)  # 深度随层级增加
                complexity_factor = 1.0 + level * 0.2  # 复杂度因子随层级增加

                expression_tree = self.tree_builder.build_expression_tree(
                    target_depth=tree_depth,
                    complexity_factor=complexity_factor,
                    variable_limit=5,
                    prefer_balanced=True
                )

                # 计算复杂度
                tree_complexity = {}
                if self.complexity_controller:
                    tree_complexity = self.complexity_controller.calculate_expression_tree_complexity(expression_tree)

                # 转换为LogicalFormula
                formula = self.tree_builder.convert_to_logical_formula(expression_tree)

                # 创建节点
                node = DAGNode(
                    node_id=f"node_{level}_{i}",
                    formula=formula,
                    level=level,
                    applied_rule="expression_tree_derivation",
                    expression_tree=expression_tree,
                    tree_complexity=tree_complexity
                )

                # Z3验证（如果启用）
                if self.enable_z3_validation and self.z3_validator:
                    premises = [p.formula.expression for p in parent_nodes[:2]]
                    validation_result = self.z3_validator.validate_reasoning_step(
                        premises, formula.expression
                    )
                    node.validation_result = validation_result
                    node.is_validated = True

                    # 如果验证失败且在严格模式下，跳过此节点
                    if not validation_result.is_valid:
                        self.logger.debug(f"节点验证失败，跳过: {formula.expression}")
                        continue

                # 添加到DAG
                dag.add_node(node.node_id, data=node)

                # 连接到父节点
                selected_parents = parent_nodes[:2] if len(parent_nodes) <= 2 else random.sample(parent_nodes, 2)
                for parent in selected_parents:
                    edge = DAGEdge(
                        source=parent.node_id,
                        target=node.node_id,
                        rule_id="expression_tree_derivation"
                    )
                    dag.add_edge(parent.node_id, node.node_id, data=edge)

                    # 更新关系
                    node.parents.append(parent.node_id)
                    parent.children.append(node.node_id)
                    dag.nodes[parent.node_id]['data'] = parent

                new_nodes.append(node)
                self.logger.debug(f"成功生成节点: {formula.expression}")

            except Exception as e:
                self.logger.warning(f"生成节点失败: {str(e)}")
                continue

        self.logger.info(f"第 {level} 层生成了 {len(new_nodes)} 个节点")
        return new_nodes

    # ==================== 原有方法保持完全不变 ====================

    def _generate_seed_formulas(self) -> List[LogicalFormula]:
        """生成种子公式"""
        num_seeds = random.randint(self.min_premises, self.max_premises)
        seeds = []

        # 预定义一些基础公式
        basic_formulas = [
            LogicalFormula(
                expression="P",
                variables={"P"},
                operators=[],
                complexity=1
            ),
            LogicalFormula(
                expression="Q",
                variables={"Q"},
                operators=[],
                complexity=1
            ),
            LogicalFormula(
                expression="R",
                variables={"R"},
                operators=[],
                complexity=1
            ),
            LogicalFormula(
                expression="P → Q",
                variables={"P", "Q"},
                operators=[LogicalOperator.IMPLIES],
                complexity=2
            ),
            LogicalFormula(
                expression="Q → R",
                variables={"Q", "R"},
                operators=[LogicalOperator.IMPLIES],
                complexity=2
            )
        ]

        # 随机选择种子公式
        for _ in range(min(num_seeds, len(basic_formulas))):
            if basic_formulas:
                formula = basic_formulas.pop(random.randint(0, len(basic_formulas) - 1))
                seeds.append(formula)

        self.logger.debug(f"生成 {len(seeds)} 个种子公式")
        return seeds

    def _generate_forward_dag(self, seed_formulas: List[LogicalFormula],
                              target_depth: int, target_complexity: int) -> nx.DiGraph:
        """前向推理生成DAG"""
        dag = nx.DiGraph()
        current_level = 0

        # 添加种子节点（前提）
        active_formulas = []
        for i, formula in enumerate(seed_formulas):
            node_id = f"premise_{i}"
            node = DAGNode(
                node_id=node_id,
                formula=formula,
                level=current_level,
                is_premise=True
            )
            dag.add_node(node_id, data=node)
            active_formulas.append(formula)

        # 前向推理扩展
        while current_level < target_depth and active_formulas:
            current_level += 1
            new_formulas = []

            # 获取当前层级可用的规则
            tier_limit = min(5, 1 + current_level // 2)  # 逐渐增加可用规则层级
            applicable_rules = self.rule_pool.get_applicable_rules(active_formulas, tier_limit)

            if not applicable_rules:
                self.logger.warning(f"在层级 {current_level} 没有找到可应用的规则")
                break

            # 应用规则生成新节点
            attempts = 0
            max_attempts = min(self.max_width, len(applicable_rules))

            while attempts < max_attempts and len(new_formulas) < self.max_width:
                rule = self.rule_pool.select_rule(applicable_rules, target_complexity)
                if rule and rule.can_apply(active_formulas):
                    try:
                        conclusions = rule.apply(active_formulas)
                        for conclusion in conclusions:
                            node_id = f"node_{current_level}_{len(new_formulas)}"
                            node = DAGNode(
                                node_id=node_id,
                                formula=conclusion,
                                level=current_level,
                                applied_rule=rule.rule_id
                            )

                            # 新增：可选的Z3验证
                            if self.enable_z3_validation and self.z3_validator:
                                premises_for_validation = [f.expression for f in active_formulas[:2]]
                                validation_result = self.z3_validator.validate_reasoning_step(
                                    premises_for_validation, conclusion.expression
                                )
                                node.validation_result = validation_result
                                node.is_validated = True

                            dag.add_node(node_id, data=node)

                            # 添加边
                            premise_nodes = [n for n, d in dag.nodes(data=True)
                                             if d['data'].level == current_level - 1]
                            for premise_node in premise_nodes[:2]:  # 限制前提数量
                                edge = DAGEdge(
                                    source=premise_node,
                                    target=node_id,
                                    rule_id=rule.rule_id
                                )
                                dag.add_edge(premise_node, node_id, data=edge)

                            new_formulas.append(conclusion)
                            self.logger.debug(f"成功应用规则 {rule.rule_id}: {conclusion.expression}")

                            # 更新规则统计
                            rule.update_statistics(True)

                            # 限制每次生成的结论数量
                            if len(new_formulas) >= self.max_width:
                                break

                    except Exception as e:
                        self.logger.error(f"应用规则 {rule.rule_id} 时出错: {e}")
                        rule.update_statistics(False)

                attempts += 1

            if not new_formulas:
                self.logger.warning(f"层级 {current_level} 没有生成新的公式，停止扩展")
                break

            active_formulas = new_formulas

        # 标记最终结论
        if active_formulas:
            final_nodes = [n for n, d in dag.nodes(data=True)
                           if d['data'].level == current_level]
            if final_nodes:
                final_node = random.choice(final_nodes)
                dag.nodes[final_node]['data'].is_conclusion = True

        return dag

    def _generate_backward_dag(self, target_formulas: List[LogicalFormula],
                               target_depth: int, target_complexity: int) -> nx.DiGraph:
        """后向推理生成DAG"""
        dag = nx.DiGraph()
        current_level = target_depth

        # 添加目标节点（结论）
        active_formulas = []
        for i, formula in enumerate(target_formulas):
            node_id = f"conclusion_{i}"
            node = DAGNode(
                node_id=node_id,
                formula=formula,
                level=current_level,
                is_conclusion=True
            )
            dag.add_node(node_id, data=node)
            active_formulas.append(formula)

        # 后向推理分解
        while current_level > 0 and active_formulas:
            current_level -= 1
            new_formulas = []

            for formula in active_formulas:
                # 寻找能推出当前公式的规则
                rules = self._find_rules_for_conclusion(formula)
                if rules:
                    rule = random.choice(rules)
                    try:
                        # 生成前提
                        premises = self._generate_premises_for_rule(rule, formula)
                        for i, premise in enumerate(premises):
                            node_id = f"node_{current_level}_{len(new_formulas)}"
                            node = DAGNode(
                                node_id=node_id,
                                formula=premise,
                                level=current_level,
                                is_premise=(current_level == 0)
                            )
                            dag.add_node(node_id, data=node)

                            # 添加边（从前提到结论）
                            target_nodes = [n for n, d in dag.nodes(data=True)
                                            if d['data'].level == current_level + 1 and
                                            d['data'].formula.expression == formula.expression]
                            for target_node in target_nodes:
                                edge = DAGEdge(
                                    source=node_id,
                                    target=target_node,
                                    rule_id=rule.rule_id
                                )
                                dag.add_edge(node_id, target_node, data=edge)

                            new_formulas.append(premise)
                    except Exception as e:
                        self.logger.error(f"后向应用规则 {rule.rule_id} 时出错: {e}")

            active_formulas = new_formulas

        return dag

    def _generate_bidirectional_dag(self, seed_formulas: List[LogicalFormula],
                                    target_depth: int, target_complexity: int) -> nx.DiGraph:
        """双向推理生成DAG"""
        # 先进行前向推理
        forward_dag = self._generate_forward_dag(seed_formulas, target_depth // 2, target_complexity)

        # 获取前向推理的结论作为后向推理的起点
        conclusion_nodes = [n for n, d in forward_dag.nodes(data=True)
                            if d['data'].is_conclusion]

        if conclusion_nodes:
            conclusion_formulas = [forward_dag.nodes[n]['data'].formula for n in conclusion_nodes]
            # 进行后向推理
            backward_dag = self._generate_backward_dag(conclusion_formulas, target_depth // 2, target_complexity)

            # 合并两个DAG
            combined_dag = nx.compose(forward_dag, backward_dag)
            return combined_dag

        return forward_dag

    def _find_rules_for_conclusion(self, conclusion: LogicalFormula) -> List[BaseRule]:
        """寻找能推出指定结论的规则"""
        applicable_rules = []

        for tier in range(1, 6):
            rules = self.rule_pool.get_rules_by_tier(tier)
            for rule in rules:
                template = rule.get_template()
                if self._can_produce_conclusion(rule, conclusion, template):
                    applicable_rules.append(rule)

        return applicable_rules

    def _can_produce_conclusion(self, rule: BaseRule, conclusion: LogicalFormula,
                                template: Dict[str, Any]) -> bool:
        """检查规则是否能产生指定结论"""
        # 简化实现：基于模板匹配
        conclusion_patterns = template.get('conclusion_patterns', [])
        for pattern in conclusion_patterns:
            if self._matches_pattern(conclusion.expression, pattern):
                return True
        return False

    def _matches_pattern(self, expression: str, pattern: str) -> bool:
        """检查表达式是否匹配模式"""
        # 简化的模式匹配（实际应该更复杂）
        return len(expression) > 0 and len(pattern) > 0

    def _generate_premises_for_rule(self, rule: BaseRule, conclusion: LogicalFormula) -> List[LogicalFormula]:
        """为规则生成前提以推出指定结论"""
        template = rule.get_template()
        premise_patterns = template.get('premise_patterns', [])

        premises = []
        for pattern in premise_patterns:
            premise = self._instantiate_premise_from_pattern(pattern, conclusion)
            if premise:
                premises.append(premise)

        return premises

    def _instantiate_premise_from_pattern(self, pattern: str, conclusion: LogicalFormula) -> Optional[LogicalFormula]:
        """从模式实例化前提"""
        # 简化实现
        variables = conclusion.variables.copy()
        if variables:
            var = list(variables)[0]
            expression = pattern.replace("{var}", var)
            return LogicalFormula(
                expression=expression,
                variables={var},
                operators=[],
                complexity=1
            )
        return None

    def _validate_dag(self, dag: nx.DiGraph) -> bool:
        """验证DAG的有效性"""
        # 检查是否为有向无环图
        if not nx.is_directed_acyclic_graph(dag):
            self.logger.error("生成的图包含环路")
            return False

        # 检查节点数据完整性
        for node_id, data in dag.nodes(data=True):
            if 'data' not in data or not isinstance(data['data'], DAGNode):
                self.logger.error(f"节点 {node_id} 数据不完整")
                return False

        # 检查边数据完整性
        for source, target, data in dag.edges(data=True):
            if 'data' not in data or not isinstance(data['data'], DAGEdge):
                self.logger.error(f"边 ({source}, {target}) 数据不完整")
                return False

        self.logger.debug("DAG验证通过")
        return True

    def _optimize_dag(self, dag: nx.DiGraph):
        """优化DAG结构"""
        # 移除孤立节点
        isolated_nodes = list(nx.isolates(dag))
        if isolated_nodes:
            dag.remove_nodes_from(isolated_nodes)
            self.logger.debug(f"移除 {len(isolated_nodes)} 个孤立节点")

        # 合并重复节点
        self._merge_duplicate_nodes(dag)

    def _merge_duplicate_nodes(self, dag: nx.DiGraph):
        """合并重复节点"""
        # 按公式表达式分组节点
        expression_groups = {}
        for node_id, data in dag.nodes(data=True):
            expression = data['data'].formula.expression
            if expression not in expression_groups:
                expression_groups[expression] = []
            expression_groups[expression].append(node_id)

        # 合并重复节点
        for expression, nodes in expression_groups.items():
            if len(nodes) > 1:
                # 保留第一个节点，合并其他节点的连接
                primary_node = nodes[0]
                for duplicate_node in nodes[1:]:
                    # 转移入边
                    for predecessor in dag.predecessors(duplicate_node):
                        if not dag.has_edge(predecessor, primary_node):
                            edge_data = dag[predecessor][duplicate_node]['data']
                            dag.add_edge(predecessor, primary_node, data=edge_data)

                    # 转移出边
                    for successor in dag.successors(duplicate_node):
                        if not dag.has_edge(primary_node, successor):
                            edge_data = dag[duplicate_node][successor]['data']
                            dag.add_edge(primary_node, successor, data=edge_data)

                    # 移除重复节点
                    dag.remove_node(duplicate_node)

                if len(nodes) > 1:
                    self.logger.debug(f"合并了 {len(nodes) - 1} 个重复节点 (表达式: {expression})")

    # ==================== 新增：便捷方法 ====================

    def generate_dataset(self, sample_count: int, mode: DAGGenerationMode = DAGGenerationMode.TREE_DRIVEN) -> List[
        Dict[str, Any]]:
        """生成训练数据集"""
        dataset = []

        for i in range(sample_count):
            try:
                dag = self.generate_dag(mode=mode)
                samples = self._extract_samples_from_dag(dag)
                dataset.extend(samples)
            except Exception as e:
                self.logger.warning(f"样本 {i} 生成失败: {str(e)}")

        self.logger.info(f"数据集生成完成 - 样本数: {len(dataset)}")
        return dataset

    def _extract_samples_from_dag(self, dag: nx.DiGraph) -> List[Dict[str, Any]]:
        """从DAG中提取训练样本"""
        samples = []

        for node_id, data in dag.nodes(data=True):
            node = data['data']

            if not node.is_premise and node.parents:  # 非前提且有父节点
                # 获取前提
                premises = []
                for parent_id in node.parents:
                    if parent_id in dag.nodes:
                        parent_node = dag.nodes[parent_id]['data']
                        premises.append(parent_node.formula.expression)

                sample = {
                    'premises': premises,
                    'conclusion': node.formula.expression,
                    'rule_applied': node.applied_rule,
                    'is_validated': node.is_validated,
                    'validation_result': node.validation_result.is_valid if node.validation_result else None
                }
                samples.append(sample)

        return samples


# ==================== 工厂方法 ====================

def create_integrated_generator(rule_pool: StratifiedRulePool,
                                enable_all: bool = True,
                                **kwargs) -> DualLayerDAGGenerator:
    """创建集成功能的DAG生成器"""
    return DualLayerDAGGenerator(
        rule_pool=rule_pool,
        enable_expression_trees=enable_all,
        enable_z3_validation=enable_all,
        **kwargs
    )


def create_basic_generator(rule_pool: StratifiedRulePool) -> DualLayerDAGGenerator:
    """创建基础版本（不启用集成功能）"""
    return DualLayerDAGGenerator(
        rule_pool=rule_pool,
        enable_expression_trees=False,
        enable_z3_validation=False
    )


if __name__ == "__main__":
    # 测试示例
    from rules.rule_pooling import StratifiedRulePool
    from rules.tiers.tier1_axioms import ModusPonensRule, ConjunctionIntroductionRule

    # 创建规则池
    rule_pool = StratifiedRulePool()
    rule_pool.register_rule(ModusPonensRule())
    rule_pool.register_rule(ConjunctionIntroductionRule())

    print("=== 最小化集成DAG生成器测试 ===")

    # 测试1：基础模式
    print("\n1. 基础模式测试:")
    basic_generator = create_basic_generator(rule_pool)
    basic_dag = basic_generator.generate_dag(mode=DAGGenerationMode.FORWARD, target_depth=3)
    print(f"基础DAG: {basic_dag.number_of_nodes()} 节点, {basic_dag.number_of_edges()} 边")

    # 测试2：集成模式（如果组件可用）
    if INTEGRATED_COMPONENTS_AVAILABLE:
        print("\n2. 集成模式测试:")
        integrated_generator = create_integrated_generator(rule_pool)

        # 测试表达式树驱动模式
        tree_dag = integrated_generator.generate_dag(mode=DAGGenerationMode.TREE_DRIVEN, target_depth=4)
        print(f"表达式树DAG: {tree_dag.number_of_nodes()} 节点, {tree_dag.number_of_edges()} 边")

        # 显示节点信息
        for node_id, data in tree_dag.nodes(data=True):
            node = data['data']
            validation_info = ""
            if node.is_validated:
                validation_info = f" (验证: {'通过' if node.validation_result and node.validation_result.is_valid else '失败'})"
            print(f"  节点 {node_id}: {node.formula.expression} [层级: {node.level}]{validation_info}")

        # 测试数据集生成
        print("\n3. 数据集生成测试:")
        dataset = integrated_generator.generate_dataset(sample_count=3, mode=DAGGenerationMode.TREE_DRIVEN)
        for i, sample in enumerate(dataset[:3]):  # 只显示前3个样本
            print(f"  样本 {i + 1}: {sample['premises']} → {sample['conclusion']}")
    else:
        print("\n2. 集成组件不可用，跳过集成模式测试")

    print("\n测试完成！")