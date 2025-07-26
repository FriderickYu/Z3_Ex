"""
表达式树构建算法
实现递归的逻辑表达式树生成，支持真正的深度控制
"""

from typing import List, Dict, Any, Optional, Union, Set
from dataclasses import dataclass
from enum import Enum
import random
import math

from rules.base.rule import LogicalFormula, LogicalOperator

try:
    from ..utils.logger_utils import ARNGLogger
except ValueError:
    from utils.logger_utils import ARNGLogger


class ExpressionType(Enum):
    """表达式类型"""
    ATOMIC = "atomic"  # 原子表达式: P, Q, R
    UNARY = "unary"  # 一元表达式: ¬P
    BINARY = "binary"  # 二元表达式: P ∧ Q, P → Q
    QUANTIFIED = "quantified"  # 量化表达式: ∀x P(x)


class ConnectorType(Enum):
    """逻辑连接符类型"""
    CONJUNCTION = "∧"  # 合取
    DISJUNCTION = "∨"  # 析取
    IMPLICATION = "→"  # 蕴含
    BICONDITIONAL = "↔"  # 双条件
    NEGATION = "¬"  # 否定


@dataclass
class ExpressionNode:
    """表达式树节点"""
    node_id: str
    expression_type: ExpressionType
    connector: Optional[ConnectorType] = None
    content: str = ""

    # 树结构
    left_child: Optional['ExpressionNode'] = None
    right_child: Optional['ExpressionNode'] = None
    parent: Optional['ExpressionNode'] = None

    # 属性
    depth: int = 0
    complexity: int = 0
    variables: Set[str] = None

    def __post_init__(self):
        if self.variables is None:
            self.variables = set()

    def is_leaf(self) -> bool:
        """判断是否为叶子节点"""
        return self.left_child is None and self.right_child is None

    def get_expression_string(self) -> str:
        """获取表达式字符串表示"""
        if self.expression_type == ExpressionType.ATOMIC:
            return self.content
        elif self.expression_type == ExpressionType.UNARY:
            if self.left_child:
                return f"{self.connector.value}{self.left_child.get_expression_string()}"
        elif self.expression_type == ExpressionType.BINARY:
            if self.left_child and self.right_child:
                left_expr = self.left_child.get_expression_string()
                right_expr = self.right_child.get_expression_string()
                return f"({left_expr} {self.connector.value} {right_expr})"

        return self.content


class ExpressionTreeBuilder:
    """
    表达式树构建器
    实现复杂度驱动的递归表达式树生成
    """

    def __init__(self, variable_pool: List[str] = None):
        self.logger = ARNGLogger("ExpressionTreeBuilder")

        # 变量池
        self.variable_pool = variable_pool or [
            'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ]

        # 复杂度配置
        self.complexity_weights = {
            ExpressionType.ATOMIC: 1,
            ExpressionType.UNARY: 2,
            ExpressionType.BINARY: 3,
            ExpressionType.QUANTIFIED: 5
        }

        # 连接符概率分布
        self.connector_probabilities = {
            ConnectorType.CONJUNCTION: 0.25,
            ConnectorType.DISJUNCTION: 0.25,
            ConnectorType.IMPLICATION: 0.30,
            ConnectorType.BICONDITIONAL: 0.15,
            ConnectorType.NEGATION: 0.05
        }

        # 构建统计
        self.build_statistics = {
            'total_trees': 0,
            'avg_depth': 0.0,
            'avg_complexity': 0.0,
            'node_type_counts': {t: 0 for t in ExpressionType}
        }

        self.logger.info("表达式树构建器初始化完成")

    def build_expression_tree(self,
                              target_depth: int,
                              complexity_factor: float = 1.0,
                              variable_limit: int = 5,
                              prefer_balanced: bool = True) -> ExpressionNode:
        """
        构建表达式树

        Args:
            target_depth: 目标深度（真正的树深度）
            complexity_factor: 复杂度因子
            variable_limit: 变量使用限制
            prefer_balanced: 是否偏好平衡树

        Returns:
            ExpressionNode: 根节点
        """
        self.logger.info(f"开始构建表达式树 - 深度: {target_depth}, 复杂度因子: {complexity_factor}")

        # 选择可用变量
        available_vars = self.variable_pool[:variable_limit]

        # 递归构建
        root = self._build_recursive(
            target_depth=target_depth,
            current_depth=0,
            available_vars=available_vars,
            complexity_factor=complexity_factor,
            prefer_balanced=prefer_balanced
        )

        # 计算最终属性
        self._calculate_tree_properties(root)

        # 更新统计信息
        self._update_statistics(root)

        self.logger.info(f"表达式树构建完成 - 根表达式: {root.get_expression_string()}")
        return root

    def _build_recursive(self,
                         target_depth: int,
                         current_depth: int,
                         available_vars: List[str],
                         complexity_factor: float,
                         prefer_balanced: bool) -> ExpressionNode:
        """递归构建表达式树"""

        # 基础情况：达到目标深度或只剩1层，生成原子表达式
        if current_depth >= target_depth or target_depth <= 1:
            return self._create_atomic_node(available_vars, current_depth)

        # 决定节点类型
        node_type = self._choose_node_type(current_depth, target_depth, complexity_factor)

        if node_type == ExpressionType.ATOMIC:
            return self._create_atomic_node(available_vars, current_depth)

        elif node_type == ExpressionType.UNARY:
            return self._create_unary_node(
                target_depth, current_depth, available_vars,
                complexity_factor, prefer_balanced
            )

        elif node_type == ExpressionType.BINARY:
            return self._create_binary_node(
                target_depth, current_depth, available_vars,
                complexity_factor, prefer_balanced
            )

        else:
            # 如果是其他类型，默认创建二元节点
            return self._create_binary_node(
                target_depth, current_depth, available_vars,
                complexity_factor, prefer_balanced
            )

    def _choose_node_type(self, current_depth: int, target_depth: int,
                          complexity_factor: float) -> ExpressionType:
        """选择节点类型"""
        remaining_depth = target_depth - current_depth

        # 如果只剩1层，必须是原子
        if remaining_depth <= 1:
            return ExpressionType.ATOMIC

        # 根据深度和复杂度因子调整概率
        if remaining_depth == 2:
            # 倒数第二层，偏向二元表达式
            weights = [0.3, 0.2, 0.5]  # atomic, unary, binary
        else:
            # 其他层级，根据复杂度因子调整
            base_weights = [0.2, 0.3, 0.5]
            weights = [w * (1 + complexity_factor * 0.5) for w in base_weights]

        # 归一化权重
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # 随机选择
        rand_val = random.random()
        cumulative = 0

        for i, (node_type, weight) in enumerate(zip(
                [ExpressionType.ATOMIC, ExpressionType.UNARY, ExpressionType.BINARY],
                weights
        )):
            cumulative += weight
            if rand_val <= cumulative:
                return node_type

        return ExpressionType.BINARY  # 默认返回二元

    def _create_atomic_node(self, available_vars: List[str], depth: int) -> ExpressionNode:
        """创建原子节点"""
        variable = random.choice(available_vars)

        node = ExpressionNode(
            node_id=f"atomic_{depth}_{variable}",
            expression_type=ExpressionType.ATOMIC,
            content=variable,
            depth=depth,
            complexity=1,
            variables={variable}
        )

        return node

    def _create_unary_node(self, target_depth: int, current_depth: int,
                           available_vars: List[str], complexity_factor: float,
                           prefer_balanced: bool) -> ExpressionNode:
        """创建一元节点（通常是否定）"""

        # 创建子节点
        child = self._build_recursive(
            target_depth, current_depth + 1, available_vars,
            complexity_factor, prefer_balanced
        )

        node = ExpressionNode(
            node_id=f"unary_{current_depth}_{random.randint(1000, 9999)}",
            expression_type=ExpressionType.UNARY,
            connector=ConnectorType.NEGATION,
            left_child=child,
            depth=current_depth
        )

        child.parent = node
        return node

    def _create_binary_node(self, target_depth: int, current_depth: int,
                            available_vars: List[str], complexity_factor: float,
                            prefer_balanced: bool) -> ExpressionNode:
        """创建二元节点"""

        # 选择连接符
        connector = self._choose_binary_connector(complexity_factor)

        # 决定子树深度分配
        left_depth, right_depth = self._allocate_child_depths(
            target_depth, current_depth, prefer_balanced
        )

        # 创建左右子树
        left_child = self._build_recursive(
            left_depth, current_depth + 1, available_vars,
            complexity_factor, prefer_balanced
        )

        right_child = self._build_recursive(
            right_depth, current_depth + 1, available_vars,
            complexity_factor, prefer_balanced
        )

        node = ExpressionNode(
            node_id=f"binary_{current_depth}_{random.randint(1000, 9999)}",
            expression_type=ExpressionType.BINARY,
            connector=connector,
            left_child=left_child,
            right_child=right_child,
            depth=current_depth
        )

        left_child.parent = node
        right_child.parent = node

        return node

    def _choose_binary_connector(self, complexity_factor: float) -> ConnectorType:
        """选择二元连接符"""
        # 根据复杂度因子调整概率
        connectors = [
            ConnectorType.CONJUNCTION,
            ConnectorType.DISJUNCTION,
            ConnectorType.IMPLICATION,
            ConnectorType.BICONDITIONAL
        ]

        base_probs = [0.25, 0.25, 0.35, 0.15]

        # 复杂度越高，偏向选择更复杂的连接符
        if complexity_factor > 1.0:
            base_probs = [0.2, 0.2, 0.4, 0.2]  # 提高蕴含和双条件概率

        return random.choices(connectors, weights=base_probs)[0]

    def _allocate_child_depths(self, target_depth: int, current_depth: int,
                               prefer_balanced: bool) -> tuple:
        """分配子树深度"""
        remaining_depth = target_depth - current_depth - 1

        if remaining_depth <= 0:
            return 1, 1

        if prefer_balanced:
            # 平衡分配
            half = remaining_depth // 2
            left_depth = current_depth + 1 + half
            right_depth = current_depth + 1 + (remaining_depth - half)
        else:
            # 随机分配，但保证至少有1层
            split_point = random.randint(1, max(1, remaining_depth))
            left_depth = current_depth + 1 + split_point
            right_depth = current_depth + 1 + (remaining_depth - split_point)

        return left_depth, right_depth

    def _calculate_tree_properties(self, node: ExpressionNode):
        """计算树的属性（后序遍历）"""
        if node.is_leaf():
            # 叶子节点已经设置了基本属性
            return

        # 递归计算子节点
        if node.left_child:
            self._calculate_tree_properties(node.left_child)
        if node.right_child:
            self._calculate_tree_properties(node.right_child)

        # 计算当前节点属性
        node.variables = set()
        child_complexity = 0

        if node.left_child:
            node.variables.update(node.left_child.variables)
            child_complexity += node.left_child.complexity

        if node.right_child:
            node.variables.update(node.right_child.variables)
            child_complexity += node.right_child.complexity

        # 设置复杂度
        node.complexity = child_complexity + self.complexity_weights[node.expression_type]

    def _update_statistics(self, root: ExpressionNode):
        """更新构建统计信息"""
        self.build_statistics['total_trees'] += 1

        # 遍历树收集统计信息
        nodes = self._collect_all_nodes(root)

        total_depth = sum(node.depth for node in nodes)
        total_complexity = sum(node.complexity for node in nodes)

        # 更新平均值
        tree_count = self.build_statistics['total_trees']
        self.build_statistics['avg_depth'] = (
                (self.build_statistics['avg_depth'] * (tree_count - 1) + root.depth) / tree_count
        )
        self.build_statistics['avg_complexity'] = (
                (self.build_statistics['avg_complexity'] * (tree_count - 1) + root.complexity) / tree_count
        )

        # 更新节点类型计数
        for node in nodes:
            self.build_statistics['node_type_counts'][node.expression_type] += 1

    def _collect_all_nodes(self, root: ExpressionNode) -> List[ExpressionNode]:
        """收集树中的所有节点"""
        nodes = [root]

        if root.left_child:
            nodes.extend(self._collect_all_nodes(root.left_child))
        if root.right_child:
            nodes.extend(self._collect_all_nodes(root.right_child))

        return nodes

    def convert_to_logical_formula(self, root: ExpressionNode) -> LogicalFormula:
        """将表达式树转换为LogicalFormula"""
        expression_str = root.get_expression_string()

        # 提取操作符
        operators = []
        self._extract_operators(root, operators)

        return LogicalFormula(
            expression=expression_str,
            variables=root.variables.copy(),
            operators=operators,
            complexity=root.complexity,
            is_compound=(root.expression_type != ExpressionType.ATOMIC)
        )

    def _extract_operators(self, node: ExpressionNode, operators: List[LogicalOperator]):
        """提取表达式树中的操作符"""
        if node.connector:
            # 映射连接符到LogicalOperator
            connector_map = {
                ConnectorType.CONJUNCTION: LogicalOperator.AND,
                ConnectorType.DISJUNCTION: LogicalOperator.OR,
                ConnectorType.IMPLICATION: LogicalOperator.IMPLIES,
                ConnectorType.BICONDITIONAL: LogicalOperator.IFF,
                ConnectorType.NEGATION: LogicalOperator.NOT
            }

            if node.connector in connector_map:
                operators.append(connector_map[node.connector])

        # 递归处理子节点
        if node.left_child:
            self._extract_operators(node.left_child, operators)
        if node.right_child:
            self._extract_operators(node.right_child, operators)

    def get_statistics(self) -> Dict[str, Any]:
        """获取构建统计信息"""
        return self.build_statistics.copy()

    def reset_statistics(self):
        """重置统计信息"""
        self.build_statistics = {
            'total_trees': 0,
            'avg_depth': 0.0,
            'avg_complexity': 0.0,
            'node_type_counts': {t: 0 for t in ExpressionType}
        }
        self.logger.info("统计信息已重置")