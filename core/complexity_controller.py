"""
扩展原有ComplexityController，添加表达式树复杂度控制功能
在原有代码基础上无缝集成新功能
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import math
import numpy as np

from rules.base.rule import LogicalFormula, BaseRule

try:
    from ..utils.logger_utils import ARNGLogger
except ValueError:
    from utils.logger_utils import ARNGLogger


class ComplexityMetric(Enum):
    """复杂度度量类型"""
    STRUCTURAL = "structural"  # 结构复杂度
    SEMANTIC = "semantic"  # 语义复杂度
    COMPUTATIONAL = "computational"  # 计算复杂度
    COGNITIVE = "cognitive"  # 认知复杂度
    # 新增表达式树相关复杂度
    TREE_DEPTH = "tree_depth"  # 树深度复杂度
    TREE_BRANCHING = "tree_branching"  # 树分支复杂度


class ComplexityGrowthStrategy(Enum):
    """复杂度增长策略"""
    LINEAR = "linear"  # 线性增长
    EXPONENTIAL = "exponential"  # 指数增长（原默认）
    LOGARITHMIC = "logarithmic"  # 对数增长
    ADAPTIVE = "adaptive"  # 自适应增长
    STAGED = "staged"  # 分阶段增长


@dataclass
class ComplexityProfile:
    """复杂度配置文件"""
    structural_weight: float = 0.25  # 调整权重以适应新维度
    semantic_weight: float = 0.20
    computational_weight: float = 0.20
    cognitive_weight: float = 0.15
    # 新增表达式树权重
    tree_depth_weight: float = 0.10
    tree_branching_weight: float = 0.10

    # 复杂度增长参数
    base_complexity: int = 1
    growth_rate: float = 1.5
    max_complexity: int = 100

    # 层级复杂度映射
    tier_complexity_multiplier: Dict[int, float] = None

    # 新增：表达式树复杂度配置
    expression_type_weights: Dict[str, float] = None
    connector_complexity_weights: Dict[str, float] = None
    depth_penalty_factor: float = 1.2
    branching_bonus_factor: float = 1.1

    def __post_init__(self):
        if self.tier_complexity_multiplier is None:
            self.tier_complexity_multiplier = {
                1: 1.0,  # 基础公理
                2: 1.5,  # 基础推理
                3: 2.0,  # 复合规则
                4: 2.5,  # 量词逻辑
                5: 3.0  # 算术逻辑
            }

        if self.expression_type_weights is None:
            self.expression_type_weights = {
                'atomic': 1.0,
                'unary': 2.0,
                'binary': 3.0,
                'quantified': 5.0
            }

        if self.connector_complexity_weights is None:
            self.connector_complexity_weights = {
                '¬': 1.5,  # 否定
                '∧': 2.0,  # 合取
                '∨': 2.0,  # 析取
                '→': 3.0,  # 蕴含
                '↔': 4.0  # 双条件
            }


class ComplexityController:
    """
    复杂度控制器 - 管理推理复杂度的递增和控制
    支持多维度复杂度评估和动态调整
    [扩展] 现在支持表达式树复杂度控制
    """

    def __init__(self, profile: Optional[ComplexityProfile] = None):
        self.profile = profile or ComplexityProfile()
        self.logger = ARNGLogger("ComplexityController")

        # 原有属性
        self.complexity_history: List[Tuple[int, Dict[str, float]]] = []
        self.current_step = 0
        self.adaptive_mode = True
        self.adjustment_factor = 0.1

        # 新增：表达式树相关属性
        self.expression_complexity_history: List[Tuple[object, Dict[str, float]]] = []
        self.complexity_cache: Dict[str, Dict[str, float]] = {}
        self.current_growth_strategy = ComplexityGrowthStrategy.EXPONENTIAL
        self.performance_feedback: List[float] = []
        self.adaptive_learning_rate = 0.1

        self.logger.info("复杂度控制器初始化完成（支持表达式树）")

    # ==================== 原有方法保持不变 ====================

    def calculate_formula_complexity(self, formula: LogicalFormula) -> Dict[str, float]:
        """
        计算公式的多维度复杂度
        [扩展] 现在支持表达式树复杂度计算
        """
        complexity = {}

        # 原有的4个维度
        complexity['structural'] = self._calculate_structural_complexity(formula)
        complexity['semantic'] = self._calculate_semantic_complexity(formula)
        complexity['computational'] = self._calculate_computational_complexity(formula)
        complexity['cognitive'] = self._calculate_cognitive_complexity(formula)

        # 新增：如果有表达式树信息，计算树复杂度
        if hasattr(formula, 'expression_tree') and formula.expression_tree:
            complexity['tree_depth'] = self._calculate_tree_depth_complexity_from_formula(formula)
            complexity['tree_branching'] = self._calculate_tree_branching_complexity_from_formula(formula)
        else:
            # 基于字符串估算树复杂度
            complexity['tree_depth'] = self._estimate_tree_depth_from_expression(formula.expression)
            complexity['tree_branching'] = self._estimate_tree_branching_from_expression(formula.expression)

        # 综合复杂度（更新权重计算）
        complexity['overall'] = self._calculate_overall_complexity(complexity)

        self.logger.debug(f"公式复杂度: {complexity}")
        return complexity

    def _calculate_structural_complexity(self, formula: LogicalFormula) -> float:
        """计算结构复杂度"""
        var_complexity = len(formula.variables) * 2
        operator_complexity = len(formula.operators) * 1.5
        nesting_complexity = self._estimate_nesting_depth(formula.expression) * 3
        return var_complexity + operator_complexity + nesting_complexity

    def _calculate_semantic_complexity(self, formula: LogicalFormula) -> float:
        """计算语义复杂度"""
        base_score = 5.0
        if any(op.value in ['∀', '∃'] for op in formula.operators):
            base_score += 10
        if any(op.value in ['→', '↔'] for op in formula.operators):
            base_score += 5
        if formula.is_compound:
            base_score += 8
        return base_score

    def _calculate_computational_complexity(self, formula: LogicalFormula) -> float:
        """计算计算复杂度"""
        base_steps = 1
        base_steps += len(formula.variables)
        base_steps += len(formula.operators) * 2
        if formula.is_compound:
            base_steps = int(base_steps * 1.5)
        return float(base_steps)

    def _calculate_cognitive_complexity(self, formula: LogicalFormula) -> float:
        """计算认知复杂度"""
        base_difficulty = 3.0
        base_difficulty += len(formula.variables) * 1.5
        difficult_operators = ['→', '↔', '∀', '∃']
        for op in formula.operators:
            if op.value in difficult_operators:
                base_difficulty += 4
        expression_length_factor = len(formula.expression) / 20
        base_difficulty += expression_length_factor
        return base_difficulty

    def _calculate_overall_complexity(self, complexity_dict: Dict[str, float]) -> float:
        """计算综合复杂度（更新以支持新维度）"""
        weights = {
            'structural': self.profile.structural_weight,
            'semantic': self.profile.semantic_weight,
            'computational': self.profile.computational_weight,
            'cognitive': self.profile.cognitive_weight,
            'tree_depth': self.profile.tree_depth_weight,
            'tree_branching': self.profile.tree_branching_weight
        }

        overall = 0.0
        for dimension, value in complexity_dict.items():
            if dimension in weights and dimension != 'overall':
                overall += value * weights[dimension]

        return overall

    def _estimate_nesting_depth(self, expression: str) -> int:
        """估算表达式的嵌套深度"""
        max_depth = 0
        current_depth = 0
        for char in expression:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth = max(0, current_depth - 1)
        return max_depth

    # ==================== 新增：表达式树复杂度方法 ====================

    def calculate_expression_tree_complexity(self, expression_tree) -> Dict[str, float]:
        """
        计算表达式树的多维度复杂度

        Args:
            expression_tree: 表达式树根节点（ExpressionNode类型）

        Returns:
            Dict[str, float]: 各维度复杂度分数
        """
        # 检查缓存
        tree_signature = self._get_tree_signature(expression_tree)
        if tree_signature in self.complexity_cache:
            return self.complexity_cache[tree_signature].copy()

        complexity = {}

        # 计算各维度复杂度
        complexity['structural'] = self._calculate_structural_complexity_tree(expression_tree)
        complexity['semantic'] = self._calculate_semantic_complexity_tree(expression_tree)
        complexity['computational'] = self._calculate_computational_complexity_tree(expression_tree)
        complexity['cognitive'] = self._calculate_cognitive_complexity_tree(expression_tree)
        complexity['tree_depth'] = self._calculate_tree_depth_complexity(expression_tree)
        complexity['tree_branching'] = self._calculate_tree_branching_complexity(expression_tree)

        # 综合复杂度
        complexity['overall'] = self._calculate_overall_complexity(complexity)

        # 缓存结果
        self.complexity_cache[tree_signature] = complexity.copy()

        # 记录到历史
        self.expression_complexity_history.append((expression_tree, complexity.copy()))

        self.logger.debug(f"表达式树复杂度: {complexity}")
        return complexity

    def _calculate_structural_complexity_tree(self, tree) -> float:
        """计算表达式树的结构复杂度"""
        if tree.is_leaf():
            return self.profile.expression_type_weights.get('atomic', 1.0)

        # 递归计算子树复杂度
        left_complexity = 0
        right_complexity = 0

        if tree.left_child:
            left_complexity = self._calculate_structural_complexity_tree(tree.left_child)
        if tree.right_child:
            right_complexity = self._calculate_structural_complexity_tree(tree.right_child)

        # 当前节点复杂度
        node_type = tree.expression_type.value if hasattr(tree.expression_type, 'value') else str(tree.expression_type)
        node_complexity = self.profile.expression_type_weights.get(node_type, 3.0)

        # 连接符复杂度
        connector_bonus = 0
        if tree.connector:
            connector_str = tree.connector.value if hasattr(tree.connector, 'value') else str(tree.connector)
            connector_bonus = self.profile.connector_complexity_weights.get(connector_str, 1.0)

        # 深度惩罚
        depth_factor = self.profile.depth_penalty_factor ** tree.depth

        total_complexity = (node_complexity + connector_bonus +
                            left_complexity + right_complexity) * depth_factor

        return total_complexity

    def _calculate_semantic_complexity_tree(self, tree) -> float:
        """计算表达式树的语义复杂度"""
        if tree.is_leaf():
            return 2.0

        base_complexity = 5.0

        # 连接符语义复杂度
        if tree.connector:
            semantic_weights = {'¬': 1.0, '∧': 1.5, '∨': 1.5, '→': 2.5, '↔': 3.0}
            connector_str = tree.connector.value if hasattr(tree.connector, 'value') else str(tree.connector)
            base_complexity += semantic_weights.get(connector_str, 1.0)

        # 嵌套语义复杂度
        nesting_bonus = tree.depth * 1.5

        # 变量多样性影响
        variable_count = len(tree.variables) if hasattr(tree, 'variables') else 1
        diversity_factor = 1.0 + (variable_count - 1) * 0.5

        # 子树语义复杂度
        child_complexity = 0
        if tree.left_child:
            child_complexity += self._calculate_semantic_complexity_tree(tree.left_child)
        if tree.right_child:
            child_complexity += self._calculate_semantic_complexity_tree(tree.right_child)

        return (base_complexity + nesting_bonus + child_complexity) * diversity_factor

    def _calculate_computational_complexity_tree(self, tree) -> float:
        """计算表达式树的计算复杂度"""
        if tree.is_leaf():
            return 1.0

        # 基于节点数量的复杂度估算
        node_count = self._count_tree_nodes(tree)

        # 连接符计算开销
        computational_costs = {'¬': 1.0, '∧': 1.5, '∨': 1.5, '→': 2.0, '↔': 2.5}
        operator_cost = 0
        self._accumulate_operator_costs(tree, computational_costs, operator_cost)

        # 变量数量影响计算复杂度
        variable_count = len(tree.variables) if hasattr(tree, 'variables') else 1
        variable_factor = variable_count ** 1.2

        # 深度影响计算步骤
        depth_factor = tree.depth * 1.3

        return node_count + operator_cost + variable_factor + depth_factor

    def _calculate_cognitive_complexity_tree(self, tree) -> float:
        """计算表达式树的认知复杂度"""
        if tree.is_leaf():
            return 1.5

        base_cognitive_load = 4.0

        # 嵌套层次对认知的影响（非线性增长）
        nesting_penalty = (tree.depth ** 1.5) * 0.8

        # 连接符认知难度
        cognitive_difficulty = {'¬': 1.0, '∧': 1.2, '∨': 1.2, '→': 2.0, '↔': 2.5}
        connector_str = tree.connector.value if hasattr(tree.connector, 'value') else str(tree.connector)
        connector_difficulty = cognitive_difficulty.get(connector_str, 1.0)

        # 变量跟踪负担
        variable_count = len(tree.variables) if hasattr(tree, 'variables') else 1
        variable_tracking_load = variable_count * 0.7

        # 子树认知复杂度
        child_cognitive_load = 0
        if tree.left_child:
            child_cognitive_load += self._calculate_cognitive_complexity_tree(tree.left_child) * 0.6
        if tree.right_child:
            child_cognitive_load += self._calculate_cognitive_complexity_tree(tree.right_child) * 0.6

        total_cognitive_complexity = (base_cognitive_load + nesting_penalty +
                                      variable_tracking_load + child_cognitive_load) * connector_difficulty

        return total_cognitive_complexity

    def _calculate_tree_depth_complexity(self, tree) -> float:
        """计算树深度复杂度"""
        max_depth = self._get_max_depth(tree)
        depth_complexity = max_depth ** 1.3

        # 深度不平衡惩罚
        depth_variance = self._calculate_depth_variance(tree)
        balance_penalty = depth_variance * 0.5

        return depth_complexity + balance_penalty

    def _calculate_tree_branching_complexity(self, tree) -> float:
        """计算树分支复杂度"""
        if tree.is_leaf():
            return 0.0

        branching_factor = 0
        if tree.left_child:
            branching_factor += 1
        if tree.right_child:
            branching_factor += 1

        # 递归计算子树分支复杂度
        child_branching = 0
        if tree.left_child:
            child_branching += self._calculate_tree_branching_complexity(tree.left_child)
        if tree.right_child:
            child_branching += self._calculate_tree_branching_complexity(tree.right_child)

        # 分支密度奖励
        density_bonus = branching_factor * self.profile.branching_bonus_factor

        return density_bonus + child_branching

    # ==================== 新增：复杂度驱动策略 ====================

    def generate_complexity_target(self,
                                   step: int,
                                   strategy: ComplexityGrowthStrategy = None,
                                   custom_parameters: Dict = None) -> Dict[str, float]:
        """
        生成复杂度驱动的目标

        Args:
            step: 当前步骤
            strategy: 增长策略
            custom_parameters: 自定义参数

        Returns:
            Dict[str, float]: 各维度复杂度目标
        """
        strategy = strategy or self.current_growth_strategy
        base_complexity = self.profile.base_complexity

        # 计算整体目标复杂度
        if strategy == ComplexityGrowthStrategy.LINEAR:
            target_overall = base_complexity + step * 2.0
        elif strategy == ComplexityGrowthStrategy.EXPONENTIAL:
            target_overall = base_complexity * (self.profile.growth_rate ** step)
        elif strategy == ComplexityGrowthStrategy.LOGARITHMIC:
            target_overall = base_complexity + math.log(step + 1) * 5.0
        elif strategy == ComplexityGrowthStrategy.STAGED:
            stage = step // 3
            target_overall = base_complexity + stage * 4.0 + (step % 3) * 1.0
        else:  # ADAPTIVE
            target_overall = self._adaptive_complexity_target(step, base_complexity)

        # 应用最大值限制
        target_overall = min(target_overall, self.profile.max_complexity)

        # 生成各维度目标
        targets = self._generate_dimension_targets(target_overall, step)

        return targets

    def suggest_expression_parameters(self, target_complexity: float) -> Dict[str, any]:
        """
        根据复杂度目标建议表达式生成参数

        Args:
            target_complexity: 目标复杂度

        Returns:
            Dict[str, any]: 建议的生成参数
        """
        suggestions = {}

        # 基于目标复杂度建议深度
        if target_complexity < 10:
            suggestions['target_depth'] = 2
            suggestions['complexity_factor'] = 0.8
        elif target_complexity < 25:
            suggestions['target_depth'] = 3
            suggestions['complexity_factor'] = 1.0
        elif target_complexity < 50:
            suggestions['target_depth'] = 4
            suggestions['complexity_factor'] = 1.3
        else:
            suggestions['target_depth'] = 5
            suggestions['complexity_factor'] = 1.5

        # 其他参数建议
        suggestions['variable_limit'] = min(8, max(3, int(target_complexity / 8)))
        suggestions['prefer_balanced'] = target_complexity < 30

        return suggestions

    # ==================== 原有方法保持不变 ====================

    def calculate_rule_complexity(self, rule: BaseRule) -> float:
        """计算规则的复杂度"""
        base_complexity = rule.get_complexity_score()
        tier_multiplier = self.profile.tier_complexity_multiplier.get(rule.tier, 1.0)
        usage_factor = 1.0
        if rule.usage_count > 0:
            usage_factor = 2.0 - rule.success_rate
        complexity = base_complexity * tier_multiplier * usage_factor
        return min(complexity, self.profile.max_complexity)

    def get_target_complexity_for_step(self, step: int,
                                       initial_complexity: float = None) -> float:
        """获取指定步骤的目标复杂度"""
        if initial_complexity is None:
            initial_complexity = self.profile.base_complexity
        target = initial_complexity * (self.profile.growth_rate ** step)
        target = min(target, self.profile.max_complexity)
        if self.adaptive_mode and self.complexity_history:
            target = self._apply_adaptive_adjustment(target, step)
        return target

    def _apply_adaptive_adjustment(self, target: float, step: int) -> float:
        """应用自适应调整"""
        if len(self.complexity_history) < 2:
            return target
        recent_complexity = [entry[1]['overall'] for entry in self.complexity_history[-3:]]
        if len(recent_complexity) >= 2:
            change_rate = (recent_complexity[-1] - recent_complexity[0]) / len(recent_complexity)
            if change_rate > target * 0.2:
                adjustment = -target * self.adjustment_factor
                target += adjustment
                self.logger.debug(f"复杂度增长过快，调整: {adjustment}")
            elif change_rate < target * 0.05:
                adjustment = target * self.adjustment_factor
                target += adjustment
                self.logger.debug(f"复杂度增长过慢，调整: {adjustment}")
        return max(self.profile.base_complexity, target)

    def record_complexity(self, step: int, complexity: Dict[str, float]):
        """记录复杂度历史"""
        self.complexity_history.append((step, complexity.copy()))
        self.current_step = step
        if len(self.complexity_history) > 100:
            self.complexity_history = self.complexity_history[-50:]

    def should_increase_tier(self, current_complexity: float,
                             target_complexity: float, current_tier: int) -> bool:
        """判断是否应该提升规则层级"""
        complexity_ratio = current_complexity / target_complexity if target_complexity > 0 else 0
        if complexity_ratio < 0.7 and current_tier < 5:
            return True
        tier_threshold = current_tier * 3
        if self.current_step >= tier_threshold:
            return True
        return False

    def get_complexity_statistics(self) -> Dict[str, any]:
        """获取复杂度统计信息"""
        if not self.complexity_history:
            return {"message": "暂无复杂度历史记录"}

        # 提取各维度复杂度数据
        structural_scores = [entry[1]['structural'] for entry in self.complexity_history]
        semantic_scores = [entry[1]['semantic'] for entry in self.complexity_history]
        computational_scores = [entry[1]['computational'] for entry in self.complexity_history]
        cognitive_scores = [entry[1]['cognitive'] for entry in self.complexity_history]
        overall_scores = [entry[1]['overall'] for entry in self.complexity_history]

        stats = {
            "total_steps": len(self.complexity_history),
            "structural": {
                "mean": sum(structural_scores) / len(structural_scores),
                "min": min(structural_scores),
                "max": max(structural_scores)
            },
            "semantic": {
                "mean": sum(semantic_scores) / len(semantic_scores),
                "min": min(semantic_scores),
                "max": max(semantic_scores)
            },
            "computational": {
                "mean": sum(computational_scores) / len(computational_scores),
                "min": min(computational_scores),
                "max": max(computational_scores)
            },
            "cognitive": {
                "mean": sum(cognitive_scores) / len(cognitive_scores),
                "min": min(cognitive_scores),
                "max": max(cognitive_scores)
            },
            "overall": {
                "mean": sum(overall_scores) / len(overall_scores),
                "min": min(overall_scores),
                "max": max(overall_scores),
                "growth_rate": self._calculate_growth_rate(overall_scores)
            }
        }

        # 新增：表达式树统计
        if self.expression_complexity_history:
            stats["expression_trees_analyzed"] = len(self.expression_complexity_history)
            stats["cache_efficiency"] = len(self.complexity_cache) / max(1, len(self.expression_complexity_history))

        return stats

    def _calculate_growth_rate(self, scores: List[float]) -> float:
        """计算复杂度增长率"""
        if len(scores) < 2:
            return 0.0
        n = len(scores)
        x_sum = sum(range(n))
        y_sum = sum(scores)
        xy_sum = sum(i * scores[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        if n * x2_sum - x_sum * x_sum == 0:
            return 0.0
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        return slope

    def reset(self):
        """重置复杂度控制器"""
        self.complexity_history.clear()
        self.expression_complexity_history.clear()
        self.complexity_cache.clear()
        self.performance_feedback.clear()
        self.current_step = 0
        self.logger.info("复杂度控制器已重置")

    def set_adaptive_mode(self, enabled: bool):
        """设置自适应模式"""
        self.adaptive_mode = enabled
        self.logger.info(f"自适应模式{'开启' if enabled else '关闭'}")

    def update_profile(self, **kwargs):
        """更新复杂度配置"""
        for key, value in kwargs.items():
            if hasattr(self.profile, key):
                setattr(self.profile, key, value)
                self.logger.info(f"更新配置: {key} = {value}")
            else:
                self.logger.warning(f"未知配置项: {key}")

    def analyze_complexity_trend(self) -> Dict[str, any]:
        """分析复杂度趋势"""
        if len(self.complexity_history) < 3:
            return {"message": "数据不足，无法分析趋势"}

        overall_scores = [entry[1]['overall'] for entry in self.complexity_history]
        recent_scores = overall_scores[-5:]
        early_scores = overall_scores[:5]
        recent_avg = sum(recent_scores) / len(recent_scores)
        early_avg = sum(early_scores) / len(early_scores)

        trend_analysis = {
            "trend_direction": "上升" if recent_avg > early_avg else "下降",
            "trend_strength": abs(recent_avg - early_avg) / early_avg if early_avg > 0 else 0,
            "recent_average": recent_avg,
            "early_average": early_avg,
            "total_change": overall_scores[-1] - overall_scores[0],
            "volatility": self._calculate_volatility(overall_scores),
            "recommendation": self._get_trend_recommendation(recent_avg, early_avg)
        }

        return trend_analysis

    def _calculate_volatility(self, scores: List[float]) -> float:
        """计算复杂度波动性"""
        if len(scores) < 2:
            return 0.0
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        return math.sqrt(variance)

    def _get_trend_recommendation(self, recent_avg: float, early_avg: float) -> str:
        """获取趋势建议"""
        change_ratio = (recent_avg - early_avg) / early_avg if early_avg > 0 else 0
        if change_ratio > 0.5:
            return "复杂度增长过快，建议降低增长率"
        elif change_ratio < 0.1:
            return "复杂度增长过慢，建议提高增长率"
        else:
            return "复杂度增长适中，保持当前设置"

    # ==================== 新增：辅助方法 ====================

    def _adaptive_complexity_target(self, step: int, base_complexity: float) -> float:
        """自适应复杂度目标生成"""
        if len(self.performance_feedback) < 2:
            # 初始阶段使用线性增长
            return base_complexity + step * 1.5

        # 基于历史表现调整增长率
        recent_performance = np.mean(self.performance_feedback[-3:])

        if recent_performance > 0.8:  # 表现良好，加速增长
            growth_rate = 2.0 + self.adaptive_learning_rate
        elif recent_performance < 0.4:  # 表现不佳，减缓增长
            growth_rate = 1.0 - self.adaptive_learning_rate
        else:  # 中等表现，正常增长
            growth_rate = 1.5

        return base_complexity + step * growth_rate

    def _generate_dimension_targets(self, overall_target: float, step: int) -> Dict[str, float]:
        """生成各维度的目标复杂度"""
        targets = {}

        # 基于当前步骤调整各维度权重
        if step < 3:
            # 早期阶段，重点关注结构复杂度
            weights = {
                'structural': 0.4,
                'semantic': 0.2,
                'computational': 0.15,
                'cognitive': 0.1,
                'tree_depth': 0.1,
                'tree_branching': 0.05
            }
        elif step < 6:
            # 中期阶段，平衡各维度
            weights = {
                'structural': 0.25,
                'semantic': 0.25,
                'computational': 0.2,
                'cognitive': 0.15,
                'tree_depth': 0.1,
                'tree_branching': 0.05
            }
        else:
            # 后期阶段，重点关注高级复杂度
            weights = {
                'structural': 0.2,
                'semantic': 0.3,
                'computational': 0.25,
                'cognitive': 0.15,
                'tree_depth': 0.05,
                'tree_branching': 0.05
            }

        for dimension, weight in weights.items():
            targets[dimension] = overall_target * weight

        targets['overall'] = overall_target
        return targets

    def record_performance_feedback(self, performance_score: float):
        """记录性能反馈用于自适应调整"""
        self.performance_feedback.append(performance_score)

        # 限制反馈历史长度
        if len(self.performance_feedback) > 20:
            self.performance_feedback = self.performance_feedback[-10:]

    def set_growth_strategy(self, strategy: ComplexityGrowthStrategy):
        """设置复杂度增长策略"""
        self.current_growth_strategy = strategy
        self.logger.info(f"复杂度增长策略设置为: {strategy.value}")

    # ==================== 辅助方法：表达式树相关 ====================

    def _get_tree_signature(self, tree) -> str:
        """获取表达式树的签名用于缓存"""
        try:
            return tree.get_expression_string()
        except:
            return str(hash(str(tree)))

    def _count_tree_nodes(self, tree) -> int:
        """计算树中节点总数"""
        count = 1
        if hasattr(tree, 'left_child') and tree.left_child:
            count += self._count_tree_nodes(tree.left_child)
        if hasattr(tree, 'right_child') and tree.right_child:
            count += self._count_tree_nodes(tree.right_child)
        return count

    def _accumulate_operator_costs(self, tree, cost_dict: Dict[str, float], total_cost: float):
        """累计操作符计算成本"""
        if hasattr(tree, 'connector') and tree.connector:
            connector_str = tree.connector.value if hasattr(tree.connector, 'value') else str(tree.connector)
            if connector_str in cost_dict:
                total_cost += cost_dict[connector_str]

        if hasattr(tree, 'left_child') and tree.left_child:
            self._accumulate_operator_costs(tree.left_child, cost_dict, total_cost)
        if hasattr(tree, 'right_child') and tree.right_child:
            self._accumulate_operator_costs(tree.right_child, cost_dict, total_cost)

    def _get_max_depth(self, tree) -> int:
        """获取树的最大深度"""
        if tree.is_leaf():
            return getattr(tree, 'depth', 0)

        max_depth = getattr(tree, 'depth', 0)
        if hasattr(tree, 'left_child') and tree.left_child:
            max_depth = max(max_depth, self._get_max_depth(tree.left_child))
        if hasattr(tree, 'right_child') and tree.right_child:
            max_depth = max(max_depth, self._get_max_depth(tree.right_child))

        return max_depth

    def _calculate_depth_variance(self, tree) -> float:
        """计算深度方差（衡量树的平衡性）"""
        depths = []
        self._collect_leaf_depths(tree, depths)

        if len(depths) <= 1:
            return 0.0

        mean_depth = sum(depths) / len(depths)
        variance = sum((d - mean_depth) ** 2 for d in depths) / len(depths)
        return variance

    def _collect_leaf_depths(self, tree, depths: List[int]):
        """收集所有叶子节点的深度"""
        if tree.is_leaf():
            depths.append(getattr(tree, 'depth', 0))
        else:
            if hasattr(tree, 'left_child') and tree.left_child:
                self._collect_leaf_depths(tree.left_child, depths)
            if hasattr(tree, 'right_child') and tree.right_child:
                self._collect_leaf_depths(tree.right_child, depths)

    # ==================== 辅助方法：基于字符串的树复杂度估算 ====================

    def _calculate_tree_depth_complexity_from_formula(self, formula: LogicalFormula) -> float:
        """从LogicalFormula计算树深度复杂度"""
        if hasattr(formula, 'expression_tree') and formula.expression_tree:
            return self._calculate_tree_depth_complexity(formula.expression_tree)
        else:
            # 基于表达式字符串估算
            return self._estimate_tree_depth_from_expression(formula.expression)

    def _calculate_tree_branching_complexity_from_formula(self, formula: LogicalFormula) -> float:
        """从LogicalFormula计算树分支复杂度"""
        if hasattr(formula, 'expression_tree') and formula.expression_tree:
            return self._calculate_tree_branching_complexity(formula.expression_tree)
        else:
            # 基于表达式字符串估算
            return self._estimate_tree_branching_from_expression(formula.expression)

    def _estimate_tree_depth_from_expression(self, expression: str) -> float:
        """基于表达式字符串估算树深度复杂度"""
        # 简化估算：基于括号嵌套深度
        max_depth = self._estimate_nesting_depth(expression)
        return max_depth * 2.0  # 转换为复杂度分数

    def _estimate_tree_branching_from_expression(self, expression: str) -> float:
        """基于表达式字符串估算树分支复杂度"""
        # 简化估算：基于逻辑操作符数量
        operators = ['∧', '∨', '→', '↔', '¬']
        operator_count = sum(expression.count(op) for op in operators)
        return operator_count * 1.5  # 转换为复杂度分数

    # ==================== 高级功能：复杂度优化建议 ====================

    def analyze_complexity_bottlenecks(self) -> Dict[str, any]:
        """分析复杂度瓶颈"""
        if len(self.expression_complexity_history) < 5:
            return {"message": "数据不足，无法分析瓶颈"}

        # 分析各维度的增长趋势
        dimensions = ['structural', 'semantic', 'computational', 'cognitive', 'tree_depth', 'tree_branching']
        bottlenecks = {}

        for dimension in dimensions:
            values = []
            for _, complexity in self.expression_complexity_history[-10:]:
                if dimension in complexity:
                    values.append(complexity[dimension])

            if len(values) >= 3:
                # 计算增长率
                growth_rate = (values[-1] - values[0]) / len(values) if values[0] > 0 else 0
                # 计算方差（稳定性）
                mean_val = sum(values) / len(values)
                variance = sum((v - mean_val) ** 2 for v in values) / len(values)

                bottlenecks[dimension] = {
                    'growth_rate': growth_rate,
                    'stability': 1.0 / (1.0 + variance),  # 方差越小，稳定性越高
                    'current_value': values[-1],
                    'trend': 'increasing' if growth_rate > 0.1 else 'stable' if growth_rate > -0.1 else 'decreasing'
                }

        # 识别问题维度
        problematic_dimensions = []
        for dim, stats in bottlenecks.items():
            if stats['growth_rate'] < 0.05 and stats['current_value'] < 5.0:  # 增长缓慢且值较低
                problematic_dimensions.append(dim)

        return {
            'bottlenecks': bottlenecks,
            'problematic_dimensions': problematic_dimensions,
            'recommendations': self._generate_optimization_recommendations(problematic_dimensions)
        }

    def _generate_optimization_recommendations(self, problematic_dimensions: List[str]) -> List[str]:
        """生成优化建议"""
        recommendations = []

        if 'structural' in problematic_dimensions:
            recommendations.append("增加表达式树的深度或节点数量")

        if 'semantic' in problematic_dimensions:
            recommendations.append("使用更复杂的逻辑连接符（如蕴含、双条件）")

        if 'tree_depth' in problematic_dimensions:
            recommendations.append("增加表达式树的目标深度")

        if 'tree_branching' in problematic_dimensions:
            recommendations.append("创建更多的分支节点，减少线性结构")

        if 'cognitive' in problematic_dimensions:
            recommendations.append("增加变量数量或使用更复杂的嵌套结构")

        if 'computational' in problematic_dimensions:
            recommendations.append("增加操作符数量或使用计算密集的逻辑结构")

        if not recommendations:
            recommendations.append("当前复杂度分布良好，保持现有策略")

        return recommendations

    def get_optimization_suggestions(self, target_complexity: float) -> Dict[str, any]:
        """获取达到目标复杂度的优化建议"""
        current_avg = 0.0
        if self.expression_complexity_history:
            recent_complexities = [comp['overall'] for _, comp in self.expression_complexity_history[-5:]]
            current_avg = sum(recent_complexities) / len(recent_complexities)

        gap = target_complexity - current_avg
        suggestions = {
            'current_average': current_avg,
            'target': target_complexity,
            'gap': gap,
            'suggestions': []
        }

        if gap > 0:  # 需要增加复杂度
            if gap < 5:
                suggestions['suggestions'].append("轻微增加：提高complexity_factor到1.1-1.2")
            elif gap < 15:
                suggestions['suggestions'].append("适度增加：增加目标深度1层，或提高complexity_factor到1.3-1.5")
            else:
                suggestions['suggestions'].append("大幅增加：增加目标深度2层以上，使用更复杂的连接符")

        elif gap < 0:  # 需要降低复杂度
            if gap > -5:
                suggestions['suggestions'].append("轻微降低：减少complexity_factor到0.8-0.9")
            elif gap > -15:
                suggestions['suggestions'].append("适度降低：减少目标深度1层，或降低complexity_factor到0.6-0.8")
            else:
                suggestions['suggestions'].append("大幅降低：显著减少深度和复杂度因子")
        else:
            suggestions['suggestions'].append("当前复杂度接近目标，保持现有参数")

        return suggestions