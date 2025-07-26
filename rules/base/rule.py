from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

from utils.logger_utils import ARNGLogger


class RuleType(Enum):
    """规则类型枚举"""
    AXIOM = "axiom"  # 基础公理
    BASIC = "basic"  # 基础推理
    COMPOUND = "compound"  # 复合规则
    QUANTIFIER = "quantifier"  # 量词逻辑
    ARITHMETIC = "arithmetic"  # 算术逻辑


class LogicalOperator(Enum):
    """逻辑操作符枚举"""
    AND = "∧"
    OR = "∨"
    NOT = "¬"
    IMPLIES = "→"
    IFF = "↔"
    FORALL = "∀"
    EXISTS = "∃"


@dataclass
class LogicalFormula:
    """逻辑公式数据结构"""
    expression: str
    variables: Set[str]
    operators: List[LogicalOperator]
    complexity: int
    is_compound: bool = False

    def __post_init__(self):
        """后处理：计算复杂度"""
        if self.complexity == 0:
            self.complexity = len(self.variables) + len(self.operators)


@dataclass
class RuleInstance:
    """规则实例"""
    premises: List[LogicalFormula]
    conclusion: LogicalFormula
    rule_name: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseRule(ABC):
    """
    规则基类 - 定义所有推理规则的统一接口
    支持双层DAG架构中的规则层
    """

    def __init__(self, rule_id: str, rule_type: RuleType, tier: int):
        self.rule_id = rule_id
        self.rule_type = rule_type
        self.tier = tier

        self.logger = ARNGLogger(f"Rule_{rule_id}")

        # 规则统计信息
        self.usage_count = 0
        self.success_rate = 0.0
        self.last_used = None

        self.logger.info(f"规则初始化: {rule_id}, 类型: {rule_type.value}, 层级: {tier}")

    @abstractmethod
    def can_apply(self, premises: List[LogicalFormula]) -> bool:
        """
        检查规则是否可以应用于给定前提

        Args:
            premises: 前提公式列表

        Returns:
            bool: 是否可以应用
        """
        pass

    @abstractmethod
    def apply(self, premises: List[LogicalFormula]) -> List[LogicalFormula]:
        """
        应用规则生成结论

        Args:
            premises: 前提公式列表

        Returns:
            List[LogicalFormula]: 生成的结论列表
        """
        pass

    @abstractmethod
    def get_template(self) -> Dict[str, Any]:
        """
        获取规则模板（用于随机生成）

        Returns:
            Dict: 规则模板信息
        """
        pass

    def validate_application(self, premises: List[LogicalFormula],
                             conclusions: List[LogicalFormula]) -> bool:
        """
        验证规则应用的正确性

        Args:
            premises: 前提公式
            conclusions: 结论公式

        Returns:
            bool: 应用是否正确
        """
        try:
            expected = self.apply(premises)
            return self._formulas_equivalent(expected, conclusions)
        except Exception as e:
            self.logger.error(f"规则验证失败: {e}")
            return False

    def _formulas_equivalent(self, formulas1: List[LogicalFormula],
                             formulas2: List[LogicalFormula]) -> bool:
        """检查两个公式列表是否等价"""
        if len(formulas1) != len(formulas2):
            return False

        # 简单的字符串比较（可以扩展为语义等价检查）
        expr1 = sorted([f.expression for f in formulas1])
        expr2 = sorted([f.expression for f in formulas2])
        return expr1 == expr2

    def get_complexity_score(self) -> int:
        """获取规则复杂度评分"""
        return self.tier * 10  # 基础评分，子类可以重写

    def update_statistics(self, success: bool):
        """更新规则使用统计"""
        self.usage_count += 1
        if success:
            self.success_rate = (self.success_rate * (self.usage_count - 1) + 1) / self.usage_count
        else:
            self.success_rate = (self.success_rate * (self.usage_count - 1)) / self.usage_count

        from datetime import datetime
        self.last_used = datetime.now()

        self.logger.debug(f"规则统计更新: 使用次数={self.usage_count}, 成功率={self.success_rate:.2f}")

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.rule_id}, tier={self.tier})"