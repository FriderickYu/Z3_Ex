# 文件：distractor/generator.py
# 说明：干扰项主控生成器，用于统一分派多个策略生成逻辑干扰项

from typing import List, Dict
from z3 import BoolRef

from distractor.strategies.illogical_reasoning import IllogicalReasoningDistractor
from distractor.strategies.adversarial_structure import AdversarialStructureDistractor
from distractor.strategies.reversed_implication import ReversedImplicationDistractor
from distractor.strategies.unrelated_fact import UnrelatedFactDistractor
from distractor.strategies.logical_complexity import LogicalComplexityDistractor


class DistractorGenerator:
    """
    干扰项主控生成器：负责协调多种策略生成干扰项
    """

    def __init__(self, available_vars: List[BoolRef], enabled_strategies: List[str] = None):
        """
        参数：
            available_vars: 可用于生成表达式的变量池
            enabled_strategies: 要启用的策略名称列表（默认启用全部）
        """
        self.available_vars = available_vars
        self.strategy_map = {
            "illogical_reasoning": IllogicalReasoningDistractor,
            "adversarial_structure": AdversarialStructureDistractor,
            "reversed_implication": ReversedImplicationDistractor,
            "unrelated_fact": UnrelatedFactDistractor,
            "logical_complexity": LogicalComplexityDistractor
        }

        if enabled_strategies is None:
            self.enabled_strategies = list(self.strategy_map.keys())
        else:
            self.enabled_strategies = enabled_strategies

    def generate_all(self, logical_steps: List[Dict], num_per_strategy: int = 2) -> List[Dict]:
        """
        为每个 logical step 生成多个干扰项（每种策略生成 num_per_strategy 个）
        返回所有生成的干扰项列表
        """
        all_distractors = []

        for strategy_name in self.enabled_strategies:
            strategy_cls = self.strategy_map[strategy_name]
            strategy = strategy_cls(self.available_vars)
            distractors = strategy.generate(logical_steps)
            # 限制数量
            all_distractors.extend(distractors[:num_per_strategy])

        return all_distractors

