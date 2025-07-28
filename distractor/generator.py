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
            try:
                strategy_cls = self.strategy_map[strategy_name]

                # 根据策略类型创建实例，处理不同的初始化参数
                if strategy_name in ["unrelated_fact", "logical_complexity"]:
                    # 这些策略需要 available_vars 参数
                    strategy = strategy_cls(self.available_vars)
                elif strategy_name == "illogical_reasoning":
                    # 这个策略有可选的 min_drop, max_drop 参数，确保传入正确的整数
                    strategy = strategy_cls(min_drop=1, max_drop=2)
                else:
                    # 其他策略使用默认构造函数
                    strategy = strategy_cls()

                distractors = strategy.generate(logical_steps)

                # 限制数量
                all_distractors.extend(distractors[:num_per_strategy])

            except Exception as e:
                print(f"⚠️  策略 {strategy_name} 执行失败: {e}")
                continue

        return all_distractors

    def generate_by_strategy(self, logical_steps: List[Dict], strategy_name: str, count: int = 1) -> List[Dict]:
        """
        使用指定策略生成干扰项
        """
        if strategy_name not in self.strategy_map:
            raise ValueError(f"未知策略: {strategy_name}")

        try:
            strategy_cls = self.strategy_map[strategy_name]

            # 根据策略类型创建实例
            if strategy_name in ["unrelated_fact", "logical_complexity"]:
                strategy = strategy_cls(self.available_vars)
            elif strategy_name == "illogical_reasoning":
                strategy = strategy_cls(min_drop=1, max_drop=2)
            else:
                strategy = strategy_cls()

            distractors = strategy.generate(logical_steps)
            return distractors[:count]

        except Exception as e:
            print(f"⚠️  策略 {strategy_name} 执行失败: {e}")
            return []