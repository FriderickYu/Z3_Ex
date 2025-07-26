import random
import logging
from typing import List, Type, Set, Dict

from rules.conjunction_elimination import ConjunctionElimination
from rules.conjunction_introduction import ConjunctionIntroduction
from rules.disjunction_introduction import DisjunctionIntroduction
from rules.rule import Rule
from rules.modus_ponens import ModusPonens
from utils.logger_utils import setup_logger
from utils.vocabulary_utils import generate_variable_names

logger = setup_logger("rules_pooling")

class RulesPooling:
    """
    规则池与变量池管理类。
    支持规则采样、语义变量命名与复用逻辑。
    """

    def __init__(self):
        self.available_rules = [
            ConjunctionIntroduction,
            ConjunctionElimination,
            DisjunctionIntroduction,
            ModusPonens,
        ]
        self.all_used_vars: Set[str] = set()
        self.generated_vars: List[str] = []

        self.var_usage_count: Dict[str, int] = {}
        # 获取 100 个语义名
        self.semantic_names: List[str] = generate_variable_names(100)
        self.semantic_index = 0

        # 用于记录 Var0 → CompletedByDev的映射
        self.var_name_map: Dict[str, str] = {}

    def _gen_var_name(self) -> str:
        """
        生成唯一的语义变量名。
        """
        while self.semantic_index < len(self.semantic_names):
            semantic_name = self.semantic_names[self.semantic_index]
            self.semantic_index += 1
            if semantic_name not in self.all_used_vars:
                self.all_used_vars.add(semantic_name)
                self.generated_vars.append(semantic_name)
                self.var_usage_count[semantic_name] = 0
                return semantic_name

        raise RuntimeError("语义变量名不足，请扩展 generate_variable_names 返回数量")

    def sample_rule(self) -> Rule:
        return random.choice(self.available_rules)

    def sample_vars(self, num: int, exclude: Set[str] = set(), reuse_ratio: float = 0.5) -> List[str]:
        candidates = [v for v in self.generated_vars if v not in exclude]
        reuse_count = min(int(num * reuse_ratio), len(candidates))
        new_count = num - reuse_count

        reused = random.sample(candidates, reuse_count) if reuse_count > 0 else []
        new_vars = [self._gen_var_name() for _ in range(new_count)]

        for v in reused:
            self.var_usage_count[v] += 1
        return reused + new_vars

    def reset(self):
        self.all_used_vars.clear()
        self.generated_vars.clear()
        self.var_usage_count.clear()
        self.semantic_index = 0
        self.var_name_map.clear()

Rule.SHORT_LABELS = {
    "ConjunctionIntroduction": "∧Intro",
    "ConjunctionElimination": "∧Elim",
    "DisjunctionIntroduction": "∨Intro",
    "ModusPonens": "→",
}

Rule.get_short_label = lambda self: Rule.SHORT_LABELS.get(self.__class__.__name__, self.__class__.__name__)