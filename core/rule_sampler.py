from typing import Optional, Tuple, List
from z3 import Solver, Bool, sat
from rules.rules_pooling import RulesPooling
from utils.logger_utils import setup_logger

logger = setup_logger("rule_sampler")


class RuleSampler:
    """
    负责规则采样与Z3验证逻辑一致性。
    """

    def __init__(self, rule_pool: RulesPooling):
        self.rule_pool = rule_pool

    def sample_and_validate(self, max_rules: int) -> Optional[Tuple[List, List[str]]]:
        """
        采样规则并验证。

        :param max_rules: 最大规则数
        :return: 验证成功返回(规则列表, z3表达式列表)，否则返回None
        """
        rules = self.rule_pool.sample_rules(max_rules)
        symbols = {name: Bool(name) for rule in rules for name in rule.get_symbol_names()}

        solver = Solver()
        for rule in rules:
            rule.apply_z3(solver, symbols)

        solver.add(symbols[rules[0].get_symbol_names()[0]])

        if solver.check() == sat:
            z3_exprs = [expr for rule in rules for expr in rule.to_z3()]
            return rules, z3_exprs
        else:
            logger.warning("Z3 validation failed.")
            return None