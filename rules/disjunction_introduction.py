from typing import List, Dict
from z3 import Solver, Or
from rules.rule import Rule
from utils.logger_utils import setup_logger

logger = setup_logger("disjunction_introduction")


class DisjunctionIntroduction(Rule):
    """
    表示析取引入规则: 如果P成立, 则 P∨Q成立。
    """

    def __init__(self, known_var: str, added_var: str):
        """
        :param known_var: 已知为真的变量(如P)
        :param added_var: 添加到析取中的另一个变量
        """
        self.known = known_var
        self.added = added_var

    def to_z3(self) -> List[str]:
        return [
            f"{self.known} = Bool('{self.known}')",
            f"{self.added} = Bool('{self.added}')",
            f"Or({self.known}, {self.added})"
        ]

    def get_main_z3_expr(self) -> str:
        return f"Implies({self.known}, Or({self.known}, {self.added}))"

    def get_conclusion_expr(self) -> str:
        return f"Or({self.known}, {self.added})"

    def get_conclusion_var(self) -> str:
        return f"Or({self.known}, {self.added})"

    def apply_z3(self, solver: Solver, symbols: dict):
        logger.info(f"Applying Or({self.known}, {self.added}) to solver.")
        solver.add(Or(symbols[self.known], symbols[self.added]))

    def get_symbol_names(self) -> List[str]:
        logger.info(f"Symbol names assigned: {self.known}, {self.added}.")
        return [self.known, self.added]

    @staticmethod
    def required_vars() -> int:
        return 2

    def describe(self) -> str:
        return f"If {self.known} is true, then {self.known} ∨ {self.added} is also true."

    def get_short_label(self) -> str:
        return "∨Intro"

    def get_descriptions(self) -> List[Dict[str, str]]:
        return [
            {"var": self.known, "description": f"{self.known} is known to be true"},
            {"var": self.added, "description": f"{self.added} is added into disjunction"},
            {"var": f"Or({self.known}, {self.added})",
             "description": f"The disjunction {self.known} ∨ {self.added} is inferred"}
        ]
