from typing import List, Dict
from z3 import Solver, And
from rules.rule import Rule
from utils.logger_utils import setup_logger

logger = setup_logger("conjunction_introduction")


class ConjunctionIntroduction(Rule):
    """
    表示合取引入规则: 如果P与Q成立, 则P∧Q成立
    """
    def __init__(self, left_var: str, right_var: str):
        self.left = left_var
        self.right = right_var

    def to_z3(self) -> List[str]:
        return [
            f"{self.left} = Bool('{self.left}')",
            f"{self.right} = Bool('{self.right}')",
            f"And({self.left}, {self.right})"
        ]

    def get_main_z3_expr(self) -> str:
        return f"And({self.left}, {self.right})"

    def get_conclusion_expr(self) -> str:
        return self.get_main_z3_expr()

    def get_conclusion_var(self) -> str:
        return f"And({self.left}, {self.right})"

    def apply_z3(self, solver: Solver, symbols: dict):
        logger.info(f"Applying And({self.left}, {self.right}) to solver.")
        solver.add(And(symbols[self.left], symbols[self.right]))

    def get_symbol_names(self) -> List[str]:
        logger.info(f"Symbol names assigned: {self.left}, {self.right}.")
        return [self.left, self.right]

    @staticmethod
    def required_vars() -> int:
        return 2

    def describe(self) -> str:
        return f"If both {self.left} and {self.right} are true, then {self.left} ∧ {self.right} is true."

    def get_short_label(self) -> str:
        return "∧Intro"

    def get_descriptions(self) -> List[Dict[str, str]]:
        return [
            {"var": self.left, "description": f"{self.left} is assumed true"},
            {"var": self.right, "description": f"{self.right} is assumed true"},
            {"var": f"And({self.left}, {self.right})",
             "description": f"The conjunction {self.left} ∧ {self.right} is inferred"}
        ]