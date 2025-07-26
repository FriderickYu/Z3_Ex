from typing import List, Dict
from z3 import Solver, And
from rules.rule import Rule
from utils.logger_utils import setup_logger

logger = setup_logger("conjunction_elimination")


class ConjunctionElimination(Rule):
    """
    表示合取消解规则: 从P∧Q推出P（或Q）
    """

    def __init__(self, left_var: str, right_var: str):
        self.left = left_var
        self.right = right_var

    def to_z3(self) -> List[str]:
        return [
            f"{self.left} = Bool('{self.left}')",
            f"{self.right} = Bool('{self.right}')",
            f"conj = And({self.left}, {self.right})",
            f"{self.left}"
        ]

    def get_main_z3_expr(self) -> str:
        return f"Implies(And({self.left}, {self.right}), {self.left})"

    def get_conclusion_expr(self) -> str:
        return self.left

    def get_conclusion_var(self) -> str:
        return self.left

    def apply_z3(self, solver: Solver, symbols: dict):
        logger.info(f"Adding And({self.left}, {self.right}) and extracting {self.left}.")
        conj = And(symbols[self.left], symbols[self.right])
        solver.add(conj)
        solver.add(symbols[self.left])

    def get_symbol_names(self) -> List[str]:
        logger.info(f"Symbol names assigned: {self.left}, {self.right}.")
        return [self.left, self.right]

    @staticmethod
    def required_vars() -> int:
        return 2

    def describe(self) -> str:
        return f"If both {self.left} and {self.right} are true, then {self.left} must also be true."

    def get_short_label(self) -> str:
        return "∧Elim"

    def get_descriptions(self) -> List[Dict[str, str]]:
        return [
            {
                "var": f"And({self.left}, {self.right})",
                "description": f"Both {self.left} and {self.right} are assumed to be true"
            },
            {
                "var": self.left,
                "description": f"{self.left} is extracted from the conjunction"
            }
        ]
