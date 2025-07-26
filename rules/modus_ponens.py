from typing import List, Dict
from z3 import Implies, Bool, Solver
from rules.rule import Rule
from utils.logger_utils import setup_logger


logger = setup_logger("modus_ponens")


class ModusPonens(Rule):
    """
    表示逻辑规则 Modus Ponens: 如果P则Q
    """
    def __init__(self, antecedent_var: str, consequent_var: str):
        """
        初始化前件与后件变量名。

        :param antecedent_var
        :param consequent_var
        """
        self.antecedent = antecedent_var
        self.consequent = consequent_var

    def to_z3(self) -> List[str]:
        """
        返回构成Z3表达式的代码字符串（用于prompt构造）。
        :return: 包含Bool定义和Implies表达式的字符串列表
        """
        return [
            f"{self.antecedent} = Bool('{self.antecedent}')",
            f"{self.consequent} = Bool('{self.consequent}')",
            f"Implies({self.antecedent}, {self.consequent})"
        ]

    def get_main_z3_expr(self) -> str:
        return f"Implies({self.antecedent}, {self.consequent})"

    def get_conclusion_expr(self) -> str:
        return self.consequent

    def get_conclusion_var(self) -> str:
        return self.consequent

    def apply_z3(self, solver: Solver, symbols: dict):
        """
        向 Z3 求解器添加当前规则对应的约束。

        :param solver: Z3求解器对象
        :param symbols: 字符符号映射表（str -> Bool）
        """
        logger.info(f"Applying Implies({self.antecedent}, {self.antecedent}) to solver.")
        solver.add(Implies(symbols[self.antecedent], symbols[self.consequent]))

    def get_symbol_names(self) -> List[str]:
        """
        返回该规则中涉及的变量名（用于prompt占位替换）。

        :return: 包含前件与后件名的列表
        """
        logger.info(f"Symbol names assigned: {self.antecedent} and {self.consequent}.")
        return [self.antecedent, self.consequent]

    @staticmethod
    def required_vars() -> int:
        return 2

    def describe(self) -> str:
        return f"If {self.antecedent} is true, then {self.consequent} must also be true."

    def get_short_label(self) -> str:
        return "→Modus"

    def get_descriptions(self) -> List[Dict[str, str]]:
        return [
            {"var": self.antecedent, "description": f"{self.antecedent} is assumed true"},
            {"var": self.consequent, "description": f"{self.consequent} is deduced from implication"}
        ]