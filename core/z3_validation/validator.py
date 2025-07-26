from z3 import Solver, parse_smt2_string
from .base import BaseValidator
from .result_parser import Z3ResultParser
from .error_handler import Z3ErrorHandler

class Z3Validator(BaseValidator):
    """主验证器类，构造求解器并执行表达式验证"""

    def __init__(self):
        self.solver = Solver()

    def check(self, expression: str) -> bool:
        try:
            self.solver.reset()
            expr = parse_smt2_string(expression)
            self.solver.add(expr)
            result = self.solver.check()
            return result.r == 1  # sat
        except Exception as e:
            Z3ErrorHandler.handle_exception(e, context="check")
            return False

    def validate(self, data: dict) -> dict:
        try:
            self.solver.reset()
            smt_str = data.get("smt", "")
            expr = parse_smt2_string(smt_str)
            self.solver.add(expr)
            return Z3ResultParser.parse_result(self.solver)
        except Exception as e:
            return Z3ErrorHandler.handle_exception(e, context="validate")