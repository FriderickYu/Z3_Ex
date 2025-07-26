from z3 import sat, unsat, unknown

class Z3ResultParser:
    """封装对 Z3 求解结果的处理与解释"""

    @staticmethod
    def parse_result(solver, model_required=True):
        result = solver.check()
        if result == sat:
            model = solver.model() if model_required else None
            return {"status": "SAT", "model": str(model) if model else None}
        elif result == unsat:
            return {"status": "UNSAT", "model": None}
        elif result == unknown:
            return {"status": "UNKNOWN", "reason": solver.reason_unknown()}
        else:
            return {"status": "ERROR", "model": None}