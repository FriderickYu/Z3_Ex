# 文件：dag/validator.py
# 说明：Z3 验证模块，用于验证 logical_steps 中每一步推理的逻辑成立性

import z3


def validate_logical_steps(steps):
    """
    使用 Z3 验证每一步 logical step 是否逻辑成立
    返回：
        valid_steps: 验证通过的步骤列表
        failed_steps: 验证失败的步骤列表（含错误信息）
    """
    valid_steps = []
    failed_steps = []

    for step in steps:
        premises = step.get('premises_expr')
        conclusion = step.get('conclusion_expr')

        # ❗ 防止缺失表达式导致异常
        if not premises or conclusion is None:
            step["error"] = "缺失 Z3 表达式对象"
            failed_steps.append(step)
            continue

        s = z3.Solver()
        s.add(*premises)
        s.add(z3.Not(conclusion))

        if s.check() == z3.unsat:
            valid_steps.append(step)
        else:
            step["error"] = "逻辑不成立"
            failed_steps.append(step)

    return valid_steps, failed_steps
