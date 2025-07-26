from typing import List, Dict
from z3 import Solver, Bool, Not, unsat
from rules.rule import Rule
from utils.logger_utils import setup_logger

logger = setup_logger("verify_leaf")


def verify_leaf_reachability(
    leaf_nodes: List[str],
    rules: List[Rule],
    symbols: Dict[str, Bool]
) -> List[str]:
    """
    验证逻辑DAG中每个叶子节点（结论）是否能通过规则集从已知事实推出。

    :param leaf_nodes: 叶子结论表达式列表（如 'And(A, B)', 'P' 等）
    :param rules: 构建当前样本的全部规则对象
    :param symbols: 所有变量名与 z3.Bool 对象的映射关系（由 sample_builder 构建）
    :return: 可达的叶子节点列表（字符串表达式）
    """
    reachable = []

    for leaf in leaf_nodes:
        matched_rule = None
        for rule in rules:
            if rule.get_conclusion_expr() == leaf:
                matched_rule = rule
                break

        if matched_rule is None:
            logger.warning(f"[Verify] No matching rule for leaf: {leaf}")
            continue

        conclusion_var = matched_rule.get_conclusion_var()
        if conclusion_var not in symbols:
            logger.warning(f"[Verify] {conclusion_var} not in symbols, skipping.")
            continue

        # 构建新求解器，添加所有规则约束
        solver = Solver()
        for rule in rules:
            try:
                rule.apply_z3(solver, symbols)
            except Exception as e:
                logger.warning(f"[Verify] Failed to apply rule {rule}: {e}")

        # 加入否定目标结论
        solver.add(Not(symbols[conclusion_var]))

        try:
            if solver.check() == unsat:
                logger.info(f"[Verify] Leaf {leaf} is reachable.")
                reachable.append(leaf)
            else:
                logger.info(f"[Verify] Leaf {leaf} is NOT reachable.")
        except Exception as e:
            logger.warning(f"[Verify] Failed on {leaf}: {e}")

    logger.info(f"[Verify] Reachable leaf count: {len(reachable)} / {len(leaf_nodes)}")
    return reachable
