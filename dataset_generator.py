# -*- coding: utf-8 -*-
# 文件：dataset_generator.py
# 说明：基于真实逻辑规则的LSAT风格数据集生成器（保持原有 DAG 可视化风格）

import json
import logging
import random
import os
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path

from dag.dag_builder import build_reasoning_dag
from dag.validator import validate_logical_steps
from dag.visualizer import visualize_dag  # 保持原有可视化

from distractor.generator import DistractorGenerator
from api_key.llm_dispatcher import LLMDispatcher
from utils.consistency_validator import ConsistencyValidator
from utils.enhanced_prompt_builder import EnhancedPromptBuilder
from utils.variable_manager import EnhancedVariableExtractor

# 仅用于“克制版”冗余判定（显然废话）
from utils.tautology_check import parse_formula, quick_obvious_redundancy
from z3 import Bool


class DatasetGenerator:
    """简洁版数据集生成器：克制冗余过滤 + 清晰统计 + 干扰项去重 + 原风格可视化"""

    def __init__(
        self,
        llm_dispatcher: LLMDispatcher,
        prompt_template_path: str,
        max_variables: int = 8,
        min_variables: int = 3,
        enable_visualization: bool = True,
        viz_output_dir: str = "output/dag_visualizations",
    ) -> None:
        self.llm = llm_dispatcher
        self.logger = logging.getLogger("dataset_generator")

        self.extractor = EnhancedVariableExtractor(max_variables=max_variables, min_variables=min_variables)
        self.validator = ConsistencyValidator(strictness_level="medium")
        self.prompt_builder = EnhancedPromptBuilder(prompt_template_path)

        self.max_retry_attempts = 5
        self.min_valid_steps = 2
        self.max_valid_steps = 6  # 注意：实际截断会跟随 target_depth_range 动态放宽

        self.max_variables = max_variables
        self.min_variables = min_variables

        self.enable_visualization = enable_visualization
        self.viz_output_dir = viz_output_dir
        if self.enable_visualization:
            Path(self.viz_output_dir).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"可视化启用：{self.viz_output_dir}")

    # -------------------- 可视化（仍用原 visualize_dag） --------------------
    def _generate_dag_visualization(self, root_node, sample_id: str, metadata: Optional[Dict] = None) -> Optional[str]:
        if not (self.enable_visualization and root_node):
            return None
        try:
            # 用“保留步数”和“实际使用变量数”命名，避免歧义
            steps_kept = (metadata or {}).get("steps", {}).get("kept", None)
            vars_used = (metadata or {}).get("variables", {}).get("used", None)
            if steps_kept is not None and vars_used is not None:
                filename = f"dag_{sample_id}_d{steps_kept}_v{vars_used}"
            else:
                depth0 = (metadata or {}).get("reasoning_depth", 0)
                vcnt0 = (metadata or {}).get("variables_count", 0)
                filename = f"dag_{sample_id}_d{depth0}_v{vcnt0}"
            out_path_noext = os.path.join(self.viz_output_dir, filename)

            visualize_dag(root_node, filename=out_path_noext, format="png", style="modern")
            out_path = f"{out_path_noext}.png"
            self.logger.info(f"已生成可视化：{out_path}")
            return out_path
        except Exception as e:
            self.logger.warning(f"可视化失败：{e}")
            return None

    # -------------------- 变量提取 --------------------
    def _extract_variables_from_dag(self, root_node) -> List[str]:
        try:
            variables = self.extractor.extract_from_dag(root_node)
            if not variables:
                self.logger.warning("从DAG未提取到变量")
                return []
            normalized = self.extractor.normalize_variable_names(variables)
            self.logger.info(f"变量数（抽取）: {len(normalized)}")
            return normalized
        except Exception as e:
            self.logger.error(f"变量提取失败：{e}")
            return []

    def _extract_variables_from_steps(self, logical_steps: List[Dict]) -> List[str]:
        try:
            variables = self.extractor.extract_from_steps(logical_steps)
            normalized = self.extractor.normalize_variable_names(variables)
            if normalized:
                self.logger.info(f"从步骤提取变量数（抽取）: {len(normalized)}")
            return normalized
        except Exception as e:
            self.logger.error(f"步骤变量提取失败：{e}")
            return []

    # -------------------- 语义绑定 --------------------
    def _generate_semantic_bindings(self, variables: List[str]) -> Dict[str, str]:
        pool = [
            "completed_coursework", "passed_examinations", "submitted_thesis", "attended_seminars",
            "received_approval", "qualified_for_degree", "defended_research", "earned_certification",
            "published_work", "won_recognition", "met_requirements", "achieved_standards",
            "project_initiated", "budget_approved", "team_assembled", "milestone_achieved",
            "client_satisfied", "contract_finalized", "payment_processed", "quality_verified",
        ]
        bindings: Dict[str, str] = {}
        for i, v in enumerate(variables):
            term = pool[i % len(pool)]
            bindings[v] = term if term not in bindings.values() else f"{term}_{i}"
        return bindings

    # -------------------- 冗余判定（克制版） --------------------
    def _is_redundant_implication_safe(self, premise_str: str, conclusion_str: str, var_bindings: Dict[str, str]) -> bool:
        """仅过滤“显然废话”：p==q、False->α、α->True；不误删 And(...)→x / MP / HS 等正常推理。"""
        try:
            symbols = {sem: Bool(sem) for sem in var_bindings.values()}
            P = parse_formula(premise_str, symbols=symbols, prefer="auto")
            Q = parse_formula(conclusion_str, symbols=symbols, prefer="auto")
            return bool(quick_obvious_redundancy(P, Q))
        except Exception:
            # 解析失败时保守不过滤
            try:
                p, q = premise_str.strip(), conclusion_str.strip()
                if p == q:
                    return True
                if q in ("True", "BoolVal(True)"):
                    return True
            except Exception:
                pass
            return False

    # -------------------- Z3 表达式格式化（+统计） --------------------
    def _format_z3_expressions(
        self,
        logical_steps: List[Dict],
        var_bindings: Dict[str, str]
    ) -> Tuple[List[str], int, Set[str]]:
        """
        返回 (z3_exprs, steps_kept, used_sem_vars)
        - steps_kept：冗余过滤后保留的蕴含步数
        - used_sem_vars：在保留的蕴含式中出现过的“语义变量”集合
        """
        z3_exprs: List[str] = []
        used_sem_vars: Set[str] = set()
        steps_kept = 0
        try:
            # 变量声明（保持原格式）
            for _, sem in var_bindings.items():
                z3_exprs.append(f"{sem} = Bool('{sem}')")

            seen = set()
            for step in logical_steps:
                premises_expr = step.get("premises_expr", [])
                conclusion_expr = step.get("conclusion_expr")
                if not premises_expr or conclusion_expr is None:
                    continue

                def _subst(s: str) -> str:
                    for var, sem in var_bindings.items():
                        s = s.replace(var, sem)
                    return s

                premise_strs: List[str] = []
                for p in premises_expr:
                    try:
                        premise_strs.append(_subst(str(p)))
                    except Exception:
                        continue
                if not premise_strs:
                    continue
                try:
                    conclusion_str = _subst(str(conclusion_expr))
                except Exception:
                    continue

                # 克制版冗余过滤（只删显然废话）
                premise_for_check = (
                    premise_strs[0] if len(premise_strs) == 1
                    else f"And({', '.join(premise_strs)})"
                )
                if self._is_redundant_implication_safe(premise_for_check, conclusion_str, var_bindings):
                    continue

                implication = (
                    f"Implies({premise_strs[0]}, {conclusion_str})"
                    if len(premise_strs) == 1
                    else f"Implies(And({', '.join(premise_strs)}), {conclusion_str})"
                )
                if implication in seen:
                    continue

                z3_exprs.append(implication)
                seen.add(implication)
                steps_kept += 1

                # 统计实际使用的语义变量
                for sem in var_bindings.values():
                    if sem in implication:
                        used_sem_vars.add(sem)

        except Exception as e:
            self.logger.error(f"Z3 格式化失败：{e}")
        return z3_exprs, steps_kept, used_sem_vars

    # -------------------- 干扰项 --------------------
    def _create_distractors(self, logical_steps: List[Dict], var_bindings: Dict[str, str]) -> List[str]:
        try:
            var_names = list(var_bindings.keys())
            safe_vars = self.extractor.create_safe_bool_vars(var_names)
            if not safe_vars:
                return self._fallback_distractors()

            strategies = [
                "illogical_reasoning",
                "adversarial_structure",
                "reversed_implication",
                "unrelated_fact",
                "logical_complexity",
            ]
            distractor_gen = DistractorGenerator(available_vars=safe_vars, enabled_strategies=strategies)
            distractors = distractor_gen.generate_all(logical_steps, num_per_strategy=2)

            # 去重并限制为 3 个
            out: List[str] = []
            for d in distractors:
                desc = d.get("description") or d.get("text") or d.get("expr") or str(d)
                s = str(desc)
                if s not in out:
                    out.append(s)
                if len(out) >= 3:
                    break

            # 不足 3 个时补齐 fallback
            if len(out) < 3:
                for fb in self._fallback_distractors():
                    if fb not in out:
                        out.append(fb)
                    if len(out) >= 3:
                        break
            return out
        except Exception as e:
            self.logger.warning(f"干扰项生成失败：{e}")
            return self._fallback_distractors()

    @staticmethod
    def _fallback_distractors() -> List[str]:
        # 与 5 策略语义对齐，选择其中 3 种作为兜底
        return [
            "不充分前提/忽略关键假设（illogical_reasoning）",
            "方向反转的错误结论（reversed_implication）",
            "无关条件造成的干扰（unrelated_fact）",
        ]

    # -------------------- 样本生成 --------------------
    def generate_single_sample(
        self,
        max_depth: int = 3,
        sample_id: Optional[str] = None,
        target_depth_range: Optional[Tuple[int, int]] = None,  # 新增：最终“保留深度”必须落在这个区间
    ) -> Optional[Dict[str, Any]]:
        sample_id = sample_id or f"{random.randint(10000, 99999)}"
        for attempt in range(self.max_retry_attempts):
            try:
                self.logger.info(f"构建DAG：深度={max_depth}，尝试={attempt + 1}")
                root, logical_steps = build_reasoning_dag(
                    max_depth=max_depth,
                    min_depth=max(max_depth // 3, 2)
                )
                if not logical_steps:
                    self.logger.warning("逻辑步骤为空，重试")
                    continue

                steps_built = len(logical_steps)
                valid_steps_all, _ = validate_logical_steps(logical_steps)
                if len(valid_steps_all) < self.min_valid_steps:
                    self.logger.warning("有效步骤过少，重试")
                    continue

                # 截断上限：跟随目标深度上限（如果提供），避免被固定 6 限制住
                cap = (target_depth_range[1] if target_depth_range else self.max_valid_steps)
                if cap is None or cap <= 0:
                    cap = len(valid_steps_all)
                valid_steps = valid_steps_all[: min(cap, len(valid_steps_all))]
                steps_valid = len(valid_steps)

                # 变量与绑定（先按抽取数判定）
                variables = self._extract_variables_from_dag(root) or self._extract_variables_from_steps(valid_steps)
                extracted_cnt = len(variables)
                if extracted_cnt == 0:
                    self.logger.warning("变量抽取为空，重试")
                    continue
                var_bindings = self._generate_semantic_bindings(variables)

                # Z3 表达式 + 统计（基于保留步）
                z3_exprs, steps_kept, used_sem_vars = self._format_z3_expressions(valid_steps, var_bindings)
                used_cnt = len(used_sem_vars)

                # ------------ 约束检查（最终产物必须满足） ------------
                # 1) 最终保留深度 ∈ target_depth_range（若提供）
                if target_depth_range:
                    dmin, dmax = target_depth_range
                    if not (dmin <= steps_kept <= dmax):
                        self.logger.info(
                            f"丢弃样本：steps_kept={steps_kept} 不在目标区间 [{dmin},{dmax}] 内，重试"
                        )
                        continue

                # 2) 最终实际变量数 ∈ [min_variables, max_variables]
                if not (self.min_variables <= used_cnt <= self.max_variables):
                    self.logger.info(
                        f"丢弃样本：variables.used={used_cnt} 不在 [{self.min_variables},{self.max_variables}] 内，重试"
                    )
                    continue

                self.logger.info(f"变量数（实际使用）: {used_cnt}; 步骤 kept: {steps_kept}")

                # Prompt 构造
                prompt = self.prompt_builder.build_constrained_prompt(
                    z3_exprs=z3_exprs,
                    var_bindings=var_bindings,
                    logical_steps=valid_steps,
                    previous_violations=[],
                )

                # 调用 LLM
                self.logger.info("调用 LLM 生成题目…")
                resp = self.llm.call(prompt)
                if not resp:
                    self.logger.error("LLM 响应为空，重试")
                    continue

                sample = self._parse_llm_response(resp)
                if not sample:
                    self.logger.error("解析 LLM 响应失败，重试")
                    continue

                # 干扰项
                sample.setdefault("distractors", self._create_distractors(valid_steps, var_bindings))

                # 质量校验
                is_valid, violations = self.validator.validate_sample(sample)
                if not is_valid:
                    self.logger.warning(f"质量校验警告：{violations}")

                # 元数据（清晰拆分）
                sample["metadata"] = {
                    "steps": {"built": steps_built, "validated": steps_valid, "kept": steps_kept},
                    "variables": {"extracted": extracted_cnt, "used": used_cnt},
                    "variable_control": {"max": self.max_variables, "min": self.min_variables, "actual": used_cnt},
                    "reasoning_depth": steps_kept,          # 兼容旧字段
                    "variables_count": extracted_cnt,        # 兼容旧字段
                    "has_warnings": not is_valid,
                }

                # 可视化（保持原风格）
                if root is not None:
                    viz = self._generate_dag_visualization(root, sample_id, sample.get("metadata"))
                    if viz:
                        sample["visualization_path"] = viz

                return sample

            except Exception as e:
                self.logger.error(f"生成失败(第{attempt + 1}次)：{e}")
                continue
        self.logger.error("多次重试仍失败")
        return None

    def _parse_llm_response(self, response: str) -> Optional[Dict]:
        try:
            i, j = response.find("{"), response.rfind("}") + 1
            if i == -1 or j <= i:
                return None
            data = json.loads(response[i:j])
            if not all(k in data for k in ("context", "question", "answers", "label")):
                return None
            if not (isinstance(data["answers"], list) and len(data["answers"]) == 4):
                return None
            if data["label"] not in ("A", "B", "C", "D"):
                return None
            return data
        except Exception:
            return None

    # -------------------- 数据集批量生成 --------------------
    def generate_dataset(self, num_samples: int, output_path: str, max_depth_range: Tuple[int, int] = (5, 8)) -> None:
        self.logger.info(f"开始生成数据集：{num_samples} 条")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        ok: List[Dict[str, Any]] = []
        attempts = 0
        max_attempts = max(num_samples * 6, num_samples + 6)  # 略微提高重试上限，满足区间更稳定

        dmin, dmax = max_depth_range

        while len(ok) < num_samples and attempts < max_attempts:
            attempts += 1
            # 仍按你原来的做法随机给 builder 一个目标深度；最终“保留深度”再用区间校验
            build_depth = random.randint(dmin, dmax)
            sid = f"sample_{len(ok)+1:04d}_{attempts:04d}"

            sample = self.generate_single_sample(
                max_depth=build_depth,
                sample_id=sid,
                target_depth_range=max_depth_range,  # 关键：最终产物必须在这个区间
            )

            if sample:
                ok.append(sample)
                self.logger.info(f"已生成：{len(ok)}/{num_samples}")
            else:
                self.logger.warning("本次失败，继续…")

            # 逐条写入（节省内存）
            with open(output_path, "a", encoding="utf-8") as f:
                if sample:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        self.logger.info(f"完成，输出：{output_path}")


# -------------------- 示例主函数 --------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    llm = LLMDispatcher(model_name="deepseek-chat", api_key_path="api_key/ds-api_key.txt", retries=3)
    generator = DatasetGenerator(
        llm_dispatcher=llm,
        prompt_template_path="prompt/lsat_prompt.txt",
        max_variables=20,
        min_variables=5,
        enable_visualization=True,
        viz_output_dir="output/dag_visualizations",
    )

    # 小批量验证
    out_file = "output/lsat_dataset_clean.jsonl"
    if os.path.exists(out_file):
        os.remove(out_file)
    generator.generate_dataset(num_samples=1, output_path=out_file, max_depth_range=(8, 15))


if __name__ == "__main__":
    main()
