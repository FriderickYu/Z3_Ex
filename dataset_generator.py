# -*- coding: utf-8 -*-
# 文件：dataset_generator.py
# 说明：基于真实逻辑规则的LSAT风格数据集生成器（保持原有 DAG 可视化风格）

import json
import logging
import random
import os
import uuid
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
        # 只保留两种模式：gibberish / llm
        semantic_binding_mode: str = "gibberish",
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

        # 仅两种：gibberish（默认）| llm
        self.semantic_binding_mode = semantic_binding_mode if semantic_binding_mode in ("gibberish", "llm") else "gibberish"

    # -------------------- 可视化（仍用原 visualize_dag） --------------------
    def _generate_dag_visualization(self, root_node, sample_id: str, metadata: Optional[Dict] = None) -> Optional[str]:
        """使用 sample_id（此处即 UUID）作为图文件名"""
        if not (self.enable_visualization and root_node):
            return None
        try:
            filename = sample_id  # 图名=UUID
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

    # -------------------- 语义绑定（仅 gibberish / llm） --------------------
    def _generate_semantic_bindings(self, variables: List[str]) -> Dict[str, str]:
        """根据配置返回变量到“语义名”的绑定。
        模式：
          - gibberish: 使用无实际语义的伪词（默认，推荐做纯逻辑推理）
          - llm: 交给 LLM 起名（不限定领域），失败则回退 gibberish
        """
        mode = self.semantic_binding_mode
        if not variables:
            return {}

        if mode == "gibberish":
            names = self._gen_pseudowords(len(variables))
            return {v: n for v, n in zip(variables, names)}

        # mode == "llm"
        mapping = self._generate_llm_semantics(variables)
        if mapping:
            return mapping
        # 兜底回退：gibberish
        names = self._gen_pseudowords(len(variables))
        return {v: n for v, n in zip(variables, names)}

    def _gen_pseudowords(self, n: int) -> List[str]:
        """生成 n 个无语义、可读性较好的伪词；高随机性 + 去重（ASCII 小写）。"""
        consonants = list("bcdfghjklmnpqrstvwxz")
        vowels = list("aeiouy")

        def one_word():
            # 2~4 个音节，随机尾辅音，随机插入重复音节，提升多样性
            syls = random.randint(2, 4)
            s = ""
            for _ in range(syls):
                s += random.choice(consonants) + random.choice(vowels)
                if random.random() < 0.45:
                    s += random.choice(consonants)
                if random.random() < 0.18:  # 偶尔重复一个元音，拉长
                    s += random.choice(vowels)
            # 5% 概率添加短后缀增强区分
            if random.random() < 0.05:
                s += "_" + random.choice("abcdefghijklmnopqrstuvwxyz")
            return s

        out, seen = [], set()
        while len(out) < n:
            w = one_word()
            # 确保唯一性；碰撞时加随机数字后缀
            if w in seen:
                w = f"{w}{random.randint(2, 97)}"
                if w in seen:
                    continue
            seen.add(w)
            out.append(w)
        return out

    def _generate_llm_semantics(self, variables: List[str]) -> Dict[str, str]:
        """用 LLM 起名；返回 {V: name}；失败返回 {}。
        规范化为 snake_case ASCII，小写；名称不强加领域限制。
        """
        try:
            var_list = ", ".join(variables)
            sys_prompt = (
                "You will assign short, diverse proposition names for logic variables. "
                "Return a pure JSON object; keys are the original variable tokens "
                "(e.g., V1, V2); values are concise snake_case ASCII names. "
                "Avoid long phrases; ensure uniqueness."
            )
            user_prompt = (
                f"Variables: [{var_list}]. "
                "Return only JSON like {\"V1\":\"wernyra\",\"V2\":\"alert_raised\",...} with unique values."
            )
            resp = self.llm.call(f"{sys_prompt}\n{user_prompt}")
            i, j = resp.find("{"), resp.rfind("}") + 1
            if i == -1 or j <= i:
                return {}
            mapping = json.loads(resp[i:j])

            # 规范化 & 去重
            out, used = {}, set()
            for v in variables:
                cand = str(mapping.get(v, "")).strip().lower()
                if not cand:
                    continue
                cand = "".join(ch for ch in cand if ch.isalnum() or ch == "_")
                cand = cand or random.choice(self._gen_pseudowords(1))
                if cand in used:
                    cand = f"{cand}_{random.randint(2, 99)}"
                used.add(cand)
                out[v] = cand

            # 如果不足，补齐
            if len(out) < len(variables):
                extras = self._gen_pseudowords(len(variables) - len(out))
                for v, e in zip([x for x in variables if x not in out], extras):
                    out[v] = e
            return out
        except Exception:
            return {}

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
        target_depth_range: Optional[Tuple[int, int]] = None,
    ) -> Optional[Dict[str, Any]]:
        # 为每条样本分配 UUID；统一使用 UUID 作为样本 ID 与图名
        sample_uuid = str(uuid.uuid4())

        for attempt in range(self.max_retry_attempts):
            try:
                self.logger.info(f"构建DAG：深度={max_depth}，尝试={attempt + 1}")
                root, logical_steps = build_reasoning_dag(
                    max_depth=max_depth,
                    min_depth=max(max_depth // 3, 2)
                )

                # 去重统计本次用到的规则
                rules_used = sorted({
                    (s.get("rule") or "").strip()
                    for s in (logical_steps or [])
                    if s.get("rule") and s.get("rule") != "Unknown"
                })
                self.logger.info("Rules used (dedup): %s", ", ".join(rules_used) if rules_used else "(none)")

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

                # Prompt 构造（gibberish 时温和提醒“聚焦逻辑，不引入现实语义”）
                extra_guidance = ""
                if self.semantic_binding_mode == "gibberish":
                    extra_guidance = (
                        "Note: variable names are abstract tokens without real-world semantics. "
                        "Focus on logical consistency; do not inject external world knowledge."
                    )

                prompt = self.prompt_builder.build_constrained_prompt(
                    z3_exprs=z3_exprs,
                    var_bindings=var_bindings,
                    logical_steps=valid_steps,
                    previous_violations=[extra_guidance] if extra_guidance else [],
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

                # 基本字段 & 元数据
                sample["id"] = sample_uuid  # 用 UUID 作为样本 ID
                sample["z3"] = z3_exprs
                sample["metadata"] = {
                    "uuid": sample_uuid,
                    "steps": {"built": steps_built, "validated": steps_valid, "kept": steps_kept},
                    "variables": {"extracted": extracted_cnt, "used": used_cnt},
                    "variable_control": {"max": self.max_variables, "min": self.min_variables, "actual": used_cnt},
                    "reasoning_depth": steps_kept,          # 兼容旧字段
                    "variables_count": extracted_cnt,        # 兼容旧字段
                    "has_warnings": not is_valid,
                    "rules_used": rules_used,                # 规则清单（去重）
                    "semantic_mode": self.semantic_binding_mode,  # 记录当前语义绑定模式
                }

                # 可视化（图名=UUID）
                if root is not None:
                    viz = self._generate_dag_visualization(root, sample_uuid, sample.get("metadata"))
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

            sample = self.generate_single_sample(
                max_depth=build_depth,
                sample_id=None,                  # 内部统一使用 UUID
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
        semantic_binding_mode="gibberish",  # 或 "llm"
    )

    # 小批量验证
    out_file = "output/lsat_dataset_clean.jsonl"
    if os.path.exists(out_file):
        os.remove(out_file)
    generator.generate_dataset(num_samples=2, output_path=out_file, max_depth_range=(3, 10))


if __name__ == "__main__":
    main()