# 文件：dataset_generator.py
# 说明：基于Z3推理结构和干扰项生成自然语言LSAT风格数据集

import json
import logging
import random
from typing import List, Dict, Any
from pathlib import Path

from dag.dag_builder import build_reasoning_dag
from dag.validator import validate_logical_steps
from distractor.generator import DistractorGenerator
from api_key.llm_dispatcher import LLMDispatcher
from utils.consistency_validator import ConsistencyValidator
from utils.enhanced_prompt_builder import EnhancedPromptBuilder


# 内嵌安全变量提取器
class SafeVariableExtractor:
    """安全的变量提取器，避免Z3表达式布尔转换错误"""

    def __init__(self):
        self.logger = logging.getLogger("safe_variable_extractor")

    def extract_from_dag(self, root_node) -> List[str]:
        """从DAG中安全地提取所有变量名"""
        variables = set()

        def safe_traverse(node):
            try:
                if not hasattr(node, 'z3_expr'):
                    return

                expr = node.z3_expr
                if expr is None:
                    return

                try:
                    expr_str = str(expr)
                    import re
                    vars_in_expr = re.findall(r'Var_\d+', expr_str)
                    variables.update(vars_in_expr)
                except Exception as e:
                    self.logger.debug(f"提取表达式字符串时出错: {e}")

                if hasattr(node, 'children') and node.children:
                    for child in node.children:
                        if child is not None:
                            safe_traverse(child)

            except Exception as e:
                self.logger.debug(f"遍历节点时出错: {e}")

        safe_traverse(root_node)
        return sorted(list(variables))

    def create_safe_bool_vars(self, var_names: List[str], max_count: int = 10):
        """安全地创建Z3布尔变量"""
        import z3
        safe_vars = []

        for var_name in var_names[:max_count]:
            try:
                bool_var = z3.Bool(var_name)
                safe_vars.append(bool_var)
            except Exception as e:
                self.logger.debug(f"创建布尔变量 {var_name} 时出错: {e}")
                continue

        return safe_vars


class DatasetGenerator:
    """
    数据集生成器：将Z3推理结构转换为自然语言LSAT风格题目
    """

    def __init__(self, llm_dispatcher: LLMDispatcher, prompt_template_path: str):
        """
        初始化数据集生成器

        :param llm_dispatcher: LLM调度器实例
        :param prompt_template_path: prompt模板文件路径
        """
        self.llm = llm_dispatcher
        self.prompt_template = self._load_prompt_template(prompt_template_path)
        self.logger = logging.getLogger("dataset_generator")
        self.extractor = SafeVariableExtractor()
        self.validator = ConsistencyValidator()
        self.prompt_builder = EnhancedPromptBuilder(prompt_template_path)

    def _load_prompt_template(self, path: str) -> str:
        """加载prompt模板"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Failed to load prompt template: {e}")
            raise

    def _extract_variables_from_dag(self, root_node) -> List[str]:
        """从DAG中提取所有变量名"""
        return self.extractor.extract_from_dag(root_node)

    def _generate_variable_bindings(self, variables: List[str]) -> Dict[str, str]:
        """为变量生成语义绑定"""
        # 预定义的语义域
        domains = [
            # 学术场景
            ["student_passed_exam", "submitted_assignment", "attended_class", "received_grade"],
            # 商业场景
            ["project_completed", "budget_approved", "deadline_met", "client_satisfied"],
            # 日常场景
            ["weather_sunny", "traffic_light", "door_locked", "alarm_set"],
            # 法律场景
            ["evidence_submitted", "witness_testified", "contract_signed", "case_filed"]
        ]

        # 随机选择一个语义域
        domain = random.choice(domains)
        bindings = {}

        for i, var in enumerate(variables):
            if i < len(domain):
                bindings[var] = domain[i]
            else:
                # 如果变量太多，生成通用描述
                bindings[var] = f"condition_{i + 1}_holds"

        return bindings

    def _format_z3_expressions(self, logical_steps: List[Dict], var_bindings: Dict[str, str]) -> List[str]:
        """将逻辑步骤转换为Z3表达式字符串"""
        z3_exprs = []

        try:
            # 1. 变量声明
            for var in var_bindings:
                z3_exprs.append(f"{var} = Bool('{var}')")

            # 2. 规则表达式
            for step in logical_steps:
                premises = step.get('premises_expr', [])
                conclusion = step.get('conclusion_expr')

                # 安全检查，避免None值和空列表
                if premises and conclusion is not None:
                    try:
                        if len(premises) == 1:
                            premise_str = str(premises[0])
                        else:
                            premise_str = f"And({', '.join(str(p) for p in premises)})"

                        z3_exprs.append(f"Implies({premise_str}, {str(conclusion)})")
                    except Exception as e:
                        self.logger.debug(f"格式化步骤时出错: {e}")
                        continue

        except Exception as e:
            self.logger.error(f"格式化Z3表达式时出错: {e}")

        return z3_exprs

    def _format_reasoning_chain(self, logical_steps: List[Dict]) -> str:
        """格式化推理链为自然语言描述"""
        chain_parts = []

        for i, step in enumerate(logical_steps, 1):
            rule = step.get('rule', 'Unknown')
            premises = step.get('premises', [])
            conclusion = step.get('conclusion', '')

            if len(premises) == 1:
                premise_desc = premises[0]
            else:
                premise_desc = f"({', '.join(premises)})"

            chain_parts.append(
                f"Step {i}: If {premise_desc}, then {conclusion} (using {rule})"
            )

        return "\n".join(chain_parts)

    def _select_distractors(self, distractors: List[Dict], num_distractors: int = 3) -> List[Dict]:
        """选择高质量的干扰项"""
        if not distractors:
            return []

        try:
            # 按策略类型分组
            by_strategy = {}
            for d in distractors:
                strategy = d.get('strategy', 'unknown')
                if strategy not in by_strategy:
                    by_strategy[strategy] = []
                by_strategy[strategy].append(d)

            # 从每种策略中选择一个，确保多样性
            selected = []
            strategies = list(by_strategy.keys())
            random.shuffle(strategies)

            for strategy in strategies[:num_distractors]:
                if by_strategy[strategy]:
                    selected.append(random.choice(by_strategy[strategy]))

            # 如果不够，随机补充
            while len(selected) < num_distractors and len(selected) < len(distractors):
                remaining = [d for d in distractors if d not in selected]
                if remaining:
                    selected.append(random.choice(remaining))
                else:
                    break

            return selected[:num_distractors]

        except Exception as e:
            self.logger.error(f"选择干扰项时出错: {e}")
            return distractors[:num_distractors] if distractors else []

    def generate_single_sample(self, max_depth: int = 3) -> Dict[str, Any]:
        """生成单个数据样本（增加双向约束验证）"""
        max_constraint_attempts = 3  # 最大约束重试次数
        previous_violations = []

        for attempt in range(max_constraint_attempts):
            try:
                # 1. 构建推理DAG（保持原有逻辑）
                root, logical_steps = build_reasoning_dag(max_depth=max_depth)
                valid_steps, failed_steps = validate_logical_steps(logical_steps)

                if len(valid_steps) < 2:
                    self.logger.warning(f"推理步骤太少，重试 (attempt {attempt + 1})")
                    continue

                # 2. 提取变量和生成绑定（保持原有逻辑）
                variables = self._extract_variables_from_dag(root)
                if not variables:
                    self.logger.warning(f"未能提取到变量，重试 (attempt {attempt + 1})")
                    continue

                var_bindings = self._generate_variable_bindings(variables)

                # 3. 使用增强的prompt构建器
                enhanced_prompt = self.prompt_builder.build_constrained_prompt(
                    z3_exprs=self._format_z3_expressions(valid_steps, var_bindings),
                    var_bindings=var_bindings,
                    logical_steps=valid_steps,
                    previous_violations=previous_violations
                )

                # 4. 调用LLM
                self.logger.info(f"调用LLM生成自然语言题目 (attempt {attempt + 1})...")
                response = self.llm.call(enhanced_prompt)

                if not response:
                    self.logger.error(f"LLM调用失败，重试 (attempt {attempt + 1})")
                    continue

                # 5. 解析响应
                sample = self._parse_llm_response(response, valid_steps, var_bindings)
                if not sample:
                    self.logger.error(f"响应解析失败，重试 (attempt {attempt + 1})")
                    continue

                # 6. 双向约束验证
                is_valid, violations = self.validator.validate_sample(sample)

                if is_valid:
                    self.logger.info("✅ 样本通过双向约束验证")
                    return sample
                else:
                    self.logger.warning(f"❌ 约束验证失败 (attempt {attempt + 1}): {violations}")
                    previous_violations = violations

                    # 如果是最后一次尝试，返回部分合格的样本
                    if attempt == max_constraint_attempts - 1:
                        self.logger.info("返回部分合格样本")
                        sample['validation_warnings'] = violations
                        return sample

            except Exception as e:
                self.logger.error(f"生成样本时出错 (attempt {attempt + 1}): {e}")
                continue

        self.logger.error("所有约束重试都失败")
        return None

    def _parse_llm_response(self, response: str, valid_steps: List[Dict], var_bindings: Dict[str, str]) -> Dict:
        """解析LLM响应并添加元数据"""
        try:
            # 提取JSON（保持原有逻辑）
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)

                # 添加元数据（保持原有逻辑）
                result['metadata'] = {
                    'depth': len(valid_steps),
                    'variables_count': len(var_bindings),
                    'z3_logical_steps': [step.get('conclusion') for step in valid_steps],
                    'validation_passed': len(valid_steps),
                    'constraint_validation': 'pending'  # 新增：约束验证状态
                }

                return result
            else:
                self.logger.error("无法从LLM响应中提取JSON")
                return None

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败: {e}")
            return None

    def generate_dataset(self, num_samples: int, output_path: str, max_depth_range: tuple = (2, 4)) -> None:
        """生成完整数据集（添加约束验证统计）"""
        self.logger.info(f"开始生成 {num_samples} 个样本的数据集（启用双向约束验证）")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        successful_samples = []
        attempts = 0
        max_attempts = num_samples * 3
        constraint_stats = {"passed": 0, "failed": 0, "partial": 0}

        while len(successful_samples) < num_samples and attempts < max_attempts:
            attempts += 1
            depth = random.randint(*max_depth_range)

            self.logger.info(f"生成样本 {len(successful_samples) + 1}/{num_samples} (尝试 {attempts})")

            sample = self.generate_single_sample(max_depth=depth)
            if sample:
                successful_samples.append(sample)

                # 统计约束验证结果
                if 'validation_warnings' not in sample:
                    constraint_stats["passed"] += 1
                    self.logger.info(f"✅ 成功生成高质量样本 {len(successful_samples)}")
                else:
                    constraint_stats["partial"] += 1
                    self.logger.info(f"⚠️ 生成部分合格样本 {len(successful_samples)}")
            else:
                constraint_stats["failed"] += 1
                self.logger.warning(f"❌ 样本生成失败 (尝试 {attempts})")

        # 保存数据集
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in successful_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        # 输出统计信息
        self.logger.info(f"数据集生成完成: {len(successful_samples)} 个样本保存到 {output_path}")
        self.logger.info(
            f"约束验证统计: 完全通过={constraint_stats['passed']}, 部分通过={constraint_stats['partial']}, 失败={constraint_stats['failed']}")


def main():
    """主函数示例"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 初始化LLM调度器 (根据你的需要选择模型)
    llm = LLMDispatcher(
        model_name="deepseek-chat",  # 或 "deepseek-chat"
        api_key_path="api_key/ds-api_key.txt",  # 根据实际路径调整
        retries=3
    )

    # 初始化数据集生成器
    generator = DatasetGenerator(
        llm_dispatcher=llm,
        prompt_template_path="prompt/lsat_prompt.txt"
    )

    # 生成数据集
    generator.generate_dataset(
        num_samples=1,
        output_path="output/lsat_dataset.jsonl",
        max_depth_range=(2, 4)
    )


if __name__ == "__main__":
    main()