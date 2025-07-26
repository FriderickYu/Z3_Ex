import json
import os
import uuid
from z3 import Bool, Solver, sat
from api_key.llm_dispatcher import LLMDispatcher
from rules.rules_pooling import RulesPooling
from core.logic_chain_builder import LogicChainBuilder
from core.var_binding_builder import build_var_binding_string
from utils.logger_utils import setup_logger
from utils.verification_utils import verify_leaf_reachability
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn



logger = setup_logger("sample_generator")

with open('prompt/lsat_prompt.txt', 'r', encoding='utf-8') as f:
    PROMPT = f.read()


class SampleBuilder:
    def __init__(self, dispatcher: LLMDispatcher, num: int = 5, depth: int = 3, max_branching: int = 2, visualize: bool = True):
        self.dispatcher = dispatcher
        self.num = num
        self.depth = depth
        self.visualize = visualize
        self.rule_pool = RulesPooling()
        self.chain_builder = LogicChainBuilder(self.rule_pool, max_branching=max_branching)

    def generate(self, out_path='output/samples_45.jsonl'):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        count, attempt = 0, 0
        max_attempts = self.num * 5
        logger.info(f"[Start] Target={self.num}, Depth={self.depth}, Visualize={self.visualize}")

        with open(out_path, 'w', encoding='utf-8') as fout, \
                Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                         BarColumn(), TextColumn("{task.completed}/{task.total}"),
                         TimeElapsedColumn()) as progress:

            task = progress.add_task("[cyan]Generating samples...", total=self.num)

            while count < self.num and attempt < max_attempts:
                attempt += 1
                logger.info(f"Attempt {attempt}/{max_attempts}, Generating sample {count + 1}/{self.num}")

                try:
                    rules, dag_info = self.chain_builder.build_dag(depth=self.depth)
                except Exception as e:
                    logger.warning(f"[Build DAG] Failed: {e}")
                    continue

                all_vars = set()
                for rule in rules:
                    all_vars.update(rule.get_symbol_names())
                symbols = {name: Bool(name) for name in all_vars}

                solver = Solver()
                for rule in rules:
                    rule.apply_z3(solver, symbols)

                if dag_info["source_nodes"]:
                    solver.add(symbols[dag_info["source_nodes"][0]])
                else:
                    logger.warning("[Z3] No source nodes, skipping.")
                    continue

                if solver.check() != sat:
                    logger.warning("[Z3] Unsat, skipping.")
                    continue

                z3_exprs, logical_steps, conclusions = [], [], []
                var_descriptions = []
                for rule in rules:
                    z3_exprs.extend(rule.to_z3())
                    logical_steps.append({
                        "rule": rule.get_main_z3_expr(),
                        "description": rule.describe()
                    })
                    conclusions.append(rule.get_conclusion_expr())
                    try:
                        descs = rule.get_descriptions()
                        if isinstance(descs, list) and all(isinstance(d, dict) for d in descs):
                            var_descriptions.extend(descs)
                        else:
                            logger.warning(f"[Format] Invalid get_descriptions(): {descs}")
                    except Exception as e:
                        logger.warning(f"[Desc] Failed to get_descriptions: {e}")

                leaf_conclusions = dag_info["leaf_nodes"]
                intermediate_nodes = dag_info["intermediate_nodes"]
                source_nodes = dag_info["source_nodes"]

                logger.info(f"Leaf conclusions: {leaf_conclusions}")
                logger.info(f"Intermediate nodes: {intermediate_nodes}")
                logger.info(f"Source nodes: {source_nodes}")

                # === Verify reachable leaves ===
                reachable_leafs = verify_leaf_reachability(leaf_conclusions, rules, symbols)
                if not reachable_leafs:
                    logger.warning("[Verify] No reachable leaf nodes found, skipping.")
                    continue

                logical_steps_str = '\n'.join([
                    f"- {step['rule']}: {step['description']}" for step in logical_steps
                ])

                var_binding_str = build_var_binding_string(var_descriptions)
                logger.info(f"VarBinding:\n{var_binding_str}")

                prompt = PROMPT.replace('{z3_exprs}', '\n'.join(z3_exprs)) \
                    .replace('{logical_steps}', logical_steps_str) \
                    .replace('{var_bindings}', var_binding_str)

                resp = self.dispatcher.call(prompt)
                if not resp:
                    logger.warning("Empty response from LLM, skipping.")
                    continue

                try:
                    first, last = resp.find('{'), resp.rfind('}')
                    json_str = resp[first:last + 1]
                    data = json.loads(json_str)
                except Exception as e:
                    logger.warning(f"[Parse] JSON error: {e}")
                    continue

                sample = {
                    'context': data.get('context'),
                    'question': data.get('question'),
                    'answers': data.get('answers'),
                    'label': data.get('label'),
                    'z3': data.get('z3'),
                    'logical_steps': logical_steps,
                    'conclusions': conclusions,
                    'leaf_conclusions': reachable_leafs,
                    'intermediate_nodes': intermediate_nodes,
                    'source_nodes': source_nodes,
                    'id': str(uuid.uuid4())
                }
                fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
                count += 1
                logger.info(f"[+] Sample {count}/{self.num} written: {sample['question']}")
                progress.update(task, advance=1)

                if self.visualize:
                    try:
                        self.chain_builder.visualize(rules, sample_id=sample['id'])
                    except Exception as e:
                        logger.warning(f"[Visualize] Failed to draw graph for sample_{count}: {e}")

        if count < self.num:
            logger.warning(f"Stopped after {attempt} attempts, only generated {count}/{self.num} samples.")
        else:
            logger.info("All samples generated successfully.")


if __name__ == '__main__':
    dispatcher = LLMDispatcher(model_name='deepseek-chat', api_key_path='api_key/ds-api_key.txt')
    SampleBuilder(dispatcher, num=1, depth=10, visualize=True, max_branching=5).generate()