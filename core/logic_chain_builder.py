import os
import random
from typing import List, Tuple, Dict
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm

from rules.rule import Rule
from rules.rules_pooling import RulesPooling
from utils.logger_utils import setup_logger

logger = setup_logger("logic_chain_builder")


class LogicChainBuilder:
    def __init__(self, rule_pool: RulesPooling, max_branching: int = 2):
        self.rule_pool = rule_pool
        self.max_branching = max_branching

    # def build_chain(self, depth: int) -> Tuple[List[Rule], List[str]]:
    #     assert depth >= 1, "深度必须 ≥ 1"
    #
    #     rules: List[Rule] = []
    #     used_vars: set = set()
    #     z3_exprs: List[str] = []
    #
    #     prev_conclusion = None
    #
    #     for i in range(depth):
    #         rule_cls = self.rule_pool.sample_rule()
    #         var_num = rule_cls.required_vars()
    #
    #         if i == 0:
    #             variables = self.rule_pool.sample_vars(var_num)
    #         else:
    #             assert prev_conclusion is not None
    #             extra_vars = self.rule_pool.sample_vars(var_num - 1, exclude={prev_conclusion})
    #             variables = [prev_conclusion] + extra_vars
    #             random.shuffle(variables)
    #
    #         rule = rule_cls(*variables)  # type: ignore
    #         rules.append(rule)
    #
    #         for v in variables:
    #             used_vars.add(v)
    #
    #         prev_conclusion = rule.get_conclusion_expr()
    #         z3_exprs.extend(rule.to_z3())
    #
    #     logger.info(f"[LogicChain] Built linear logic chain with {depth} rules")
    #     return rules, z3_exprs

    def build_dag(self, depth: int) -> Tuple[List[Rule], Dict[str, List[str]]]:
        """
        构建一个包含多分支和合流结构的推理 DAG。

        :param depth: 控制规则数量（即 DAG 中的边数）。
        :return: (规则列表, 图结构信息)
        """
        assert depth >= 1, "深度必须 ≥ 1"

        rules: List[Rule] = []
        used_vars: set = set()
        conclusion_vars: List[str] = []  # 被作为结论输出的变量
        usage_count: Dict[str, int] = {}  # 变量作为前提使用次数

        for _ in range(depth):
            rule_cls = self.rule_pool.sample_rule()
            var_num = rule_cls.required_vars()

            # 从已有结论中选部分变量作为输入，限制 max_branching 次数
            candidate_inputs = [
                v for v in conclusion_vars if usage_count.get(v, 0) < self.max_branching
            ]
            selected_inputs = []

            for v in random.sample(candidate_inputs, len(candidate_inputs)):
                if len(selected_inputs) >= var_num - 1:
                    break
                selected_inputs.append(v)
                usage_count[v] = usage_count.get(v, 0) + 1

            need_new = var_num - len(selected_inputs)
            new_vars = self.rule_pool.sample_vars(need_new, exclude=used_vars)
            variables = selected_inputs + new_vars
            random.shuffle(variables)

            rule = rule_cls(*variables)  # type: ignore
            rules.append(rule)

            # 更新变量记录
            for v in variables:
                used_vars.add(v)

            conclusion_var = rule.get_conclusion_expr()
            conclusion_vars.append(conclusion_var)
            used_vars.add(conclusion_var)

        dag_info = self.build_dag_from_rules(rules)
        logger.info(f"[LogicDAG] DAG built with {len(rules)} rules, "
                    f"{len(dag_info['source_nodes'])} sources, "
                    f"{len(dag_info['leaf_nodes'])} leaf nodes.")
        return rules, dag_info

    def compute_graph_depth(self, graph: Dict[str, List[str]], leaf_nodes: List[str]) -> int:
        """
        计算 DAG 的最大深度（最长推理链）。
        :param graph: 变量依赖图 {conclusion: [premises]}
        :param leaf_nodes: 叶子节点列表
        :return: 最大深度（int）
        """
        memo = {}

        def dfs(node: str) -> int:
            if node not in graph or not graph[node]:
                return 1
            if node in memo:
                return memo[node]
            max_depth = 1 + max(dfs(parent) for parent in graph[node])
            memo[node] = max_depth
            return max_depth

        all_depths = [dfs(leaf) for leaf in leaf_nodes]
        max_depth = max(all_depths) if all_depths else 0
        logger.info(f"[Depth] Computed DAG max depth: {max_depth}")
        return max_depth


    def _build_graph_from_rules(self, rules: List[Rule]) -> nx.DiGraph:
        """
        内部函数：从规则列表构建 DAG 图（用于可视化或结构分析）
        """
        G = nx.DiGraph()
        max_label_len = 35

        for i, rule in enumerate(rules):
            try:
                symbol_names = rule.get_symbol_names()
                if not symbol_names:
                    continue

                output_var = rule.get_conclusion_expr()
                input_vars = [v for v in symbol_names if v != output_var]
                label = f"{rule.get_short_label()}_{i}"

                output_var_short = output_var[:max_label_len] + "..." if len(output_var) > max_label_len else output_var
                input_vars_short = [
                    v[:max_label_len] + "..." if len(v) > max_label_len else v
                    for v in input_vars
                ]

                for inp in input_vars_short:
                    G.add_edge(inp, output_var_short, label=label)
            except Exception as e:
                logger.warning(f"[DAG Build] Rule {type(rule).__name__} failed: {e}")
                continue

        return G

    def build_dag_from_rules(self, rules: List[Rule]) -> Dict[str, List[str]]:
        """
        构建 DAG 并提取各类节点。
        """
        G = self._build_graph_from_rules(rules)

        all_nodes = list(G.nodes)
        leaf_nodes = [n for n in all_nodes if G.out_degree(n) == 0]
        source_nodes = [n for n in all_nodes if G.in_degree(n) == 0]
        intermediate_nodes = [
            n for n in all_nodes if G.in_degree(n) > 0 and G.out_degree(n) > 0
        ]

        return {
            "leaf_nodes": leaf_nodes,
            "intermediate_nodes": intermediate_nodes,
            "source_nodes": source_nodes
        }

    def visualize(self, rules: List[Rule], sample_id: str, out_dir: str = "images/graphs/"):
        """
        根据规则列表生成逻辑图可视化。
        """
        os.makedirs(out_dir, exist_ok=True)
        G = self._build_graph_from_rules(rules)

        # 计算节点深度（用于着色）
        depth_map = {}
        for node in nx.topological_sort(G):
            preds = list(G.predecessors(node))
            depth_map[node] = 0 if not preds else max(depth_map[p] for p in preds) + 1

        max_depth = max(depth_map.values()) if depth_map else 1
        cmap = cm.get_cmap('Blues', max_depth + 1)
        node_colors = [cmap(depth_map.get(n, 0)) for n in G.nodes()]

        # 检测叶子节点（无出边）
        leaves = [n for n in G.nodes if G.out_degree(n) == 0]

        # 布局
        try:
            pos = nx.nx_pydot.graphviz_layout(G, prog="neato")
        except Exception:
            pos = nx.spring_layout(G, k=1.5, seed=42, iterations=100)

        plt.figure(figsize=(15, 10))
        node_color_final = [
            "#FDAE61" if n in leaves else node_colors[i]
            for i, n in enumerate(G.nodes)
        ]

        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_color_final,
            node_size=2000,
            edgecolors='k',
            linewidths=1.2
        )

        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', verticalalignment='center', horizontalalignment='center')

        nx.draw_networkx_edges(
            G,
            pos,
            edge_color='black',
            arrows=True,
            arrowsize=30,
            arrowstyle='-|>',
            node_size=2000,
            width=2,
            min_source_margin=15,
            min_target_margin=15
        )

        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=10,
            rotate=False,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
        )

        plt.title(f"Logic DAG for Sample {sample_id}", fontsize=14, pad=20)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{sample_id}.png"), dpi=300)
        plt.savefig(os.path.join(out_dir, f"{sample_id}.svg"), format="svg")
        plt.close()