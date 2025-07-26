"""
健壮的DAG HTML可视化器
使用安全的字符串处理，避免转义BUG
"""

import networkx as nx
from typing import Dict, List, Any, Tuple
from pathlib import Path
from datetime import datetime
import html
import json


class RobustDAGVisualizer:
    """
    健壮的DAG可视化器
    使用JSON数据传递和安全的字符串处理，避免HTML转义问题
    """

    def __init__(self):
        self.dags = []

    def add_dag(self, dag: nx.DiGraph, title: str, description: str = ""):
        """添加DAG到可视化集合中"""
        dag_data = self._extract_dag_data(dag)
        self.dags.append({
            'data': dag_data,
            'title': html.escape(title),
            'description': html.escape(description)
        })

    def _extract_dag_data(self, dag: nx.DiGraph) -> Dict[str, Any]:
        """提取DAG数据为JSON安全格式"""
        nodes = []
        edges = []
        levels = {}

        # 提取节点数据
        for node_id, data in dag.nodes(data=True):
            node_info = {
                'id': str(node_id),
                'label': '',
                'level': 0,
                'type': 'intermediate',
                'expression': '',
                'rule': ''
            }

            if 'data' in data and hasattr(data['data'], 'formula'):
                node_data = data['data']
                formula = node_data.formula

                node_info.update({
                    'label': str(getattr(formula, 'expression', node_id))[:15],
                    'expression': str(getattr(formula, 'expression', node_id)),
                    'level': getattr(node_data, 'level', 0),
                    'rule': str(getattr(node_data, 'applied_rule', '')),
                    'type': self._get_node_type(node_data)
                })
            else:
                node_info['label'] = str(node_id)[:15]
                node_info['expression'] = str(node_id)

            nodes.append(node_info)

            # 构建层级映射
            level = node_info['level']
            if level not in levels:
                levels[level] = []
            levels[level].append(node_info)

        # 提取边数据
        for source, target, edge_data in dag.edges(data=True):
            edge_info = {
                'source': str(source),
                'target': str(target),
                'rule': ''
            }

            if 'data' in edge_data and hasattr(edge_data['data'], 'rule_id'):
                edge_info['rule'] = str(edge_data['data'].rule_id)

            edges.append(edge_info)

        return {
            'nodes': nodes,
            'edges': edges,
            'levels': levels,
            'stats': {
                'node_count': len(nodes),
                'edge_count': len(edges),
                'max_level': max(levels.keys()) if levels else 0
            }
        }

    def _get_node_type(self, node_data) -> str:
        """获取节点类型"""
        if getattr(node_data, 'is_premise', False):
            return 'premise'
        elif getattr(node_data, 'is_conclusion', False):
            return 'conclusion'
        else:
            return 'intermediate'

    def generate_html(self, output_file: str = "dag_unified_visualization.html") -> str:
        """生成HTML文件"""
        if not self.dags:
            raise ValueError("没有DAG可供可视化")

        # 计算总体统计
        total_stats = self._calculate_total_stats()

        # 准备数据
        dags_json = json.dumps(self.dags, ensure_ascii=False, indent=2)

        # 生成HTML
        html_content = self._generate_html_template(total_stats, dags_json)

        # 保存文件
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(output_path.absolute())

    def _calculate_total_stats(self) -> Dict[str, int]:
        """计算总体统计"""
        total_nodes = sum(dag['data']['stats']['node_count'] for dag in self.dags)
        total_edges = sum(dag['data']['stats']['edge_count'] for dag in self.dags)
        max_depth = max((dag['data']['stats']['max_level'] for dag in self.dags), default=0)

        return {
            'total_dags': len(self.dags),
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'max_depth': max_depth
        }

    def _generate_html_template(self, stats: Dict[str, int], dags_json: str) -> str:
        """生成HTML模板"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ARNG推理DAG可视化</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        }}
        .main-title {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .summary-card {{
            background: linear-gradient(45deg, #3498db, #2980b9);
            color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        }}
        .summary-number {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .dag-section {{
            margin-bottom: 50px;
            border: 2px solid #e74c3c;
            border-radius: 12px;
            padding: 25px;
            background: white;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        }}
        .dag-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
            flex-wrap: wrap;
            gap: 15px;
        }}
        .dag-title {{
            color: #e74c3c;
            font-size: 1.8em;
            font-weight: 600;
        }}
        .dag-stats {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}
        .stat-badge {{
            background: #34495e;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.9em;
        }}
        .dag-container {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 25px;
            overflow-x: auto;
        }}
        .level {{
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 20px 0;
            flex-wrap: wrap;
            gap: 15px;
        }}
        .level-label {{
            font-size: 14px;
            color: #666;
            font-weight: bold;
            margin-bottom: 15px;
            text-align: center;
            width: 100%;
            padding: 8px;
            background: rgba(52, 73, 94, 0.1);
            border-radius: 20px;
        }}
        .node {{
            display: inline-block;
            padding: 12px 18px;
            border-radius: 25px;
            color: white;
            font-weight: bold;
            text-align: center;
            min-width: 80px;
            max-width: 180px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            font-size: 14px;
            margin: 5px;
        }}
        .node:hover {{
            transform: translateY(-3px) scale(1.05);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }}
        .node.premise {{
            background: linear-gradient(45deg, #27ae60, #2ecc71);
        }}
        .node.intermediate {{
            background: linear-gradient(45deg, #3498db, #5dade2);
        }}
        .node.conclusion {{
            background: linear-gradient(45deg, #e74c3c, #ec7063);
        }}
        .arrows {{
            text-align: center;
            font-size: 20px;
            color: #7f8c8d;
            margin: 10px 0;
        }}
        .rule-info {{
            text-align: center;
            font-size: 13px;
            color: #666;
            font-style: italic;
            margin: 8px 0;
            background: rgba(52, 152, 219, 0.1);
            padding: 6px 12px;
            border-radius: 15px;
            display: inline-block;
        }}
        .tooltip {{
            display: none;
            position: absolute;
            background: #2c3e50;
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            z-index: 1000;
            top: -60px;
            left: 50%;
            transform: translateX(-50%);
            white-space: nowrap;
            max-width: 200px;
        }}
        .node:hover .tooltip {{
            display: block;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 30px 0;
            flex-wrap: wrap;
            padding: 20px;
            background: rgba(236, 240, 241, 0.8);
            border-radius: 12px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 500;
        }}
        .legend-color {{
            width: 24px;
            height: 24px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }}
        .legend-color.premise {{ background: linear-gradient(45deg, #27ae60, #2ecc71); }}
        .legend-color.intermediate {{ background: linear-gradient(45deg, #3498db, #5dade2); }}
        .legend-color.conclusion {{ background: linear-gradient(45deg, #e74c3c, #ec7063); }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 14px;
        }}
        @media (max-width: 768px) {{
            .container {{ padding: 15px; }}
            .dag-header {{ flex-direction: column; }}
            .summary-stats {{ grid-template-columns: 1fr; }}
            .node {{ min-width: 60px; font-size: 12px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="main-title">🌲 ARNG推理DAG可视化</h1>
        
        <div class="summary-stats">
            <div class="summary-card">
                <div class="summary-number">{stats['total_dags']}</div>
                <div>推理模式</div>
            </div>
            <div class="summary-card">
                <div class="summary-number">{stats['total_nodes']}</div>
                <div>总节点数</div>
            </div>
            <div class="summary-card">
                <div class="summary-number">{stats['total_edges']}</div>
                <div>总边数</div>
            </div>
            <div class="summary-card">
                <div class="summary-number">{stats['max_depth']}</div>
                <div>最大深度</div>
            </div>
        </div>

        <div class="legend">
            <div class="legend-item">
                <div class="legend-color premise"></div>
                <span>前提 (Premises)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color intermediate"></div>
                <span>中间结论 (Intermediate)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color conclusion"></div>
                <span>最终结论 (Conclusion)</span>
            </div>
        </div>

        <div id="dag-sections"></div>

        <div class="footer">
            <p>💡 <strong>提示</strong>：将鼠标悬停在节点上查看详细信息</p>
            <p>🕒 生成时间: {timestamp}</p>
        </div>
    </div>

    <script>
        // DAG数据
        const dagsData = {dags_json};
        
        // 安全的文本处理函数
        function safeText(text) {{
            if (!text) return '';
            return String(text).replace(/[<>&"']/g, function(match) {{
                const escapeMap = {{
                    '<': '&lt;',
                    '>': '&gt;',
                    '&': '&amp;',
                    '"': '&quot;',
                    "'": '&#39;'
                }};
                return escapeMap[match];
            }});
        }}
        
        // 生成DAG部分的HTML
        function generateDAGSection(dagInfo) {{
            const data = dagInfo.data;
            const stats = data.stats;
            
            let html = `
                <div class="dag-section">
                    <div class="dag-header">
                        <h2 class="dag-title">${{dagInfo.title}}</h2>
                        <div class="dag-stats">
                            <span class="stat-badge">${{stats.node_count}} 节点</span>
                            <span class="stat-badge">${{stats.edge_count}} 边</span>
                            <span class="stat-badge">深度 ${{stats.max_level}}</span>
                        </div>
                    </div>`;
            
            if (dagInfo.description) {{
                html += `<p style="color: #666; margin-bottom: 20px; font-style: italic;">${{dagInfo.description}}</p>`;
            }}
            
            html += '<div class="dag-container">';
            
            // 按层级组织节点
            const levels = data.levels;
            const sortedLevels = Object.keys(levels).map(Number).sort((a, b) => a - b);
            
            for (let i = 0; i < sortedLevels.length; i++) {{
                const level = sortedLevels[i];
                const nodes = levels[level];
                
                html += `<div class="level-label">层级 ${{level}}</div>`;
                html += '<div class="level">';
                
                for (const node of nodes) {{
                    const tooltipText = `节点: ${{safeText(node.id)}}\\n表达式: ${{safeText(node.expression)}}\\n层级: ${{level}}` + 
                                      (node.rule ? `\\n规则: ${{safeText(node.rule)}}` : '');
                    
                    html += `
                        <div class="node ${{node.type}}">
                            ${{safeText(node.label)}}
                            <div class="tooltip">${{safeText(tooltipText)}}</div>
                        </div>`;
                }}
                
                html += '</div>';
                
                // 添加箭头和规则信息（除最后一层）
                if (i < sortedLevels.length - 1) {{
                    const rules = getRulesBetweenLevels(data.edges, level, sortedLevels[i + 1]);
                    if (rules.length > 0) {{
                        html += '<div class="arrows">↓</div>';
                        html += `<div style="text-align: center;"><span class="rule-info">${{rules.join(' | ')}}</span></div>`;
                    }}
                }}
            }}
            
            html += '</div></div>';
            return html;
        }}
        
        // 获取层级间的规则
        function getRulesBetweenLevels(edges, fromLevel, toLevel) {{
            const rules = new Set();
            // 简化实现：返回所有边的规则
            for (const edge of edges) {{
                if (edge.rule) {{
                    rules.add(safeText(edge.rule));
                }}
            }}
            return Array.from(rules);
        }}
        
        // 渲染所有DAG
        function renderDAGs() {{
            const container = document.getElementById('dag-sections');
            let html = '';
            
            for (const dagInfo of dagsData) {{
                html += generateDAGSection(dagInfo);
            }}
            
            container.innerHTML = html;
        }}
        
        // 页面加载完成后渲染
        document.addEventListener('DOMContentLoaded', renderDAGs);
    </script>
</body>
</html>'''

    def clear(self):
        """清空DAG数据"""
        self.dags.clear()


# 便捷函数
def create_unified_visualization(dag_list: List[Tuple[nx.DiGraph, str, str]],
                                output_file: str = "logs/dag_unified_visualization.html") -> str:
    """
    创建统一的DAG可视化

    Args:
        dag_list: [(dag, title, description), ...] 的列表
        output_file: 输出文件路径

    Returns:
        str: 生成的HTML文件路径
    """
    visualizer = RobustDAGVisualizer()

    for dag, title, description in dag_list:
        visualizer.add_dag(dag, title, description)

    return visualizer.generate_html(output_file)