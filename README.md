
## 项目架构

```text
ARNG_Generator/
├── core/
│   ├── dag_generator.py           # 双层DAG生成器
│   ├── complexity_controller.py   # 复杂度控制器
│   └── z3_validator.py           # Z3验证器
│   └── expression_tree_builder.py # 表达式树构建器
├── rules/
│   ├── base/
│   │   └── rule.py               # 规则基类
│   ├── tiers/
│   │   ├── tier1_axioms.py       # 基础公理
│   │   ├── tier2_basic.py        # 基础推理
│   │   ├── tier3_compound.py     # 复合规则
│   │   ├── tier4_quantifier.py   # 量词逻辑
│   │   └── tier5_arithmetic.py   # 算术逻辑
│   └── rule_pool.py              # 分层规则池
├── distractors/
│   ├── structural_distractor.py  # 结构扰动
│   ├── semantic_distractor.py    # 语义误导
│   └── z3_distractor.py         # Z3驱动干扰项
├── generation/
│   ├── sample_generator.py       # 样本生成器
│   ├── qa_generator.py          # 问答生成器
│   └── llm_interface.py         # LLM接口
├── visualization/
│   ├── dag_visualizer.py        # DAG可视化
│   ├── step_visualizer.py       # 步骤可视化
│   └── debug_dashboard.py       # 调试面板
├── utils/
│   ├── z3_utils.py              # Z3工具函数
│   ├── validation_utils.py      # 验证工具
│   └── logger_utils.py          # 日志工具
└── tests/
    ├── test_rules.py            # 规则测试
    ├── test_dag_generation.py   # DAG生成测试
    └── test_integration.py      # 集成测试
```


### day1_demo.py
完整的Day 1功能演示脚本，包含：
- 系统初始化演示
- DAG生成演示
- 复杂度控制演示
- 不同生成模式对比
- 规则选择策略对比
- 复杂度递增演示
- 统计分析演示

### quick_test.py
快速测试脚本，用于：
- 验证系统是否正常安装
- 快速检查核心功能
- 开发时的快速验证

## 使用方法

```bash
# 运行完整演示
cd ARNG_Generator_v2
python examples/day1_demo.py

# 运行快速测试
python examples/quick_test.py