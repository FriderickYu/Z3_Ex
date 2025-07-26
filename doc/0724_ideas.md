## 现在项目存在的问题

1. `LogicChainBuilder.build_dag()`方法中的`depth`参数实际上控制的是**规则数量**，而不是真正的推理深度
2. DAG的构建，是随机选择已有结论作为输入，这将导致：
   1. 大量短路径(depth = 1 or 2)
   2. 缺乏深层嵌套结构
   3. 无法保证指定的推理深度
3. `max_branching`限制了变量复用次数，但没有直接控制推理树的分支数量
4. 没有验证生成的DAG是否真正达到了指定的推理深度
5. 规则组合过于简单，当前`build_dag`方法本质上

```python
# 随机选规则 → 随机选变量 → 应用规则
for _ in range(depth):
    rule_cls = self.rule_pool.sample_rule()
    variables = selected_inputs + new_vars  # 简单拼接
    rule = rule_cls(*variables)
```

将导致

* 扁平化：大部分规则都直接作用于源变量
* 短路径：缺乏深层嵌套的复合表达式
* 随机性过强：无法控制推理的方向和深度

6. 未充分利用Z3PY的特点，而Z3PY的优势在于：

   1. 表达式组合：`And(P, Q), Or(P, Q), Implies(P, Q)`	
   2. 嵌套结构：`And(Implies(P, Q), Or(R, S))`
   3. 约束求解：可以验证推理的有效性

   

## 解决思路

采用表达式树建构法+Z3表达式复杂度驱动构建，去解决DAG复杂度和Z3PY的问题

复杂度驱动 **complexity-driven**

将z3表达式按照逻辑复杂度分层，从简单到复杂逐步构建推理链

复杂度定义

```
# 复杂度层级：
complexity_1: P, Q, R                    # 原子变量
complexity_2: And(P, Q), Or(P, R)       # 简单组合
complexity_3: And(P, Or(Q, R))          # 嵌套组合  
complexity_4: Implies(And(P, Q), Or(R, S))  # 深度嵌套
```

构建过程示例

```python
# Step 1: 从复杂度1开始
available = ["P", "Q", "R", "S"]  # complexity = 1

# Step 2: 构建复杂度2的表达式
rule1 = ConjunctionIntroduction("P", "Q")  # → "And(P, Q)"
available.append("And(P, Q)")  # complexity = 2

# Step 3: 构建复杂度3的表达式  
rule2 = DisjunctionIntroduction("R", "S")  # → "Or(R, S)"
rule3 = ConjunctionIntroduction("And(P, Q)", "Or(R, S)")  # → "And(And(P, Q), Or(R, S))"
available.append("And(And(P, Q), Or(R, S))")  # complexity = 3

# 继续这个过程直到达到目标深度...
```

表达式树驱动

将推理链建模为二叉树结构，其中每个内部节点代表一个逻辑操作，叶子节点是原子变量

```
				And(...)                    # 根节点 (depth=3)
               /        \
        Implies(...)    Or(...)            # 内部节点 (depth=2)  
         /       \      /       \
       P         Q    R         S          # 叶子节点 (depth=1)
```

```python
def build_expression_tree(depth=3):
    if depth == 1:
        return create_atomic_variable()  # 返回 "P", "Q", "R"...
    
    # 递归构建左右子树
    left_subtree = build_expression_tree(depth - 1)
    right_subtree = build_expression_tree(depth - 1)
    
    # 选择逻辑连接符
    connector = random.choice(["And", "Or", "Implies"])
    
    if connector == "And":
        return ConjunctionIntroduction(left_subtree, right_subtree)
    elif connector == "Or":
        return DisjunctionIntroduction(left_subtree, right_subtree)
    else:  # Implies
        return ModusPonens(left_subtree, right_subtree)

# 结果：自动生成depth=3的推理树
```





