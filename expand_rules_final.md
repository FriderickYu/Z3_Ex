新规则库

## 命题逻辑基础规则（10个）

1. ModusPonensRule - 肯定前件
* 符号表达: `P, P→Q ⊢ Q`
* 选择理由:
  * DAG构建：生成简单的两前提单结论结构，是最基础的推理模式
  * 代码实现：已存在完整实现，无需修改

2. HypotheticalSyllogismRule - 假言三段论
* 符号表达: `P→Q, Q→R ⊢ P→R`
* 选择理由:
  * DAG构建：自然形成链式推理结构，适合构建深层DAG
  * 代码实现：已实现，处理蕴含的传递性，代码成熟

3. ConjunctionIntroductionRule - 合取引入
* 符号表达: `P, Q ⊢ P∧Q`
* 选择理由:
  * DAG构建：可接受2-4个前提，生成多分支汇聚的DAG节点
  * 代码实现：已实现，支持可变参数，灵活性高

4. ConjunctionEliminationRule - 合取消除
* 符号表达: `P∧Q ⊢ P` 或 `P∧Q ⊢ Q`
* 选择理由:
  * DAG构建：单前提规则，可作为推理链的中间步骤
  * 代码实现：已实现，随机选择消除项，增加变化性

5. DisjunctionIntroductionRule - 析取引入
* 符号表达: `P ⊢ P∨Q`
* 选择理由:
  * DAG构建：弱化推理，适合作为推理链的扩展步骤
  * 代码实现：已实现，自动生成附加项Q

6. BiconditionalEliminationRule - 双条件消除
* 符号表达: `P↔Q ⊢ (P→Q)∧(Q→P)`
* 选择理由:
  * DAG构建：将双条件分解为两个蕴含，增加DAG复杂度
  * 代码实现：已实现，生成复合结构

7. ModusTollens - 否定后件
* 符号表达: `¬Q, P→Q ⊢ ¬P`
* 选择理由:
  * DAG构建：与ModusPonens对称，提供否定推理路径
  * 代码实现：简单的两前提规则，只需处理否定和蕴含

8. DisjunctiveSyllogism - 析取三段论
* 符号表达: `P∨Q, ¬P ⊢ Q`
* 选择理由:
  * DAG构建：处理析取和否定，增加逻辑多样性
  * 代码实现：结构清晰，易于实现can_apply检查

9. ConstructiveDilemma - 建构两难
* 符号表达: `(P→Q)∧(R→S), P∨R ⊢ Q∨S`
* 选择理由:
  * DAG构建：复合前提生成复合结论，创建丰富的分支结构
  * 代码实现：虽复杂但模式固定，可分解为子操作

10. DestructiveDilemma - 破坏两难
* 符号表达: `(P→Q)∧(R→S), ¬Q∨¬S ⊢ ¬P∨¬R`
* 选择理由:
  * DAG构建：与ConstructiveDilemma对称，提供否定推理变体
  * 代码实现：模式与ConstructiveDilemma相似，复用代码结构

## 等价规则（4个）

11. DoubleNegation - 双重否定律
* 符号表达: `¬¬P ⟷ P`
* 选择理由:
  * DAG构建：单前提单结论，简单的转换节点
  * 代码实现：只需检查和处理两层否定，实现trivial

12. DeMorganLaws - 德摩根律
* 符号表达: `¬(P∧Q) ⟷ (¬P∨¬Q) 和 ¬(P∨Q) ⟷ (¬P∧¬Q)`
* 选择理由:
  * DAG构建：在合取和析取之间转换，增加表达式变化
  * 代码实现：模式匹配清晰，z3提供is_and/is_or/is_not判断

13. MaterialImplication - 材料蕴含
* 符号表达: `P→Q ⟷ ¬P∨Q`
* 选择理由:
  * DAG构建：蕴含与析取互转，连接不同逻辑形式
  * 代码实现：简单的表达式转换，两种形式互相generate

14. Contraposition - 逆否等价
* 符号表达: `P→Q ⟷ ¬Q→¬P`
* 选择理由:
  * DAG构建：生成等价但形式不同的蕴含，丰富推理路径
  * 代码实现：交换和否定操作，与ModusTollens配合良好

## 布尔代数律（9个）
15. CommutativeLaws - 交换律
* 符号表达: `P∧Q ⟷ Q∧P 和 P∨Q ⟷ Q∨P`
* 选择理由:
  * DAG构建：调整子表达式顺序，不改变逻辑结构
  * 代码实现：简单的children重排，使用random.shuffle

16. AssociativeLaws - 结合律
* 符号表达: `(P∧Q)∧R ⟷ P∧(Q∧R) 和 (P∨Q)∨R ⟷ P∨(Q∨R)`
* 选择理由:
  * DAG构建：重组嵌套结构，支持更深的DAG
  * 代码实现：递归flatten后重新组合，处理多项运算

17. DistributiveLaws - 分配律
* 符号表达: `P∧(Q∨R) ⟷ (P∧Q)∨(P∧R) 和 P∨(Q∧R) ⟷ (P∨Q)∧(P∨R)`
* 选择理由:
  * DAG构建：在合取和析取间分配，创建复杂表达式
  * 代码实现：展开或因式分解，增加表达式复杂度

18. AbsorptionLaws - 吸收律
* 符号表达: `P∨(P∧Q) ⟷ P 和 P∧(P∨Q) ⟷ P`
* 选择理由:
  * DAG构建：简化复杂表达式，控制DAG大小
  * 代码实现：识别公共子项，实现表达式化简

19. IdempotentLaws - 幂等律
* 符号表达: `P∧P ⟷ P 和 P∨P ⟷ P`
* 选择理由:
  * DAG构建：去除重复项，简化推理链
  * 代码实现：检测重复的children，简单去重

20. IdentityLaws - 同一律
* 符号表达: `P∧⊤ ⟷ P 和 P∨⊥ ⟷ P`
* 选择理由:
  * DAG构建：处理常量True/False，边界情况
  * 代码实现：使用z3.BoolVal(True/False)，过滤常量

21. DominationLaws - 支配律
* 符号表达: `P∨⊤ ⟷ ⊤ 和 P∧⊥ ⟷ ⊥`
* 选择理由:
  * DAG构建：生成常量结论，终止推理链
  * 代码实现：检测常量存在，直接返回常量

22. NegationLaws - 否定律（排中律/矛盾律）
* 符号表达: `P∨¬P ⟷ ⊤ 和 P∧¬P ⟷ ⊥`
* 选择理由:
  * DAG构建：生成恒真/恒假表达式，逻辑边界
  * 代码实现：检测互补对，返回布尔常量

23. ResolutionRule - 归结规则
* 符号表达: `P∨Q, ¬P∨R ⊢ Q∨R`
* 选择理由:
  * DAG构建：消除互补文字，经典的归结推理
  * 代码实现：识别互补对（P和¬P），合并剩余项