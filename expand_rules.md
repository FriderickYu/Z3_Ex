# 规则库结构

## 命题逻辑（25个）
### 基础推理（7个）

* ModusTollens（否定后件）：`¬Q, P→Q ⊢ ¬P`
* DisjunctiveSyllogism（析取三段论）：`P∨Q, ¬P ⊢ Q`
* ConjunctionIntroductionRule（合取引入）：`P, Q ⊢ P∧Q` **已存在**
* ConjunctionEliminationRule（合取消除）：`P∧Q ⊢ P` **已存在**
* DisjunctionIntroductionRule（析取引入）：`P ⊢ P∨Q` **已存在**
* BiconditionalEliminationRule（双条件消除）：`P↔Q ⊢ (P→Q)∧(Q→P)` **已存在**
* ConsequentStrengthening（后件强化）：`P→Q ⊢ P→(P∧Q)`

### 复合推理（5个）
* ConstructiveDilemma：`(P→Q)∧(R→S), P∨R ⊢ Q∨S`
* DestructiveDilemma：`(P→Q)∧(R→S), ¬Q∨¬S ⊢ ¬P∨¬R`
* ResolutionRule（归结）：`P∨Q, ¬P∨R ⊢ Q∨R`
* ModusPonensRule（肯定前件）：`P, P→Q ⊢ Q` **已存在**
* HypotheticalSyllogismRule（假言三段论）：`P→Q, Q→R ⊢ P→R` **已存在**

### 等价与布尔代数基础律（13个）
* DoubleNegation（双重否定）：`¬¬P ⟷ P`
* DeMorganLaws（德摩根）：`¬(P∧Q) ⟷ (¬P∨¬Q)；¬(P∨Q) ⟷ (¬P∧¬Q)`
* DistributiveLaws（分配律）：`P∧(Q∨R) ⟷ (P∧Q)∨(P∧R)；P∨(Q∧R) ⟷ (P∨Q)∧(P∨R)`
* CommutativeLaws（交换律）：`P∧Q ⟷ Q∧P；P∨Q ⟷ Q∨P`
* AssociativeLaws（结合律）：`(P∧Q)∧R ⟷ P∧(Q∧R)；(P∨Q)∨R ⟷ P∨(Q∨R)`
* AbsorptionLaws（吸收律）：`P∨(P∧Q) ⟷ P；P∧(P∨Q) ⟷ P`
* IdempotentLaws（幂等律）：`P∧P ⟷ P；P∨P ⟷ P`
* IdentityLaws（同一律）：`P∧⊤ ⟷ P；P∨⊥ ⟷ P`
* DominationLaws（支配/恒等式）：`P∨⊤ ⟷ ⊤；P∧⊥ ⟷ ⊥`
* NegationLaws（排中/矛盾）：`P∨¬P ⟷ ⊤；P∧¬P ⟷ ⊥`
* MaterialImplication（蕴含等价）：`P→Q ⟷ ¬P∨Q`
* Contraposition（对当等价）：`P→Q ⟷ ¬Q→¬P`
* BiconditionalDecomposition（双条件分解）：`P↔Q ⟷ (P→Q)∧(Q→P)`

## 一阶逻辑扩充（13个）
### 量词操作（4个）
* ExistentialGeneralization（存在泛化）：`P(a) ⊢ ∃x P(x)`
* UniversalInstantiationRule（全称实例化）：`∀x P(x) ⊢ P(a)` **已存在**
* UniversalGeneralization（全称泛化，侧条件）：`P(x) ⊢ ∀x P(x) （侧条件：x 不自由出现在未解除前提中）`
* ExistentialElimination（存在消去，证明规则）： `∃x P(x) ⇒ 取 fresh 常量 c，在子证明中以假设 P(c) 推出目标 φ，且 c 不出现在 φ/任何开放前提 中，则得 ⊢ φ`

### 量词转换（5个）
* QuantifierNegation-1：`¬∀x P(x) ⟷ ∃x ¬P(x)`
* QuantifierNegation-2：`¬∃x P(x) ⟷ ∀x ¬P(x)`
* UniversalDistribution（对∧的分配）：`∀x (P(x)∧Q(x)) ⟷ (∀x P(x)) ∧ (∀x Q(x))`
* ExistentialDistribution（对∨的分配）：`∃x (P(x)∨Q(x)) ⟷ (∃x P(x)) ∨ (∃x Q(x))`
* ExistentialConjunction（一向）：`∃x (P(x)∧Q(x)) ⊢ (∃x P(x)) ∧ (∃x Q(x))`

### 等式推理（4个）
* EqualitySubstitution（等式替换）：`a = b, P(a) ⊢ P(b)`
* ReflexivityRule（自反）：`⊢ a = a`
* SymmetryRule（对称）：`a = b ⊢ b = a`
* TransitivityRule（传递）：`a = b, b = c ⊢ a = c` **已存在**

## 条件逻辑（6个）
### 蕴含类（4个）
* ExportationRule（外延）：`P→(Q→R) ⊢ (P∧Q)→R`
* ImportationRule（内延）：`(P∧Q)→R ⊢ P→(Q→R)`
* ContrapositionRule：`P→Q ⊢ ¬Q→¬P`
* MaterialImplication：`P→Q ⊢ ¬P∨Q`

### 双条件类（2个）
* BiconditionalIntroduction：`P→Q, Q→P ⊢ P↔Q`
* BiconditionalTransitivity：`P↔Q, Q↔R ⊢ P↔R`

## 结构规则（4个｜证明控制层）
* CutRule（割）：`P⊢Q, Q⊢R ⟹ P⊢R`
* WeakeningRule（弱化）：`P⊢Q ⟹ P∧R⊢Q`
* ContractionRule（收缩）：`P∧P⊢Q ⟹ P⊢Q`
* ExchangeRule（交换）：`P∧Q⊢R ⟹ Q∧P⊢R`


## 归约推理（3个）
* NegationIntroduction（否定引入）：`Γ ∪ {P} ⊢ ⊥ ⟹ Γ ⊢ ¬P`
* ProofByContradiction / RAA（反证法）：`Γ ∪ {¬P} ⊢ ⊥ ⟹ Γ ⊢ P`
* Explosion / EFQ（爆炸律，可选）：`Γ ⊢ ⊥ ⟹ Γ ⊢ φ`

## 算术逻辑（24个｜ℤ/ℝ 指明域，带必要侧条件）
### 代数基础律（10个）
* AddComm（加法交换）：`a + b = b + a`
* MulComm（乘法交换）：`a × b = b × a`
* AddAssoc（加法结合）：`(a + b) + c = a + (b + c)`
* MulAssoc（乘法结合）：`(a × b) × c = a × (b × c)`
* LeftDistributive（左分配）：`a × (b + c) = a×b + a×c`
* RightDistributive（右分配）：`(a + b) × c = a×c + b×c`
* AddIdentity（加法恒等元）：`a + 0 = a`
* MulIdentity（乘法恒等元）：`a × 1 = a`
* MulZero（乘零归零）：`a × 0 = 0`
* AddInverse（加法逆元）：`a + (−a) = 0`

### 整数/实数公共推理（不等式与等式）（5个）
* ArithmeticTransitivity（< 传递）：`a < b, b < c ⊢ a < c`
* AdditionPreservesEq（加法保等式）：`a = b ⊢ a + c = b + c`
* MultiplicationPreservesEq（乘法保等式）：`a = b ⊢ a × c = b × c（无需 c ≠ 0）`
* Antisymmetry（反对称）：`a ≤ b, b ≤ a ⊢ a = b`
* LinearInequality（线性不等式）：`a + b ≤ c + d, a ≥ c ⊢ b ≤ d`

### 不等式运算规则（4个）
* AddMonotone：`a < b ⊢ a + c < b + c`
* MulMonotonePos：`a < b, c > 0 ⊢ a×c < b×c`
* MulMonotoneNeg：`a < b, c < 0 ⊢ a×c > b×c`
* SumPositives：`x > 0, y > 0 ⊢ x + y > 0`

### 除法（2个）
* RealDivision（实数）：`a = b × c, c ≠ 0 ⊢ a / c = b`
* IntDivision（整数，SMT-LIB 欧几里得除法）：`a = b × c, c ≠ 0 ⊢ (a div c) = b`

### 平方与非负（1个）
* SquareNonnegativity（平方非负，ℝ）：`x ∈ ℝ ⊢ x² ≥ 0`

### 模运算（2个，侧条件：n > 0）
* ModularAddition（加法同余）：`a ≡ b (mod n), c ≡ d (mod n) ⊢ a + c ≡ b + d (mod n)`
* ModularTransitivity（同余传递）：`a ≡ b (mod n), b ≡ c (mod n) ⊢ a ≡ c (mod n)`