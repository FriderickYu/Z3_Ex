# 归约推理

"""
* NegationIntroduction（否定引入）：`Γ ∪ {P} ⊢ ⊥ ⟹ Γ ⊢ ¬P`
* ProofByContradiction / RAA（反证法）：`Γ ∪ {¬P} ⊢ ⊥ ⟹ Γ ⊢ P`
* Explosion / EFQ（爆炸律，可选）：`Γ ⊢ ⊥ ⟹ Γ ⊢ φ`
"""