from abc import ABC, abstractmethod
from typing import List, Dict
from z3 import Solver


# =============================
# Rule 基类与示例子类
# =============================


class Rule(ABC):
    @abstractmethod
    def to_z3(self) -> List[str]:
        """输出 z3 表达式字符串（用于 LLM prompt）"""
        pass

    @abstractmethod
    def get_main_z3_expr(self) -> str:
        """返回主 z3 表达式（用于逻辑边）"""
        pass

    @abstractmethod
    def get_conclusion_expr(self) -> str:
        """返回结论表达式（用于检查目标）"""
        pass

    @abstractmethod
    def apply_z3(self, solver: Solver, symbols: dict):
        """添加 z3 约束到 solver"""
        pass

    @abstractmethod
    def get_symbol_names(self) -> List[str]:
        """返回用于 prompt 的变量名"""
        pass

    @staticmethod
    @abstractmethod
    def required_vars() -> int:
        """返回规则所需的变量个数"""
        pass

    @abstractmethod
    def describe(self) -> str:
        """返回规则的自然语言表达"""
        pass

    @abstractmethod
    def get_short_label(self) -> str:
        """边的可视化表达"""
        pass
    @abstractmethod
    def get_descriptions(self) -> List[Dict[str, str]]:
        """
        返回 {变量名: 自然语言描述}，用于 prompt 中的变量绑定提示。
        子类必须实现。
        """
        raise NotImplementedError("Subclasses must implement get_descriptions().")

    @abstractmethod
    def get_conclusion_var(self):
        pass