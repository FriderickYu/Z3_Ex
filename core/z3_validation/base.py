from abc import ABC, abstractmethod

class BaseValidator(ABC):
    """验证器基类，定义统一接口"""

    @abstractmethod
    def check(self, expression: str) -> bool:
        """检查表达式是否满足条件"""
        pass

    @abstractmethod
    def validate(self, data: dict) -> dict:
        """对给定结构化数据进行完整验证"""
        pass