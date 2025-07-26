"""
测试模块
包含单元测试、集成测试和性能测试
"""

# 测试套件
import unittest
from .test_rules import TestTier1Rules, TestStratifiedRulePool
from .test_dag_generation import TestDAGGeneration, TestComplexityController
from .test_integration import TestIntegration


def create_test_suite():
    """创建测试套件"""
    suite = unittest.TestSuite()

    # 添加单元测试
    suite.addTest(unittest.makeSuite(TestTier1Rules))
    suite.addTest(unittest.makeSuite(TestStratifiedRulePool))
    suite.addTest(unittest.makeSuite(TestDAGGeneration))
    suite.addTest(unittest.makeSuite(TestComplexityController))

    # 添加集成测试
    suite.addTest(unittest.makeSuite(TestIntegration))

    return suite


def run_all_tests():
    """运行所有测试"""
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    run_all_tests()