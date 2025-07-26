# logger_utils.py

import logging
import os
from datetime import datetime


def setup_logger(name: str = "sample_logger", level=logging.DEBUG) -> logging.Logger:
    """
    初始化并返回一个控制台输出日志记录器。
    :param name: 日志记录器名称
    :param level: 日志等级，默认 DEBUG
    :return: 配置好的 logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 handler
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S")

    # 控制台输出 handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger