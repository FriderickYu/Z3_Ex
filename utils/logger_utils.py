import os
import logging
import sys
from datetime import datetime
from pathlib import Path
import locale
import codecs


class ARNGLogger:
    """
    支持中文字符的高级日志系统
    所有实例共享同一个日志文件，避免创建多个日志文件
    """

    # 类级别的共享属性
    _shared_log_file = None
    _shared_file_handler = None
    _initialized = False

    def __init__(self, name: str, base_dir: str = "logs"):
        self.name = name
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

        # 设置编码支持中文
        self._setup_encoding()

        # 初始化共享日志文件（只在第一次创建时）
        if not ARNGLogger._initialized:
            self._initialize_shared_log()
            ARNGLogger._initialized = True

        # 配置当前实例的日志器
        self.logger = self._setup_logger()

    def _setup_encoding(self):
        """设置编码以支持中文字符"""
        try:
            locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
        except locale.Error:
            try:
                locale.setlocale(locale.LC_ALL, 'C.UTF-8')
            except locale.Error:
                pass  # 使用默认编码

    def _initialize_shared_log(self):
        """初始化共享的日志文件"""
        # 创建统一的日志文件名（按日期）
        date_stamp = datetime.now().strftime("%Y%m%d%H%M")
        log_filename = f"ARNG_{date_stamp}.log"
        ARNGLogger._shared_log_file = self.base_dir / log_filename

        # 创建共享的文件处理器
        ARNGLogger._shared_file_handler = logging.FileHandler(
            ARNGLogger._shared_log_file,
            mode='a',  # 追加模式
            encoding='utf-8'
        )
        ARNGLogger._shared_file_handler.setLevel(logging.DEBUG)

        # 设置文件处理器的格式
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ARNGLogger._shared_file_handler.setFormatter(file_formatter)

    def _setup_logger(self):
        """配置当前实例的日志器"""
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)

        # 避免重复添加处理器
        if logger.handlers:
            logger.handlers.clear()

        # 添加共享的文件处理器
        logger.addHandler(ARNGLogger._shared_file_handler)

        # 创建控制台处理器（每个实例独立）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # 控制台格式化器（简化格式）
        console_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        return logger

    def info(self, message: str):
        """记录信息级别日志"""
        self.logger.info(message)

    def debug(self, message: str):
        """记录调试级别日志"""
        self.logger.debug(message)

    def warning(self, message: str):
        """记录警告级别日志"""
        self.logger.warning(message)

    def error(self, message: str):
        """记录错误级别日志"""
        self.logger.error(message)

    def critical(self, message: str):
        """记录严重错误级别日志"""
        self.logger.critical(message)

    @classmethod
    def get_current_log_file(cls):
        """获取当前日志文件路径"""
        return cls._shared_log_file

    @classmethod
    def reset_log_system(cls):
        """重置日志系统（用于测试或重新初始化）"""
        if cls._shared_file_handler:
            cls._shared_file_handler.close()
        cls._shared_log_file = None
        cls._shared_file_handler = None
        cls._initialized = False

    def session_start(self, session_name: str):
        """标记会话开始"""
        separator = "=" * 60
        self.info(f"{separator}")
        self.info(f"会话开始: {session_name}")
        self.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info(f"{separator}")

    def session_end(self, session_name: str):
        """标记会话结束"""
        separator = "=" * 60
        self.info(f"{separator}")
        self.info(f"会话结束: {session_name}")
        self.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info(f"{separator}")
        self.info("")