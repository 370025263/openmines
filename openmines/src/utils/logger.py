import logging
import time, os
from pathlib import Path

# 项目主路径
PROJECT_ROOT_PATH = Path(__file__).parent.parent.parent
LOG_FILE_PATH = PROJECT_ROOT_PATH / "src" / "data" / "logs"

class MineLogger:
    def __init__(self, log_path=LOG_FILE_PATH, file_level=logging.DEBUG, console_level=logging.INFO):
        # 时间(兼容windows文件名)
        time_format = "%Y-%m-%d-%H-%M-%S"
        time_str = time.strftime(time_format, time.localtime())

        # LOGGER配置
        # 日志格式
        LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'

        # 日志级别
        CONSOLE_LOG_LEVEL = logging.INFO
        FILE_LOG_LEVEL = file_level

        # 控制台处理器
        self.CONSOLE_HANDLER = logging.StreamHandler()
        self.CONSOLE_HANDLER.setLevel(CONSOLE_LOG_LEVEL)
        self.CONSOLE_HANDLER.setFormatter(logging.Formatter(LOG_FORMAT))

        # 文件处理器
        self.log_path = log_path
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.FILE_HANDLER = logging.FileHandler(self.log_path / f'openmines_sim_{time_str}.log')
        self.FILE_HANDLER.setLevel(FILE_LOG_LEVEL)
        self.FILE_HANDLER.setFormatter(logging.Formatter(LOG_FORMAT))

    def get_logger(self, name):
        """
        在初始化后，通过get_logger方法获取模块级logger
        :param name:
        :return:
        """
        logger = logging.getLogger(name)
        # 检查是否已经添加了控制台处理器
        if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
            logger.addHandler(self.CONSOLE_HANDLER)
        # 检查是否已经添加了文件处理器
        if not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
            logger.addHandler(self.FILE_HANDLER)
        logger.setLevel(logging.DEBUG)
        return logger