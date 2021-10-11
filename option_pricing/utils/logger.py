# coding=utf-8

import logging
import sys, os, time
from functools import wraps


class Singleton(object):
    def __init__(self, cls):
        self._cls = cls
        self._instance = {}

    def __call__(self, *args, **kwargs):
        if self._cls not in self._instance:
            self._instance[self._cls] = self._cls(*args, **kwargs)
        return self._instance[self._cls]


# BASE_NAME = BASE_DIR.split(os.sep)[-1]
@Singleton
class LoggerHelper(object):
    def __init__(self, app_name=None, file_path="test.log", log_level=logging.INFO):
        app_name = str(app_name)
        # 获取logger实例，如果参数为空则返回root logger
        self.logger = logging.getLogger(app_name)

        # 指定logger输出格式
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')

        # 文件日志
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式

        # 控制台日志
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = formatter  # 也可以直接给formatter赋值

        # 为logger添加的日志处理器，可以自定义日志处理器让其输出到其他地方
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # 指定日志的最低输出级别，默认为WARN级别
        self.logger.setLevel(log_level)

        self.warn = self.logger.warning
        self.info = self.logger.info
        self.critical = self.logger.critical
        self.debug = self.logger.debug

    def deco(self, level='info', timer=True, extra_name: str = ''):
        if not extra_name.endswith('.') and extra_name != '':
            extra_name = extra_name + '.'

        def _deco(func):
            log_print_func = getattr(self, level)

            @wraps(func)
            def wrapper(*args, **kwargs):

                if timer:
                    start = time.time()
                    log_print_func(f'start to execute {extra_name}{func.__name__}! timer start! ')
                else:
                    start = 0
                    log_print_func(f'start to execute {extra_name}{func.__name__}!')
                res = func(*args, **kwargs)
                # log_print_func = globals().get('Logger', None)
                if timer:
                    spend = time.time() - start
                    log_print_func(f"{extra_name}{func.__name__} executed! spend: {spend}")
                else:
                    log_print_func(f"{extra_name}{func.__name__} executed!")
                return res

            return wrapper

        return _deco

    def sql(self, info: str):
        self.info("[SQL]: " + info)

    def status(self, info):
        self.info("[STATUS]： " + info)


# Logger = LoggerHelper(file_path="test.log", log_level=logging.INFO)
if __name__ == '__main__':
    # print(BASE_DIR, Logger)
    # Logger.warn('test')
    # Logger.info('test2')
    # Logger.debug('test3')
    # Logger.critical('test3')
    # Logger.sql('test3')
    # Logger.status('tst5')
    pass
