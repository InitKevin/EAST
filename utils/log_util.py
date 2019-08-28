import logging
from logging import handlers
import os
import os.path as ops

def init_logger(level=logging.DEBUG,
                when="D",  # 每天回滚一个
                backup=10, # 回滚10个文件
                _format="%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d行 %(message)s"):

    log_path = ops.join(os.getcwd(), 'logs/east.log')
    _dir = os.path.dirname(log_path)
    if not os.path.isdir(_dir):os.makedirs(_dir)

    formatter = logging.Formatter(_format)
    logging.setLevel(level)

    handler = handlers.TimedRotatingFileHandler(log_path, when=when, backupCount=backup)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logging.addHandler(handler)

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logging.addHandler(handler)
