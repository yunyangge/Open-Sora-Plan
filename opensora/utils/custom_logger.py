import logging
import colorlog
import accelerate.logging

def get_logger(name, level=logging.DEBUG, use_accelerate=True):
    if use_accelerate:
        logger = accelerate.logging.get_logger(name)
        logger.logger.setLevel(level)
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    color_formatter = colorlog.ColoredFormatter(
        '%(yellow)s[%(asctime)s] %(log_color)s[%(levelname)s] %(green)s[%(name)s] %(log_color)s%(message)s',
        datefmt=None,
    	reset=True,
        log_colors={
            'DEBUG': 'purple',
            'INFO': 'cyan',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
	    style='%'
    )
    
    if use_accelerate:
        for handler in logger.logger.handlers:
            logger.logger.removeHandler(handler)

        logger.logger.addHandler(console_handler)
    else:
        for handler in logger.handlers:
            logger.removeHandler(handler)

        logger.addHandler(console_handler)

    return logger