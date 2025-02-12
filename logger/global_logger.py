from logger.logger import Logger
from logger.console_logger_provider import ConsoleLoggerProvider

def get_logger():
    """
    Returns a singleton logger instance.
    """
    provider = ConsoleLoggerProvider()
    return Logger(provider)
