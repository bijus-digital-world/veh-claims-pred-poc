"""
logger.py

Centralized logging configuration for the Nissan Telematics POC application.

Provides structured logging with:
- Console output for development
- File output for production debugging
- Configurable log levels
- Separate loggers for different modules
- Rotation support for production logs
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler
from datetime import datetime


# ANSI color codes for console output
class LogColors:
    """ANSI color codes for prettier console logs"""
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    COLORS = {
        logging.DEBUG: LogColors.GRAY,
        logging.INFO: LogColors.CYAN,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.RED + LogColors.BOLD,
    }
    
    def format(self, record):
        # Add color to levelname
        color = self.COLORS.get(record.levelno, LogColors.RESET)
        record.levelname = f"{color}{record.levelname}{LogColors.RESET}"
        
        # Add color to name
        record.name = f"{LogColors.BLUE}{record.name}{LogColors.RESET}"
        
        return super().format(record)


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup a logger with console and optional file handlers.
    
    Args:
        name: Logger name (usually module name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for file logging
        console_output: Whether to output to console
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    
    Example:
        >>> from utils.logger import setup_logger
        >>> logger = setup_logger("my_module", level="DEBUG", log_file="logs/my_module.log")
        >>> logger.info("Application started")
        >>> logger.error("An error occurred", exc_info=True)
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False
    
    # Console handler with colors
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        console_formatter = ColoredFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        file_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


# Create module-level loggers for different components
# These can be imported and used throughout the application

# Main application logger
app_logger = setup_logger(
    "app",
    level="INFO",
    log_file="logs/app.log"
)

# Helper functions logger
helper_logger = setup_logger(
    "helper",
    level="INFO",
    log_file="logs/helper.log"
)

# Chat functionality logger
chat_logger = setup_logger(
    "chat_helper",
    level="INFO",
    log_file="logs/chat.log"
)

# Components logger
components_logger = setup_logger(
    "components",
    level="INFO",
    log_file="logs/components.log"
)

# Config logger
config_logger = setup_logger(
    "config",
    level="INFO",
    log_file="logs/config.log"
)


def log_function_call(logger: logging.Logger, func_name: str, **kwargs):
    """
    Log a function call with parameters (for debugging).
    
    Args:
        logger: Logger instance to use
        func_name: Name of the function being called
        **kwargs: Function parameters to log
    
    Example:
        >>> log_function_call(logger, "fetch_dealers", lat=36.0, lon=-115.0)
    """
    params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.debug(f"Calling {func_name}({params})")


def log_performance(logger: logging.Logger, operation: str, duration_ms: float):
    """
    Log performance metrics.
    
    Args:
        logger: Logger instance to use
        operation: Name of the operation
        duration_ms: Duration in milliseconds
    
    Example:
        >>> import time
        >>> start = time.time()
        >>> # ... do work
        >>> log_performance(logger, "data_load", (time.time() - start) * 1000)
    """
    logger.info(f"Performance: {operation} completed in {duration_ms:.2f}ms")


def log_dataframe_info(logger: logging.Logger, df_name: str, df):
    """
    Log DataFrame information for debugging.
    
    Args:
        logger: Logger instance to use
        df_name: Name of the DataFrame
        df: pandas DataFrame
    
    Example:
        >>> log_dataframe_info(logger, "df_history", df)
    """
    if df is None:
        logger.warning(f"DataFrame {df_name} is None")
        return
    
    try:
        logger.info(f"DataFrame {df_name}: shape={df.shape}, columns={list(df.columns)}")
        logger.debug(f"DataFrame {df_name} dtypes: {df.dtypes.to_dict()}")
    except Exception as e:
        logger.error(f"Failed to log DataFrame info for {df_name}: {e}")


# Convenience function to get logger for current module
def get_logger(name: str = None) -> logging.Logger:
    """
    Get logger for a module.
    
    Args:
        name: Logger name (defaults to calling module name)
    
    Returns:
        Logger instance
    
    Example:
        >>> from utils.logger import get_logger
        >>> logger = get_logger(__name__)
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return logging.getLogger(name)


# Export commonly used loggers
__all__ = [
    'setup_logger',
    'get_logger',
    'app_logger',
    'helper_logger',
    'chat_logger',
    'components_logger',
    'config_logger',
    'log_function_call',
    'log_performance',
    'log_dataframe_info',
]

