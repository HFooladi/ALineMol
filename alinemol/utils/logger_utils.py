from __future__ import absolute_import, division, print_function
import logging
import os
import sys
import datetime
from logging.handlers import TimedRotatingFileHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ColorCodes:
    """ANSI color codes for terminal output."""

    # Reset
    RESET = "\033[0m"

    # Regular Colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bold Colors
    BOLD_BLACK = "\033[1;30m"
    BOLD_RED = "\033[1;31m"
    BOLD_GREEN = "\033[1;32m"
    BOLD_YELLOW = "\033[1;33m"
    BOLD_BLUE = "\033[1;34m"
    BOLD_MAGENTA = "\033[1;35m"
    BOLD_CYAN = "\033[1;36m"
    BOLD_WHITE = "\033[1;37m"

    # Background Colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


class ColoredFormatter(logging.Formatter):
    """A custom formatter that adds colors to log messages based on log level."""

    # Color mapping for different log levels
    LEVEL_COLORS = {
        "DEBUG": ColorCodes.BOLD_BLUE,
        "INFO": ColorCodes.BOLD_GREEN,
        "WARNING": ColorCodes.BOLD_YELLOW,
        "ERROR": ColorCodes.BOLD_RED,
        "CRITICAL": ColorCodes.BOLD_MAGENTA + ColorCodes.BG_WHITE,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Check if we're in a terminal that supports colors
        self.use_colors = self._supports_color()

    def _supports_color(self):
        """Check if the current terminal supports color output."""
        # Check if stdout is a TTY and not redirected
        if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
            return False

        # Check for common environment variables that indicate color support
        if "COLORTERM" in os.environ:
            return True

        term = os.environ.get("TERM", "").lower()
        if any(color_term in term for color_term in ["color", "ansi", "xterm", "linux"]):
            return True

        # Windows terminal color support
        if os.name == "nt":
            try:
                # Try to enable ANSI escape sequences on Windows 10+
                import ctypes

                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
                return True
            except Exception:
                return False

        return False

    def format(self, record):
        """Format the log record with colors."""
        # Get the original formatted message
        formatted_message = super().format(record)

        # Add colors if supported
        if self.use_colors and record.levelname in self.LEVEL_COLORS:
            color = self.LEVEL_COLORS[record.levelname]

            # Color the level name and make timestamp cyan
            colored_message = formatted_message.replace(
                record.levelname, f"{color}{record.levelname}{ColorCodes.RESET}"
            )

            # Color the timestamp
            timestamp_str = record.asctime if hasattr(record, "asctime") else ""
            if timestamp_str in colored_message:
                colored_message = colored_message.replace(
                    timestamp_str, f"{ColorCodes.CYAN}{timestamp_str}{ColorCodes.RESET}"
                )

            # Color the logger name
            logger_name = record.name if hasattr(record, "name") else ""
            if logger_name in colored_message:
                colored_message = colored_message.replace(
                    f"| {logger_name} |", f"| {ColorCodes.BOLD_CYAN}{logger_name}{ColorCodes.RESET} |"
                )

            # Color the file path and line number
            if hasattr(record, "relativepath") and record.relativepath:
                colored_message = colored_message.replace(
                    f"| {record.relativepath} | {record.lineno} |",
                    f"| {ColorCodes.BLUE}{record.relativepath}{ColorCodes.RESET} | {ColorCodes.MAGENTA}{record.lineno}{ColorCodes.RESET} |",
                )

            return colored_message

        return formatted_message


class PackagePathFilter(logging.Filter):
    """A custom logging filter for adding the relative path to the log record."""

    def filter(self, record):
        """add relative path to record"""
        pathname = record.pathname
        record.relativepath = None
        abs_sys_paths = map(os.path.abspath, sys.path)
        for path in sorted(abs_sys_paths, key=len, reverse=True):  # longer paths first
            if not path.endswith(os.sep):
                path += os.sep
            if pathname.startswith(path):
                record.relativepath = os.path.relpath(pathname, path)
                break
        return True


class Logger(object):
    """A custom logger class that provides logging functionality to console and file."""

    def __init__(self, logger_name="None", level=logging.INFO):
        """
        Args:
            logger_name (str): The name of the logger (default: 'None')
        """
        self.logger = logging.getLogger(logger_name)
        logging.root.setLevel(logging.NOTSET)
        self.log_file_name = "alinemol_{0}.log".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

        cwd_path = os.path.abspath(os.getcwd())
        self.log_path = os.path.join(cwd_path, "logs")

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.backup_count = 5

        self.console_output_level = level
        self.file_output_level = level
        self.DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

        # Regular formatter for file output (no colors)
        self.file_formatter = logging.Formatter(
            "%(asctime)s | %(relativepath)s | %(lineno)s | %(levelname)s | %(name)s | %(message)s",
            self.DATE_FORMAT,
        )

        # Colored formatter for console output
        self.console_formatter = ColoredFormatter(
            "%(asctime)s | %(relativepath)s | %(lineno)s | %(levelname)s | %(name)s | %(message)s",
            self.DATE_FORMAT,
        )

    def get_logger(self):
        """
        Get the logger object.

        Returns:
            logging.Logger - a logger object.

        """
        if not self.logger.handlers:
            # Console handler with colored output
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self.console_formatter)
            console_handler.setLevel(self.console_output_level)
            console_handler.addFilter(PackagePathFilter())
            self.logger.addHandler(console_handler)

            # File handler with regular (non-colored) output
            file_handler = TimedRotatingFileHandler(
                filename=os.path.join(self.log_path, self.log_file_name),
                when="D",
                interval=1,
                backupCount=self.backup_count,
                delay=True,
                encoding="utf-8",
            )
            file_handler.setFormatter(self.file_formatter)
            file_handler.setLevel(self.file_output_level)
            file_handler.addFilter(PackagePathFilter())
            self.logger.addHandler(file_handler)
        return self.logger

    def set_level(self, level):
        """
        Set the logging level for both console and file handlers.

        Args:
            level (str or int): The logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())

        self.logger.setLevel(level)

        # Update handler levels
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, TimedRotatingFileHandler):
                # Console handler
                self.console_output_level = level
                handler.setLevel(level)
            elif isinstance(handler, TimedRotatingFileHandler):
                # File handler
                self.file_output_level = level
                handler.setLevel(level)


# Create the default logger instance
logger = Logger("ALineMol").get_logger()
logger.setLevel(logging.INFO)


def demo_colorful_logging():
    """Demonstrate the colorful logging functionality."""
    print("🎨 ALineMol Colorful Logging Demo:")
    print("=" * 50)

    # Set to DEBUG level to show all messages
    logger.setLevel(logging.DEBUG)

    logger.debug("This is a DEBUG message - shows detailed diagnostic information")
    logger.info("This is an INFO message - general information about program execution")
    logger.warning("This is a WARNING message - something unexpected happened")
    logger.error("This is an ERROR message - a serious problem occurred")
    logger.critical("This is a CRITICAL message - the program may not be able to continue")

    # Reset to INFO level
    logger.setLevel(logging.INFO)
    print("\n✅ Color logging setup complete!")


if __name__ == "__main__":
    demo_colorful_logging()
