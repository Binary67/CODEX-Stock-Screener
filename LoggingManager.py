import logging

class LoggingManager:
    """Configure application-wide logging."""

    @staticmethod
    def SetupLogging(log_file: str = "application.log") -> None:
        """Set up logging to file and console with module names."""
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        root_logger.handlers.clear()
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
