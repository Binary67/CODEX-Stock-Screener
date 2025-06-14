import os
import unittest
import logging
from LoggingManager import LoggingManager


class TestLoggingManager(unittest.TestCase):
    def test_setup_logging_creates_file(self):
        log_file = "test_app.log"
        if os.path.exists(log_file):
            os.remove(log_file)
        LoggingManager.SetupLogging(log_file)
        logging.getLogger(__name__).info("test message")
        logging.shutdown()
        self.assertTrue(os.path.exists(log_file))
        with open(log_file, "r", encoding="utf-8") as file:
            content = file.read()
        self.assertIn("test message", content)
        os.remove(log_file)


if __name__ == "__main__":
    unittest.main()
