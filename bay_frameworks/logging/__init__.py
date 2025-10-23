import logging
import os


def _create_logger() -> logging.Logger:
	logger = logging.getLogger("bay_frameworks")
	if not logger.handlers:
		level_name = os.getenv("BAY_FRAMEWORKS_LOG_LEVEL", "INFO").upper()
		level = getattr(logging, level_name, logging.INFO)
		logger.setLevel(level)
		handler = logging.StreamHandler()
		handler.setLevel(level)
		formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
		handler.setFormatter(formatter)
		logger.addHandler(handler)
		logger.propagate = False
	return logger


logger = _create_logger()

__all__ = ["logger"]

