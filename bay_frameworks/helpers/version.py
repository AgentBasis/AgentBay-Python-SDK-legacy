from importlib.metadata import PackageNotFoundError, version

from MYSDK.bay_frameworks.logging import logger


def get_bay_frameworks_version():
	try:
		pkg_version = version("bay_frameworks")
		return pkg_version
	except Exception as e:
		logger.debug("bay_frameworks version not found: %s", e)
		return "unknown"

