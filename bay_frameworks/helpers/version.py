from importlib.metadata import PackageNotFoundError, version

from agentbay.bay_frameworks.logging import logger


def get_bay_frameworks_version():
	"""
	Get the version of bay_frameworks (which is part of agentbay package).
	
	Returns the agentbay package version, as bay_frameworks is a submodule.
	"""
	try:
		# Try to get agentbay version (since bay_frameworks is part of agentbay)
		pkg_version = version("agentbay")
		return pkg_version
	except Exception:
		# Fallback: try to import directly from agentbay
		try:
			from agentbay import __version__
			return __version__
		except Exception as e:
			logger.debug("agentbay version not found: %s", e)
			return "unknown"

