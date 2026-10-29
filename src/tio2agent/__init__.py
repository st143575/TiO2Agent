"""TiO2Agent: RAG + multi-step Agent QA backend for TiO2 technical documents."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tio2agent")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0"

__all__ = ["__version__"]
