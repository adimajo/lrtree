# Metadata live in pyproject.toml; this shim only keeps `python setup.py sdist bdist_wheel` (used by the internal
# GitLab pipeline) working
from setuptools import setup

setup()
