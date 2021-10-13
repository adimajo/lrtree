from setuptools import setup, find_packages
import os
import codecs
import re

here = os.path.abspath(os.path.dirname(__file__))


def find_version(*file_paths):
    with codecs.open(os.path.join(here, *file_paths), 'r') as fp:
        version_file = fp.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


with open('requirements.txt') as fp:
    install_requires = [x.split("/")[-1] for x in fp.read().splitlines()[1:]]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='glmtree',
      version=find_version("glmtree", "__init__.py"),
      description='glmtree: logistic regression trees',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url=None,
      packages=find_packages(),
      include_package_data=True,
      author='Adrien Ehrhardt, Dmitry Gaynullin',
      author_email='adrien.ehrhardt@credit-agricole-sa.fr',
      install_requires=install_requires,
      classifiers=(
          "Programming Language :: Python :: 3",
          "Operating System :: OS Independent",
      ),
      )
