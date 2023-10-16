# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get requirments from requirments.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# This call to setup() does all the work
setup(
    name="ageml",
    version="0.0.1",
    description="Age Modellingi package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Computational Neuroimaging Lab Bilbao",
    author_email="jorgegarciacondado@gmail.com",
    url="https://github.com/compneurobilbao/AgeModelling",
    license="GNU General Public License v3.0",
    packages=[],
    include_package_data=True,
    python_requires="==3.8",
    install_requires=[requirements]
)
