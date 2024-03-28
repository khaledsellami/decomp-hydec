# setup.py
from setuptools import setup, find_packages

from hydec import __version__


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

if __name__ == "__main__":
    setup(
        name='hydec',
        version=__version__,
        packages=find_packages(exclude=['test']),
        install_requires=requirements,
        python_requires=">=3.9",
    )