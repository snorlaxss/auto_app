from setuptools import find_packages
from setuptools import setup

setup(
    name='fake_interface_pkg',
    version='0.0.0',
    packages=find_packages(
        include=('fake_interface_pkg', 'fake_interface_pkg.*')),
)
