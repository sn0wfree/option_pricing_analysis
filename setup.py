# coding=utf-8
from setuptools import setup, find_packages

from option_pricing_analysis import __version__, __author__

setup(
    name="nodes",
    version=__version__,
    keywords= ("Node", "Databases"),
    description="node programming",
    long_description="node databases",
    license="MIT Licence",

    url="http://www.github.com/sn0wfree",
    author=__author__,
    author_email="snowfreedom0815@gmail.com",

    packages=find_packages(),
    include_package_data=True,


)
