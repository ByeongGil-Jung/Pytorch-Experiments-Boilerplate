#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='project',
    version='0.0.1',
    description='Pytorch Project Template',
    author='Byeonggil Jung',
    author_email='jbkcose@gmail.com',
    url='https://github.com/ByeongGil-Jung/Pytorch-Project-Template',
    install_requires=['torch'],
    packages=find_packages(),
)
