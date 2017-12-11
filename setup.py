#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:14:27 2017

@author: dnicholson
"""

from setuptools import setup

setup(name='Gasex',
      version='0.1',
      description='functions for dissolved gases and air-sea exchange for oceanography',
      url='http://github.com/dnicholson/gasex-python',
      author='David Nicholson',
      author_email='dnicholson@whoi.edu',
      license='MIT',
      packages=['gasex'],
      install_requires=['gsw'],
      zip_safe=False)
