#!/usr/bin/env python
# -*- coding: utf-8 -*-

# sudo apt-get install libasound2-dev

from setuptools import setup

setup(
    name = "Audio Classification",
    version="1.0",
    description="--",
    long_description="--",
    long_description_content_type='text/markdown',
    author="Pedro H.",
    author_email="pedrohcd@hotmail.com",
    python_requires='>=3.6.0',
    #packages=[],
    install_requires=[
        'numpy', 
        'jupyter',
        'matplotlib',
        'audioread',
        'librosa',
        'simpleaudio',
        'scipy',
        'tensorflow==2.5.3'
    ],
    #include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    # $ setup.py publish support.
    #cmdclass={
    #}
)