#!/usr/bin/env python3

from distutils.core import setup

setup(
    name='LOTClass',
    version='0.1.0',
    description='',
    url='https://github.com/goncalogiga/LOTClass',
    author='Gonçalo Giga',
    author_email='',
    license='',
    packages=['LOTClass'],
    install_requires=[
        'torch==1.5.0',
        'transformers==3.3.1',
        'joblib==0.16.0',
        'nltk==3.5',
        'numpy==1.18.5',
        'tqdm==4.47.0'
    ]
)
