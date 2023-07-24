import os
from setuptools import setup

VERSION = '0.0.2'

setup(
    name='LxMLS Toolkit',
    version=VERSION,
    author='LxMLS team',
    description='Machine Learning and Natural Language toolkit',
    license='MIT',
    keywords='machine learning',
    url='https://github.com/LxMLS/lxmls-toolkit',
    py_modules=['lxmls'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
    ],
)
