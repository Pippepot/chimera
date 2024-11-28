from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name='chimera',
    version='0.0.1',
    description='Feverdream programming language',
    author='Philip Thomsen',
    license='MIT',
    packages=['chimera'],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)