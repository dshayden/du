import pip
import logging
import pkg_resources
try:
  from setuptools import setup
except ImportError:
  from distutils.core import setup

try:
  install_reqs = _parse_requirements("requirements.txt")
except Exception:
  logging.warning('Fail load requirements file, so using default ones.')
  install_reqs = []

def readme():
  with open('README.rst') as f:
    return f.read()

setup(name='du',
  version='0.2',
  description='Data handling utilities',
  long_description=readme(),
  classifiers=[
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ],
  keywords='data science utility',
  url='',
  author='David S. Hayden',
  author_email='dshayden@mit.edu',
  license='MIT',
  packages=['du', 'du.stats'],
  install_requires=install_reqs,
  test_suite='nose.collector',
  tests_require=['nose', 'nose-cover3'],
  include_package_data=True,
  zip_safe=False)
