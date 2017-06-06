from setuptools import setup


def readme():
  with open('README.rst') as f:
    return f.read()

setup(name='du',
  version='0.1',
  description='Data handling utilities',
  long_description=readme(),
  classifiers=[
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 2.7',
  ],
  keywords='data science utility',
  url='',
  author='David S. Hayden',
  author_email='dshayden@mit.edu',
  license='MIT',
  packages=['du'],
  install_requires=[
    'joblib',
  ],
  test_suite='nose.collector',
  tests_require=['nose', 'nose-cover3'],
  include_package_data=True,
  zip_safe=False)
