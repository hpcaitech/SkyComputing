#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from setuptools import find_packages, setup


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def read_requirements():
    with open('requirements.txt', 'r') as f:
        content = f.readlines()
    return content


if __name__ == '__main__':
    setup(name='dllb',
          version='0.0.1',
          description='Enable load balance for distributed deep learning training',
          long_description=readme(),
          long_description_content_type="text/markdown",
          author='HPC-AI Inc.',
          author_email='my_email',
          url='my_github',
          keywords='Python, scripts',
          packages=find_packages(),
          classifiers=[
              'Development Status :: 4 - Beta',
              'License :: OSI Approved :: Apache Software License',
              'Operating System :: OS Independent',
              'Programming Language :: Python :: 3',
              'Programming Language :: Python :: 3.5',
              'Programming Language :: Python :: 3.6',
              'Programming Language :: Python :: 3.7',
          ],
          license='Apache License 2.0',
          install_requires=read_requirements(),
          zip_safe=False,
          )
