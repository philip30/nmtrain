from setuptools import setup, find_packages
import sys

sys.path.append("./nmtrain")

install_requires = [
  'Chainer>=2.0.0',
  'numpy>=1.9.0',
  'tabulate>=0.7.7',
  'protobuf>=3.1.0'
]

setup(
  name='nmtrain',
  version='1.0.0',
  description='Neural Machine Translation toolkit',
  author='Philip Arthur',
  author_email='philip.arthur30@gmail.com',
  license='MIT License',
  install_requires=install_requires,
  packages=[
      'nmtrain',
  ],
  platforms='requires Python Chainer Numpy'
)
