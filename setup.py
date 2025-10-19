from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

setup(
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules = cythonize([Extension("ic_index.ic_index", ["src/ic_index/ic_index.pyx"],include_dirs=["src/ic_index", numpy.get_include()])]),
)
