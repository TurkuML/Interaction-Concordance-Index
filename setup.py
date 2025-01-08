from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize(["A_index.pyx", "swapped.pyx"]),
    include_dirs=[numpy.get_include()]
)
