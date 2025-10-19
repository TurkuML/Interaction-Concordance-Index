from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize([Extension("ic_index", ["ic_index/ic_index.pyx"])]),
    include_dirs=[numpy.get_include()]
)
