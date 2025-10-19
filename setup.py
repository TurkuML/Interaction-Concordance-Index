from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    package_dir={"": "src"},
    packages=["ic_index"],
    ext_modules = cythonize([Extension("ic_index", ["ic_index/ic_index.pyx"])]),
    include_dirs=[numpy.get_include()]
)
