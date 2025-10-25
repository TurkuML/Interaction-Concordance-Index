from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

extensions = [Extension("ic_index.ic_index", ["src/ic_index/ic_index.pyx"], include_dirs=["src/ic_index",numpy.get_include()])]

ext_modules = cythonize(
    extensions, 
    compiler_directives = {'language_level' : "3"}
)

setup(
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules = ext_modules
)

