from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION','CYTHON_TRACE=1')],
    ext_modules = cythonize("tree.pyx"),
    include_dirs=[numpy.get_include()]
)