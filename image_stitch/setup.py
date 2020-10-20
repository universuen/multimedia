from distutils.core import setup
from Cython.Build import cythonize
setup(name='utilities', ext_modules=cythonize('utilities.pyx'),)
