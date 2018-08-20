from distutils.core import setup
from distutils.extension import Extension

USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [Extension("attribute_calculations", ['attribute_calculations'+ext]),
              Extension("create_clsf_raster", ["create_clsf_raster"+ext])]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(
    ext_modules = extensions
)
