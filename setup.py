try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

USE_CYTHON = True

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [Extension("lib.attribute_calculations", ['lib/attribute_calculations' + ext]),
              Extension("lib.create_clsf_raster", ["lib/create_clsf_raster" + ext]),
              Extension("lib.rescale_intensity", ['lib/rescale_intensity' + ext])]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions, annotate=True)

setup(
    ext_modules = extensions,
)
