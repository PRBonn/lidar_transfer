#!/usr/bin/env python
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
from sys import platform

print(numpy.get_include())

if platform == "darwin":
  ext = Extension('RayTracerCython', sources=["RayTracerCython.pyx"],
                  language="c++",
                  include_dirs=[numpy.get_include(), '.'],
                  extra_compile_args=["-std=c++14", '-O3', '-msse3', '-fPIC', '-march=native'])
else:
  ext = Extension('RayTracerCython', sources=["RayTracerCython.pyx"],
                  language="c++",
                  include_dirs=[numpy.get_include(), '.'],
                  extra_compile_args=["-std=c++14", '-O3', '-fopenmp', '-msse3', '-fPIC', '-march=native'],
                  extra_link_args=['-lgomp'],)

setup(name="RayTracerCython",
      ext_modules = cythonize(ext, language_level="3"))
