from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize([
        "BFC_libs/electrochemistry/__init__.pyx",
        "BFC_libs/electrochemistry/biologic.pyx",
        
        "BFC_libs/Raman/__init__.pyx"
        ])
)