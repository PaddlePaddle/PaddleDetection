from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [Pybind11Extension("cocoeval_ext", ["cocoeval.cc"])]

setup(
    name="cocoeval_ext",
    version="0.0.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
