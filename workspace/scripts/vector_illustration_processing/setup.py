from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension("pi_scripts",  ["pi_line.py", "pi_arithmetic.py", "pi_objects.py", "pi_path.py", "pi_point.py", "pi_sliders.py", "pi_video.py"]),
]

setup(
    name = 'pi_scripts',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)