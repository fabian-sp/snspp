import io
import os
from setuptools import setup, find_packages

# Package meta-data.
NAME = 'snspp'
DESCRIPTION = 'Algorithms for stochastic composite optimization'
URL = 'https://github.com/fabian-sp/snspp'
EMAIL = 'fabian.schaipp@gmail.com'
AUTHOR = 'Fabian Schaipp'
REQUIRES_PYTHON = '>=3.7.0'
VERSION = "0.1.0"

# What packages are required for this module to be executed?
REQUIRED = [
    "numpy>=1.18.1", "numba>=0.49.0", "pandas", "scipy",
    "matplotlib", "seaborn"]

# What packages are optional?
EXTRAS = {
    }

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Software Development
Operating System :: Unix
"""


# Where the magic happens:
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["scripts.*", "scripts"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='',
    keywords=[
    ],
    classifiers=[_f for _f in CLASSIFIERS.split("\n") if _f]
)
