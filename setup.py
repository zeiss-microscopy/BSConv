import os.path
import re

import setuptools


package_name = "bsconv"


def get_version():
    version_filename = os.path.join(os.path.abspath(os.path.dirname(__file__)), package_name, "__init__.py")
    with open(version_filename, "r") as f:
        for line in f:
            line = line.strip()
            match = re.match(r"^__version__ = \"([^\"]*)\"", line)
            if match is not None:
                return match.group(1)
    raise RuntimeError("Could not find version string in file ''".format(version_filename))
    

setuptools.setup(
    name=package_name,
    version=get_version(),
    description="Reference implementation for Blueprint Separable Convolutions (BSConv)",
    url="https://github.com/zeiss-microscopy/BSConv",
    license="BSD 3-Clause Clear License",
    packages=setuptools.find_packages(),
)
