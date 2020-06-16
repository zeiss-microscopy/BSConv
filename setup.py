import glob
import os.path
import re

import setuptools


package_name = "bsconv"


package_dir = os.path.abspath(os.path.dirname(__file__))
source_dir = os.path.join(package_dir, package_name)
bin_dir = os.path.join(package_dir, "bin")


def get_version():
    version_filename = os.path.join(source_dir, "__init__.py")
    with open(version_filename, "r") as f:
        for line in f:
            line = line.strip()
            match = re.match(r"^__version__ = \"([^\"]*)\"", line)
            if match is not None:
                return match.group(1)
    raise RuntimeError("Could not find version string in file ''".format(version_filename))


def get_bin_filenames(extensions=("py", "sh")):
    filenames = []
    for extension in extensions:
        filenames += list(glob.glob(os.path.join(bin_dir, "*.{}".format(extension))))
    return filenames
    

setuptools.setup(
    name=package_name,
    version=get_version(),
    description="Reference implementation for Blueprint Separable Convolutions (BSConv)",
    url="https://github.com/zeiss-microscopy/BSConv",
    license="BSD 3-Clause Clear License",
    packages=setuptools.find_packages(),
    scripts=get_bin_filenames(),
)
