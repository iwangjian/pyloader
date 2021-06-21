###########################################################
# METADATA
###########################################################

# Version.  For each new release, the version number should be updated
# in the file VERSION.
import os

try:
    # If a VERSION file exists
    version_file = os.path.join(os.path.dirname(__file__), "VERSION")
    with open(version_file, "r") as infile:
        __version__ = infile.read().strip()
except NameError:
    __version__ = "unknown (running code interactively?)"
except IOError as ex:
    __version__ = "unknown (%s)" % ex

if __doc__ is not None:  # fix for the ``python -OO``
    __doc__ += "\n@version: " + __version__

__license__ = "Apache License, Version 2.0"
__longdescr__ = "An asynchronous Python dataloader for loading big datasets with limited memory."
__keywords__ = [
    "deep learning",
    "data loader",
]

###########################################################
# TOP-LEVEL MODULES
###########################################################

from pyloader.datareader import *
from pyloader.dataset import *
from pyloader.datacollator import *
from pyloader.dataloader import *
