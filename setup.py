# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup
from setuptools import find_packages


setup(
    name="pyloader",
    version="0.5.0",
    description="An asynchronous Python dataloader for loading big datasets with limited memory.",
    keywords=[
        "deep learning",
        "data loader",
    ],
    license="Apache 2.0",
    author="iwangjian",
    author_email="jwanglvy@gmail.com",
    url="https://github.com/iwangjian/pyloader",
    python_requires=">=3.6.0",
    install_requires=[
        "numpy",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_data={"pyloader": ["VERSION"]},
    packages=find_packages(),
    zip_safe=False,
)
