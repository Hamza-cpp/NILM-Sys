# Copyright 2024 OKHADIR Hamza
#
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

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="NILM-Sys",
    version="0.1.1",
    author="OKHADIR Hamza",
    author_email="hamza.okhadir2018@gmail.com",
    description="NILM-Sys focuses on optimizing energy consumption in buildings through Non-Intrusive Load Monitoring (NILM). By leveraging advanced neural network architectures, the system disaggregates aggregate power data to estimate the power usage of individual appliances, contributing to climate change mitigation by enhancing energy efficiency in building",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Hamza-cpp/NILM-Sys",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy==1.24.3",
        "scikit-learn==1.5.1",
        "scipy==1.14.1",
        "pandas==2.2.2",
        "PyYAML==6.0.2",
        "torch==2.4.0",
        "matplotlib==3.9.2",
        "ray[tune]==2.34.0",
        "ipykernel==6.29.5",
    ],
)
