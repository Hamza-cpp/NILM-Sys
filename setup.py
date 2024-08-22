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

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="NILM-Sys",
    version="0.1.0",
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
    python_requires=">=3.7",
    install_requires=[
        "numpy==1.19.5",
        "scikit-learn==0.24.1",
        "scipy==1.6.0",
        "pandas==1.2.1",
        "PyYAML==5.4.1",
        "torch==1.7.1",
        "matplotlib==3.5.3",
        "ray[tune]==2.7.2",
        "ipykernel==6.16.2",
    ],
)
