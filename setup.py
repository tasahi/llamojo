from setuptools import setup, find_packages
import os

setup(
    name="llamojo",
    version="0.1.0",
    description="Pure Mojo LLM engine with llama-cpp-python API compatibility",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    package_data={"llamojo": ["libllamojo.so", "*.so"]},
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

