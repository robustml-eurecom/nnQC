from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nnQC",
    version="0.1.0",
    author="Vincenzo Marciano",
    author_email="vincenzo.marciano@eurecom.fr",
    description="Quality Control for Medical Image Segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SanBast/nnQC",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "nnqc-train-ae=nnqc.cli:train_autoencoder",
            "nnqc-train-diffusion=nnqc.cli:train_diffusion", 
            "nnqc-inference=nnqc.cli:run_inference",
            "nnqc-evaluate=nnqc.cli:evaluate_validation",
        ],
    },
    include_package_data=True,
    package_data={
        "nnQC": ["config/*.json"],
    },
) 