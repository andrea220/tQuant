from setuptools import setup, find_packages

with open("README.md", encoding="utf8") as f:
    long_description = f.read()

setup(
    name="tensorquant",
    version="1.0.0",
    packages=find_packages(include=["tensorquant", "tensorquant.*"]),
    install_requires=[
        "tensorflow>=2.16",
        "numpy>=1.26",
        "scipy>=1.11",
        "pandas>=2.2,<3",
        "python-dateutil>=2.9",
    ],
    extras_require={
        "examples": ["matplotlib>=3.8", "ipykernel>=6"],
    },
    description="TensorFlow-Python financial library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Andrea Carapelli",
    author_email="carapelliandrea@email.com",
    url="https://github.com/andrea220/tQuant",
    license="GPL-3.0-only",
    license_files=["LICENSE"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
