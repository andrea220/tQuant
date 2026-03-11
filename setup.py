from setuptools import setup, find_packages

setup(
    name="tensorquant",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.0",
        "python-dateutil==2.9.0.post0",
        "pandas==2.2.2",
    ],
    description="TensorFlow-Python financial library",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    author="Andrea Carapelli",
    author_email="carapelliandrea@email.com",
    url="https://github.com/andrea220/tQuant",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    test_suite="tests",
)