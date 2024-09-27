from setuptools import setup, find_packages

setup(
    name="tensorquant",  
    version="0.0.6",  
    packages=find_packages(),  
    install_requires=[
        'tensorflow>=2.0',
        'tensorflow-probability>=0.20',
        'pandas',
        'python-dateutil',
    ],  
    description="TensorFlow-Python financial library",
    long_description=open('README.md', encoding="utf8").read(),  
    long_description_content_type='text/markdown',  
    author="Andrea Carapelli",
    author_email="carapelliandrea@email.com",
    url="https://github.com/andrea220/tQuant",  
    classifiers=[  
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL License",  
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',  
    test_suite='tests'
)