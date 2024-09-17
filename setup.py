from setuptools import setup, find_packages

setup(
    name="tensorquant",  
    version="0.0.4",  
    packages=find_packages(),  
    install_requires=[],  # Dipendenze richieste, puoi aggiungere pacchetti necessari qui
    description="TensorFlow-Python financial library",
    long_description=open('README.md').read(),  
    long_description_content_type='text/markdown',  
    author="Andrea Carapelli",
    author_email="carapelliandrea@email.com",
    url="https://github.com/andrea220/tQuant",  
    classifiers=[  
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',  
    test_suite='tests'
)