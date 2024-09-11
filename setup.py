from setuptools import setup, find_packages

setup(
    name="tquant",  # Nome del pacchetto
    version="0.0.1",  # Versione del pacchetto
    packages=find_packages(),  # Trova automaticamente tutti i pacchetti e sottopacchetti
    install_requires=[],  # Dipendenze richieste, puoi aggiungere pacchetti necessari qui
    description="Pacchetto di analisi e caricamento dati per tQuant",
    long_description=open('README.md').read(),  # Usa il file README per la descrizione lunga
    long_description_content_type='text/markdown',  # Specifica il formato della descrizione lunga
    author="Andrea Carapelli",
    author_email="carapelliandrea@email.com",
    url="https://github.com/andrea220/tQuant",  # URL del progetto (puoi usare GitHub o un sito personale)
    classifiers=[  # Classificatori per descrivere il pacchetto
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Specifica la licenza
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',  # Versione minima di Python richiesta
    test_suite='tests'
)