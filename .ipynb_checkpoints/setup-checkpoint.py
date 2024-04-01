from setuptools import setup, find_packages

setup(
    name="d3rlpy",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch==2.1.0",
        "tqdm==4.66.1",
        "pandas==2.0.3",
        "numpy==1.23.3",
        "h5py==3.1.0",
        "gym==0.26.2",
        "click==8.1.7",
        "typing-extensions==4.3.0",
        "structlog==23.2.0",
        "colorama==0.4.6",
        "gymnasium==0.29.1",
        "dataclasses-json==0.6.2",
        "wandb"
    ]
)
