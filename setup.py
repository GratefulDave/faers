from setuptools import setup, find_packages

setup(
    name="faers_processor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "tqdm",
        "dask",
        "vaex",
        "pyarrow",
        "requests",
    ],
    python_requires=">=3.8",
)
