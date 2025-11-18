from setuptools import setup, find_packages

setup(
    name="dbsi_toolbox",
    version="0.2.0",  # Aggiornata per includere DL e Calibrazione
    author="Francesco Guarnaccia",
    description="A comprehensive toolbox for Diffusion Basis Spectrum Imaging (Standard & Deep Learning)",
    url="https://github.com/guarnich/pyDBSI",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
    install_requires=[
        "numpy",
        "scipy",
        "nibabel",
        "dipy",
        "tqdm",
        "pandas",
        "matplotlib",
        "torch",  
    ],
)