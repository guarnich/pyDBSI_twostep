from setuptools import setup, find_packages

setup(
    name='dbsi_toolbox',
    version='0.1.0',
    description='Python toolbox for fitting the Diffusion Basis Spectrum Imaging (DBSI) model.',
    author='Francesco Guarnaccia',
    author_email='francesco.guarnaccia@univr.it',
    url='https://github.com/guarnich/dbsi-fitting-toolbox', # URL della tua repo
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'nibabel',
        'dipy',
        'tqdm'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # O la licenza che hai scelto
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Science Apps.'
    ],
    python_requires='>=3.8',
)