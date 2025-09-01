# iob-ia

[![License](https://img.shields.io/pypi/l/iob-ia.svg?color=green)](https://github.com/loicsauteur/iob-ia/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/iob-ia.svg?color=green)](https://pypi.org/project/iob-ia)
[![Python Version](https://img.shields.io/pypi/pyversions/iob-ia.svg?color=green)](https://python.org)
[![CI](https://github.com/loicsauteur/iob-ia/actions/workflows/ci.yml/badge.svg)](https://github.com/loicsauteur/iob-ia/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/loicsauteur/iob-ia/branch/main/graph/badge.svg)](https://codecov.io/gh/loicsauteur/iob-ia)

Package for 3D image analysis.

# Description

This package contains functions for 3D cell segmentation, classification, measurements and visualization.
- Segmentation:
   - Wrapper for 3D cellpose segmentation
   - Measurement-based (region-props) filtering of objects
   - Cell and cytoplasm mask generation from nuclear segmentations
- Classification:
   - Create tables with measurements (region-props) for different cell compartments
   - Classify based on measurements
   -

# Usage

use via Jupyter notebooks. napari is used for visualization.


# Installation

Create environment (with conda):

`conda create -n iob-ia python=3.10`

Activate environment:

`conda activate iob-ia`

In the environment:

Install pytorch with cuda 12.6 (for the newest cellpose):

Fow Windows and Linux:

<!--For cellpose v4.*; according to docs-->
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`

Install the package:

`pip install git+https://github.com/loicsauteur/iob-ia.git`

<!--
Install Test dependencies
    `pip install -e ".[test, dev]"`
-->


<!--
for cellpose approx. v4.0.2
pip install torch --index-url https://download.pytorch.org/whl/cu118

For cellpose v3.x
`pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu118`

Here the version for older cellpose installs:
`pip install torch==1.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113`


# older versions of bioio needed scyjava installation
# seems like this is not necessary anymore...
Install JVM for [Bioio-BioFormats](https://github.com/bioio-devs/bioio-bioformats):

`conda install -c conda-forge scyjava`

Note: `scyjava` may require to deactivate and reactive the environment. [See](https://github.com/bioio-devs/bioio-bioformats)

In case of `JVMNoFoundExceptions`, set the `JAVA_HOME`:

Windows: `set JAVA_HOME=%CONDA_PREFIX%\Library`

macOS and Linux: `export JAVA_HOME=$CONDA_PREFIX`
-->
