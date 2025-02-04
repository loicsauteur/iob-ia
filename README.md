# iob-ia

[![License](https://img.shields.io/pypi/l/iob-ia.svg?color=green)](https://github.com/loicsauteur/iob-ia/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/iob-ia.svg?color=green)](https://pypi.org/project/iob-ia)
[![Python Version](https://img.shields.io/pypi/pyversions/iob-ia.svg?color=green)](https://python.org)
[![CI](https://github.com/loicsauteur/iob-ia/actions/workflows/ci.yml/badge.svg)](https://github.com/loicsauteur/iob-ia/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/loicsauteur/iob-ia/branch/main/graph/badge.svg)](https://codecov.io/gh/loicsauteur/iob-ia)

Package for image analysis.


# Installation

Create environment (with conda):

`conda create -n iob-ia python=3.10`

Activate environment:

`conda activate iob-ia`

In the environment:

<!--- 
Don't use conda to install pytorch...
`conda install pytorch==1.12.0 cudatoolkit=11.3 -c pytorch`
--->

Install newer version of pytorch with cuda 11.8 (to have the newest cellpose):

`pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu118`

<!--
Below the old version...

`pip install torch==1.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113`
-->

`conda install -c conda-forge scyjava`

`scyjava` may require to deactivate and reactive the environment. [See](https://github.com/bioio-devs/bioio-bioformats)
And maybe (I did not need to following):

mac and linux:
`export JAVA_HOME=$CONDA_PREFIX`

windows:
`set JAVA_HOME=%CONDA_PREFIX%\Library`

Install the package... TODO

