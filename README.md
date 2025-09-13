---
title: Installation
---

## MATLAB Installation

### Dependencies

- MATLAB ≥R2024a
- kneighbor: Statistics and Machine Learning Toolbox (if `method="indirect"`)
- mumap: Manopt (`trustregions` solver) or Deep Learning Toolbox (`adam` solver)
- mumap: Parallel Computing Toolbox (if `gpu=true`)

### Simple installation

Add `./abct-matlab/abct` to your MATLAB path.
```
addpath('./abct-matlab/abct')
```

### Package installation

Install using the MATLAB Package Manager.
```
mpminstall("./abct-matlab/abct")
```

### Getting help on a function

```
help functionname
doc functionname
```

## Python Installation

### Dependencies

- Python ≥3.11
- numpy
- scipy
- igraph
- pytorch
- pynndescent
- pymanopt
- pydantic

### Direct install

Download latest release and install using pip.
```
pip install ./abct-python
```

### Pypi install

Install directly from pypi (may not be the latest release).
```
pip install abct
```

### Getting help on a function

```
import abct

help(abct.functionname)  # python
abct.functionname?       # ipython
```

## Computational notebooks

All the provided examples can be run online as Jupyter notebooks in [Google Colab](https://colab.research.google.com/) (You need a Google account). To run in Colab, click the "Open in Colab" button at the top of the page of each example.
