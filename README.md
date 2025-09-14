# Installation

## MATLAB

### Dependencies

- MATLAB ≥R2024a
- kneighbor: Statistics and Machine Learning Toolbox (if `method="indirect"`)
- mumap: Manopt (if `solver="trustregions"`) or Deep Learning Toolbox (if `solver="adam"`)
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

## Python

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
