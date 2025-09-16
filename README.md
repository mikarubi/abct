# abct

abct ([abct.rubinovlab.net](https://abct.rubinovlab.net)) is a MATLAB and Python toolbox for unsupervised learning, network science, and imaging/network neuroscience.

The toolbox includes three variants of global residualization, the Loyvain and co-Loyvain methods (for _k_-means, _k_-modularity, or spectral clustering of data or network inputs), as well as binary and weighted canonical and co-neighbor components, and m-umap embeddings. It also includes computation of degree centralities (first, second, and residual), dispersion centralities (squared coefficient of variation, _k_-participation coefficient), and network shrinkage.

See this reference for more details: Rubinov M. Unifying equivalences across unsupervised learning, network science, and imaging/network neuroscience. arXiv, 2025, [doi:10.48550/arXiv.2508.10045](https://doi.org/10.48550/arXiv.2508.10045).

### Download

[ABCT MATLAB](https://github.com/mikarubi/abct/raw/refs/heads/main/abct-matlab.zip){.btn .btn-primary}
[ABCT Python](https://github.com/mikarubi/abct/raw/refs/heads/main/abct-python.zip){.btn .btn-primary}
[GitHub](https://github.com/mikarubi/abct){.btn .btn-dark}

### Interactive examples

The examples illustrate the main analyses in the above study, and can be run interactively as online Jupyter notebooks in [Google Colab](https://colab.research.google.com/) (You need a Google account). To run in Colab, click the **Open in Colab** button at the top of the page of each example.

Our example brain-imaging data come from the [Human Connectome Project](https://www.humanconnectome.org/), a large brain-imaging resource.

NB: For illustrative ease, some example analyses are simplified versions of analyses in the original study.

### Contact

Mika Rubinov: mika.rubinov at vanderbilt.edu


# Getting started

## MATLAB

### Dependencies

- MATLAB ≥R2024a
- kneighbor: Statistics and Machine Learning Toolbox (if `method="indirect"`)
- mumap: [Manopt](https://www.manopt.org/) (if `solver="trustregions"`) or Deep Learning Toolbox (if `solver="adam"`)
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

### Usage

Basic usage:

```
[output1, output2, ...] = function_name(input1, input2, ..., name=value)
```

See the individual function pages for more details.

See the individual examples for example usage (in Python).

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

Download the latest release and install using pip.
```
pip install ./abct-python
```

### Pypi install

Install directly from pypi (may not be the latest release).
```
pip install abct
```

### Usage

Basic usage:

```
import abct
output1, output2, ... = abct.function_name(input1, input2, ..., name=value)
```

See the individual function pages for more details.

See the individual examples for example usage.

