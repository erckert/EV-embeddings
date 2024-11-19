# EV-embeddings

EV-embeddings provides the trained models from our paper [Assessing the role of evolutionary information for enhancing protein language model embeddings](https://doi.org/10.1038/s41598-024-71783-8). The used dataset is available at [https://doi.org/10.5281/zenodo.10026192](https://doi.org/10.5281/zenodo.10026192). 

## Table of Contents

- [Requirements](#Requirements)
- [Setup](#Setup)
- [Team](#Team)
- [License](#License)
- [Citation](#Citation)

### Requirements

For generating predictions [Python 3.10.1](https://www.python.org/downloads/release/python-3101/), 
[Pytorch](https://pytorch.org/), [NumPy](https://numpy.org/), [Colorama](https://pypi.org/project/colorama/) and 
[Biopython](https://biopython.org/) are needed. Please check the [requirements.txt](requirements.txt) for specific 
versions.

### Setup

#### Install Requirements

If you use `pip`, you can install the requirements for your environment with:

```
pip install -r requirements.txt
```

#### Setup config file

To set up your prefered configurations edit the `src/ev_embeddings.ini` file according to the type of prediction you want to run. `src/ev_embeddings_example.ini` provides an example for running single sequence predictions for the provided example files and `src/ev_embeddings_template.ini` provides an empty template.

#### Run prediction

To run the code execute:
```
python src/main.py
```

If you want to use a different `.ini` file use 
```
python src/main.py -f FILEPATH
```
instead.


### Team

Technical University of Munich - Rostlab

| Kyra Erckert | Burkhard Rost |
|:------------:|:-------------:|
|<img width=120/ src="https://github.com/erckert/EV-embeddings/raw/main/images/erckert.jpg"> |<img width=120/ src="https://github.com/erckert/EV-embeddings/raw/main/images/rost.jpg">|


### License

The pretrained models and provided code are released under terms of the [GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)

### Citation

If you use this code or our pretrained models for your publication, please cite the original paper:
```
@ARTICLE
{EV_embeddings,
author={Erckert, Kyra and Rost, Burkhard},
title={Assessing the role of evolutionary information for enhancing protein language model embeddings},
journal={Scientific Reports},
year={2024},
month={Sep},
day={05},
volume={14},
number={1},
pages={20692},
doi={10.1038/s41598-024-71783-8}}
```