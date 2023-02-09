# EV-embeddings

EV-embeddings provides the trained models from our paper XXXX. The used dataset is available at XXXX. 

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
{XXXXXXX,
author={Erckert, Kyra and Rost, Burkhard},
journal={XXXXXX},
title={Exploring the Evolutionary Information Encoded in Protein Language Model Embeddings},
year={2023},
volume={},
number={},
pages={XXXXXXXX},
doi={XXXXXXXXX}}
```