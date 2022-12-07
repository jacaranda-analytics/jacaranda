Jacaranda 
========================

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Jacaranda ](#jacaranda)
- [Description](#description)
- [Examples](#examples)
- [Installation](#installation)
    - [GitHub](#github)
    - [Pip](#pip)
- [Examples](#examples-1)
- [License](#license)

<!-- markdown-toc end -->


# Description 

Jacaranda is a wrapper package around several Data Science and Machine Learning
librarys,  such as 

- [PyTorch](https://pytorch.org)
- [XGboost](https://xgboost.readthedocs.io/en/stable/)

which creates an easy interface to interact, and automatically tune models produced 
by these libraries. 


# Examples 

Examples for using the Jacaranda API to tune the following list of models is available in the examples folder. 

- Autoencoder 
- Variational Autoencode 
- Xgboost decicion tree
- 1D Convolutional Neural Network 
- Multilayer Perceptron 


# Installation 

Currently, there are various ways this package can be installed. 
These include 

- GitHub 
- pip

## GitHub 

To install from GitHub there are two options, 
the first option is to clone the repository and do a local installation from the cloned directory. 

```sh
git clone git@github.com:jacaranda-analytics/jacaranda.git
cd jacaranda/ && pip install . 
```

The second option is to install from GitHub without first cloning the repository, 
to install the latest master branch, run the command. 

```sh
pip install https://github.com/jacaranda-analytics/jacaranda/archive/master.zip
```

## Pip 

To install through pip, simply run 

```python 
pip install jacaranda
```



# License 

- [MIT](LICENSE.md)
