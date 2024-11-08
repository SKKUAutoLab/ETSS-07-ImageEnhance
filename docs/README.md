# ETSS-07-ImageEnhance

![](https://img.shields.io/github/downloads/SKKU-AutoLab-VSW/ETSS-07-ImageEnhance/total.svg)

## Installation

```shell
git clone https://phlong3105@github.com/phlong3105/mon
cd mon
chmod +x install.sh

# On Linux
conda init bash
bash -i install.sh

# On Mac
conda init zsh
zsh -i install.sh
```

The code is fully compatible with [PyTorch](https://pytorch.org/) >= 2.0.

## Directory Organization

```text
code
 |_ mon
     |_ data                 # Default location to store working datasets.
     |_ docs                 # Documentation.
     |_ env                  # Environment variables.
     |_ project              # Project-specific code.
     |_ src                  # Source code.
     |   |_ mon              # Python code.
     |       |_ config       # Configuration functionality.
     |       |_ core         # Base functionality for other packages.
     |       |_ data         # Data processing package.
     |       |_ nn           # Machine learning package.
     |       |_ vision       # Computer vision package.
     |_ tools                # Tools.
     |_ zoo                  # Model zoo.
     |_ .gitignore           # 
     |_ install.sh           # Installation script.
     |_ LICENSE              #
     |_ mkdocs.yaml          # mkdocs setup.
     |_ pyproject.toml       # 
     |_ README.md            # Github Readme.
```

## Cite
If you find our work useful, please cite the following:
```text
@misc{Pham2022,  
    author       = {Long Hoang Pham, Duong Nguyen-Ngoc Tran, Quoc Pham-Nam Ho},  
    title        = {🐈 mon},  
    publisher    = {GitHub},
    journal      = {GitHub repository},
    howpublished = {https://github.com/phlong3105/mon},
    year         = {2024},
}
```

## Contact
If you have any questions, feel free to contact `Long H. Pham` ([longpham3105@gmail.com](longpham3105@gmail.com) or [phlong@skku.edu](phlong@skku.edu))
