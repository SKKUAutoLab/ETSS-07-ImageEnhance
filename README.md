<div align="center">
	<h1 align="center">🐈 MON</h1>
</div>

`🐈 mon` is a research **[Monorepo](https://github.com/matakanobu/python-monorepo/tree/main)** for computer vision, built using [Python](https://www.python.org/) and [PyTorch](https://pytorch.org/). 

<details>
  <summary></summary>
  Yes, you are right. It's named after my cat 🐈 (Mon).
</details>

## Installation

```shell
git clone https://phlong3105@github.com/phlong3105/mon
cd mon

sudo chmod +x install.sh
./install.sh
```

The code is fully compatible with [PyTorch](https://pytorch.org/) >= 2.0.

## Repo Structure

`🐈 mon` is a **[Monorepo](https://github.com/matakanobu/python-monorepo/tree/main)** with multiple projects (``projects``) and shared common libs (``shared``).

<details>
  <summary>Directory Structure</summary>

  ```text
  mon 
  |__ docs                    # Documentation.
  |__ projects                # Project-specific code.
  |   |__ project_A           # Example project.
  |   |   |__ config          # Configuration files for experiments.
  |   |   |__ data            # Data related files.
  |   |   |__ docker          # Docker files for deployment. 
  |   |   |__ run             
  |   |   |__ notebook        # Notebooks for experiments.
  |   |   |__ tests           # Unit tests.
  |   |   |__ tools           # Useful scripts.
  |   |   |__ project_A       # Project's code.
  |   |   |   |__ ...
  |   |   |__ .dockerignore   
  |   |   |__ .gitignore      
  |   |   |__ Makefile        
  |   |   |__ pyproject.toml  
  |   |   |__ README.md       
  |   |   |__ ...             
  |   |__ ...                 
  |                          
  |__ shared                  # Common code shared across multiple projects is located.
  |   |__ mon                 # My main package
  |   |   |__ mon
  |   |   |   |__ core        # Base functionality for other packages.
  |   |   |   |__ datasets    # Dataset package.
  |   |   |   |__ vision      # Computer vision package.
  |   |   |   |__ ...      
  |   |   |__ pyproject.toml             
  |   |__ ...                 
  |                          
  |__ setup                   # Installation assets.
  |__ tools                   # Useful tools and scripts.
  |__ zoo                     # Model zoo (i.e., pre-trained models).
  |__ .gitignore               
  |__ .gitmodules             
  |__ install.sh              # Installation script.
  |__ LICENSE                 
  |__ mkdocs.yaml             # mkdocs setup.
  |__ pyproject.toml          $ Root configuration file. 
  |__ README.md               # Readme file.
  ```
</details>

## Cite
If you find our work useful, please consider citing the following:
```text
@misc{Pham2022,  
    author       = {Long Hoang Pham, Duong Nguyen-Ngoc Tran, Quoc Pham-Nam Ho},  
    title        = {🐈 mon},  
    publisher    = {GitHub},
    journal      = {GitHub repository},
    howpublished = {https://github.com/phlong3105/mon},
    year         = {2022},
}
```

## Contact
If you have any questions, feel free to contact `Long H. Pham` ([longpham3105@gmail.com](longpham3105@gmail.com) or [phlong@skku.edu](phlong@skku.edu))
