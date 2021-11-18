# Image Enhancement

This is the source code for image enhancement tasks. Currently, it supports rain removal in daytime conditions.


# Installation

### Requirements

- Anaconda3 
- Python 3.9
- CUDA 11.1
- PyTorch 1.10
- Code has been tested on Ubuntu 20.04 / Windows 10

### Setup Environment

- Download the whole source code.
- Goto setup folder `cd image_enhancement/setup`
- Create the Anaconda environment: `conda env create -f mlkit.yml`


# Training

- Download the training data from: [download](https://o365skku-my.sharepoint.com/:u:/g/personal/phlong_o365_skku_edu/ETZ4XCf9oxhEvfhrchrXXZwBecAZaP1YFBBzrGwQlwM5Kw?e=TtQfeL) (~7 GB).
- Extract the data to `image_enhancement/data`. 
  - It should be located at: `image_enhancement/data/rain`
- Run the training scripts: `python image_enhancement/exps/run/train.py`


# Inference

- If you have retrained the model, find the best weight from:
  `image_enhancement/exps/checkpoints/mprnet/mprnet_rain/<version>/weights
  /best...ckpt`
- Copy the best weight to `image_enhancement/models_zoo`. Rename it as: 
  `mprnet_rain_version_0.ckpt`
- Run the inference scripts: `python image_enhancement/exps/run/infer.py`
