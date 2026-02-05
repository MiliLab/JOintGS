
<div align="center">


# JOintGS: Joint Optimization of Cameras, Bodies and 3D Gaussians for In-the-Wild Monocular Reconstruction

[![Paper](https://img.shields.io/badge/Under_Review-b31b1b.svg)](http://arxiv.org/abs/2602.04317)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-FFD21E.svg)](https://huggingface.co/louzihan/JOintGS)
[![License](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

**JOintGS jointly optimizes cameras, human poses, and 3DGS for robust, animatable 3D human avatar reconstruction from monocular video with coarse initialization.**

[**Introduction**](#introduction) | [**Getting Started**](#getting-started) | [**Dataset & Pre-Process**](#dataset) | [**Evaluation**](#evaluation) | [**Training**](#training) | [**License**](#license) | [**Citation**](#citation)

</div>






# Introduction

<p float="center">
  <img src="assets/JOintGS_Framework.png" width="100%" />
</p>



# Getting Started

We tested our system on Ubuntu 22.04.5 LTS using a CUDA 13.0 compatible GPU

- Clone our repo:
```
git clone https://github.com/MiliLab/JOintGS
```

- Run the setup script to create a conda environment and install the required packages.
```
source scripts/conda_setup.sh
```


# Dataset
- Download the [SMPL](https://smpl.is.tue.mpg.de/) neutral body model.
- Download [NeuMan](https://docs-assets.developer.apple.com/ml-research/datasets/neuman/dataset.zip) dataset.
- Download [EMDB](https://emdb.ait.ethz.ch/) (Ethical Multi-Device Body) dataset.

We recommend following the step-by-step instructions provided in `data/scripts/readme.md` to refine the datasets. These scripts handle essential tasks such as camera parameter extraction and SMPL fitting alignment.

After following the above steps, you should obtain a folder structure similar to this:

```
data/
├── smpl
│   └── SMPL_NEUTRAL.pkl
├── neuman
│   ├── bike
│   └── ...
└── emdb
    ├── P0_08_outdoor_remove_jacket
    │   ├── images
    │   ├── masks
    │   ├── sparse
    │   └── sam3db
```


# Evaluation

## 💾  Pre-trained Checkpoints
You can download our pre-trained model checkpoints directly from Hugging Face Hub, allowing you to bypass the training process.
All checkpoints are hosted at the following Hugging Face repository. **Please visit this URL to download the files:**
[**Hugging Face Repository: louzihan/JOintGS**](https://huggingface.co/louzihan/JOintGS)

After following the above steps, you should obtain a folder structure similar to this:
```
checkpoints/
├── neuman
│   ├── bike
│   ├── citron
│   ├── jogging
│   ├── lab
│   ├── parkinglot
│   └── seattle
```

# Training

# Citation

## 📜 License

This project is intended for **academic research purposes only**.

* **Source Code**: The software in this repository is licensed under the [MIT License](LICENSE).
* **Model Weights**: The pre-trained checkpoints are released under the [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
* **Third-party Data & Models**:
    * **SMPL**: The SMPL model is subject to the [SMPL Model License](https://smpl.is.tue.mpg.de/modellicense.html).
    * **Datasets**: Images and annotations from [NeuMan](https://www.apple.com/kl/ml-research/datasets/neuman/) and [EMDB](https://emdb.ait.ethz.ch/) adhere to their original licensing terms (strictly for non-commercial research).

By downloading or using these materials, you agree to comply with the respective licenses of all components.


