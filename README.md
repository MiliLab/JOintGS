# JOintGS: Joint Optimization of Cameras, Bodies and 3D Gaussians for In-the-Wild Monocular Reconstruction

This repository is a reference implementation for JOintGS, a unified framework that jointly optimizes camera extrinsics, human poses, and 3D Gaussian representations to achieve robust, high-fidelity, and animatable 3D human avatar reconstruction from unconstrained monocular videos with coarse initialization.



[[Paper](https://arxiv.org/abs/???)]


> [**JOintGS**](https://arxiv.org/abs/???),            
> [Zihan Lou](???), 
> [Jinlong Fan](???), 
> [Jing Zhang](???),   


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

# Preparing the datasets and models

## Datasets
- Download the [SMPL](https://smpl.is.tue.mpg.de/) neutral body model.
- Download [NeuMan](https://docs-assets.developer.apple.com/ml-research/datasets/neuman/dataset.zip) dataset.
- Download [EMDB](https://emdb.ait.ethz.ch/) (Ethical Multi-Device Body) dataset.

## Pre-Process
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

