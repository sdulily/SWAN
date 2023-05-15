---
typora-root-url: images
---

# SWAN 

## Preparation

### 1. Create virtual environment (optional)

All code was developed and tested on Ubuntu 18.04 with Python 3.10.9 (Anaconda) and PyTorch 1.12.1.

```bash
$ conda create -n SWAN python=3.10.9
$ conda activate SWAN
```

### 2. Download datasets

You can download our dataset from https://pan.baidu.com/s/1oXFVv1tZCkUSoH2pSxWFSA with password `nqhi`

## Run codes

### Train models

启动http server：

```
$ python -m visdom.server
```

train：

```
$ python train.py
```

### Test models

```bash
$ python test.py
```

## Results

### 1. Qualitative Results of Different Models (seen)

![](/seen.png)

### 2. Qualitative Results of Different Models (unseen)

![](/unseen.png)

## Contact

If you encounter any problems, please contact us.

## Reference

Our project borrows some source files from ChipGAN(https://github.com/PKU-IMRE/ChipGAN). 