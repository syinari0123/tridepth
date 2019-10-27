# TriDepth: Triangular Patch-based Deep Depth Prediction
This is an official implementation of "TriDepth: Triangular Patch-based Deep Depth Prediction" presented in [ICCV Deep Learning for Visual SLAM Workshop 2019 (oral)](http://www.visualslam.ai/). 
See the [project webpage](https://meshdepth.github.io/) and [demo video]() for more details.

<p align="center">
    <img src='https://drive.google.com/uc?export=view&id=1SUQUD4fJAIdIWXisrN8-HpNqIqvcVcTn' width=90%/></a>
</p>


## Environments

This code was developed and tested with Python 3.6.8 and PyTorch 1.1.0 on Ubuntu 16.04 LTS machine (CUDA 10.1).

* [Triangle](https://github.com/drufat/triangle) (for 2D mesh extraction)
* [Autotrace](https://github.com/autotrace/autotrace) (for edge vectorization)
* python3-tk
* Some pip libraries (in requirements.txt)

``` 
# Clone this repository with submodule
git clone --recursive git@github.com:syinari0123/tridepth.git # This repository

# Install Triangle (version==20190115.3)
cd thirdparty/triangle
python setup.py install

# Install Autotrace
sudo apt-get install autotrace

# Some libraries
sudo apt-get install python3-tk
pip install -r requirements.txt
```
* [torch-scatter](https://github.com/rusty1s/pytorch_scatter)==1.3.1

``` 
python -c "import torch; print(torch.__version__)"
>>> 1.1.0

echo $PATH
>>> /usr/local/cuda/bin:...

echo $CPATH
>>> /usr/local/cuda/include:...

# Then run:
pip install torch-scatter==1.3.1
```
* [PyMesh](https://pymesh.readthedocs.io/en/latest/index.html)==0.2.1


## Installation

Some parts of this code use the CUDA implementation from [Neural Mesh Renderer](https://github.com/daniilidis-group/neural_renderer) (PyTorch version by Nikos Kolotouros).
So you need to build it with following commands.

``` 
cd tridepth/renderer/cuda/
python setup.py install
```

## Dataset

Download the preprocessed [NYU Depth v2](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) dataset in HDF5 formats (from [Sparse-to-Dense](https://github.com/fangchangma/sparse-to-dense.pytorch) by Fangchang et al.).
The NYU dataset requires 32G of storage space.

```
mkdir data && cd data
wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz
tar -xvf nyudepthv2.tar.gz && rm -f nyudepthv2.tar.gz
```

## Training & evaluation

```
CUDA_VISIBLE_DEVICES=$EMPTY_GPU python train.py \
    --theme "train" \
    --log-path "log" \
    --data-path $DATA_DIR \
    --model-type "upconv" \
    --lr 1e-3 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --batchsize 8 \
    --workers 4 \
    --nepoch 30 \
    --print-freq 100 \
    --img-print-freq 1000 \
    --print-progress "true"
```
DATA_DIR is a dataset path (Default: data/nyudepthv2).


## Inference

```
CUDA_VISIBLE_DEVICES=$EMPTY_GPU python inference_nyu.py \
    --data-path $DATA_DIR \
    --model-type "upconv" \
    --pretrained-path $MODEL_PATH \
    --print-progress "true" \
    --rep-type "patch_cloud" \       # Choose representation type from ["patch_cloud", "mesh"]
    --result-path "result"
```
MODEL_PATH is a file path to the pretrained model (Default: pretrained/weight_upconv.pth).


## Comparison with other representations
See [comparison](https://github.com/syinari0123/tridepth/tree/master/comparison) directory.


## Pretrained model
You can download our pretrained model('weight_upconv.pth') with the following command.
```
cd pretrained
./download.sh
```


## FAQ
**What is the difference between [meshdepth](https://arxiv.org/abs/1905.01312) and tridepth?**

Same. We rename our representation from "disconnected mesh" (in [meshdepth paper](https://arxiv.org/abs/1905.01312)) to "triangular-patch-cloud" (in tridepth paper).


**Why does your result & score looks better than the result in your ICCVW paper?**

We update our implementation and re-evaluate our method after the paper submission. 
We'll submit an improved version of our paper to arXiv in near future.


## Citation

If you use our code or method in your work, please cite our paper!

```
@misc{kaneko19tridepth,
  Author = {Masaya Kaneko and Ken Sakurada and Kiyoharu Aizawa},
  Title = {TriDepth: Triangular Patch-based Deep Depth Prediction},
  Booktitle = {ICCV Deep Learning for Visual SLAM Workshop},
  Year = {2019},
}
```
