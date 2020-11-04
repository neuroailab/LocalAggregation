# Local Aggregation for Unsupervised Learning of Visual Embeddings
This repo implements the Local Aggregation (LA) algorithm on ImageNet and related transfer learning pipelines for both ImageNet and Places205.
Pytorch implementation of this algorithm is at [LocalAggregation-Pytorch](https://github.com/neuroailab/LocalAggregation-Pytorch).
This repo also includes a tensorflow implementation for the Instance Recognition (IR) task introduced in paper "Unsupervised Feature Learning via Non-Parametric Instance Discrimination".

# Pretrained Model
A Local-Aggregation pretrained ResNet-18 model can be found at [link](http://visualmaster-models.s3.amazonaws.com/la_orig/checkpoint-1901710.tar), though this model may not be as good as a fully trained model by this repo, as it's a slightly earlier checkpoint than the final one.

# Instructions for training

## Prerequisites
We have tested this repo under Ubuntu 16.04 with tensorflow version 1.9.0.
Training LA model requires `faiss==1.6.1`.

## Data preparation
Prepare the ImageNet data as the raw JPEG format used in pytorch ImageNet training (see [link](https://github.com/pytorch/vision/blob/master/torchvision/datasets/imagenet.py)).
Then run the following command:
```
python dataset_miscs/build_tfrs.py --save_dir /path/to/imagenet/tfrs --img_folder /path/to/imagenet/raw/folder
```

## Model training 
We provide implementations for LA trained AlexNet, VggNet, ResNet-18, and ResNet-50.
We provide commands for ResNet-18 training, while commands for other networks can be acquired through slightly modifying these commands after inspecting for `exp_configs/la_final.json`.
As LA algorithm requires training the model using IR algorithm for 10 epochs as a warm start, we first run the IR training using the following command:
```
python train.py --config exp_configs/la_final.json:res18_IR --image_dir /path/to/imagenet/tfrs --gpu [your gpu number] --cache_dir /path/to/model/save/folder
```
Then run the following command to do the LA training:
```
python train.py --config exp_configs/la_final.json:res18_LA --image_dir /path/to/imagenet/tfrs --gpu [your gpu number] --cache_dir /path/to/model/save/folder
```

### Code reading

For your convenience, the most important function you want to look at is function `build_targets` in script `model/instance_model.py`.

## Transfer learning to ImageNet
After finishing the LA training, run the following command to do the transfer learning to ImageNet:
```
python train_transfer.py --config exp_configs/la_trans_final.json:trans_res18_LA --image_dir /path/to/imagenet/tfrs --gpu [your gpu number] --cache_dir /path/to/model/save/folder
```

## Transfer learning to Places205
Generate the tfrecords for Places205 using the following command:
```
python dataset_miscs/build_tfrs_places.py --out_dir /path/to/places205/tfrs --csv_folder /path/to/places205/csvs --base_dir /path/to/places205/raw/folder --run
```
`/path/to/places205/csvs` should include `train_places205.csv` and `val_places205.csv` for Places205.
`/path/to/places205/raw/folder` should include the raw Places205 images such as `/path/to/places205/raw/folder/data/vision/torralba/deeplearning/images256/a/abbey/gsun_0003586c3eedd97457b2d729ebfe18b5.jpg`

Then, run this command for transfer learning:
```
python train_transfer.py --config exp_configs/la_plc_trans_final.json:plc_trans_res18_LA --image_dir /path/to/imagenet/tfrs --gpu [your gpu number] --cache_dir /path/to/model/save/folder
```

## Multi-GPU training
Unfortunately, this implementation does not support an efficient multi-gpu training, which is non-trivial in tensorflow.
Instead, we provide another implementation using [TFUtils](https://github.com/neuroailab/tfutils), which supports multi-gpu training but requires installing TFUtils.
After installing TFUtils, run the same training commands using `train_tfutils.py` and `train_transfer_tfutils.py` with multi-gpu argument such as `--gpu a,b,c,d`, where `a,b,c,d` are the gpu numbers used.
