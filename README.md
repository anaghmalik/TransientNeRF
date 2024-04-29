# Transient Neural Radiance Fields for Lidar View Synthesis and 3D Reconstruction
### [Project Page](http://www.anaghmalik.com/TransientNeRF/) 

## Installation

1. To create a virtual environment use these commands
```
python3 -m venv venv
. venv/bin/activate
```
2. Additionally please install PyTorch, we tested on `torch1.12.1+cu116`

```
# torch 1.12.1+cu116
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

3. Install the requirements file with 

```
pip install -r requirements.txt
```

## Dataset 

A README of the dataset can be found in the `loaders` folder. 

Datasets can also be downloaded using the `download_datasets.py` script. With flags `--scenes o1 o2 o3`, replacing `o1`, `o2` and `o3` with scenes you want to download. You can use shorthand `all`, `captured` or `simulated` or otherwise specify scenes by their names.

Datasets can also be downloaded from [Dropbox](https://www.dropbox.com/scl/fo/02hsk2e686mkjwziyofzt/AN9Op5vDidmS6roxN3Ho5mE?rlkey=op6qgnbrde2jcjzp2g2hw803a&st=oemq9pk6&dl=0).

Please unzip it and have it in the root directry. If you want to install it elsewhere, please manually change the directories in the config.

## Training

You can train TransientNeRF by specifying a config and view number of the scene you want to train on, for example for lego two views, you would:

```
python train.py -c="./configs/train/simulated/lego_two_views.ini"
```

To see the summary during training, run the following
```
tensorboard --logdir=./results/ --port=6006
```


You can then evaluate the same scene (to get quantitative and image results) with
```
python eval.py -c="./configs/train/captured/cinema_two_views.ini" -tc="./configs/test/simulated/lego_quantitative.ini" --checkpoint_dir=<trained model directory root>
```


## Files 
- `train.py` main training script containing training loop
- `utils.py` contains rendering function `render_transient`, which calls the occgrid to generate sample points and then samples them 
- `misc/transient_volrend.py` called by utils contains the rendering code i.e. the whole image formation model, including convolution with pulse in `mapping_dist_to_bin_mitsuba`


## Changes 

I have realised that the models for the paper (the captured ones) were a bit undertrained (150k iterations), the configs thus train for longer than suggested in the paper (500k iterations). The difference is mainly important for the 5 views case, where PSNR increases by ~3/4dB. 

## Citation

```
@inproceedings{malik2023transient,
  title = {Transient Neural Radiance Fields for Lidar View Synthesis and 3D Reconstruction}, 
  author = {Anagh Malik and Parsa Mirdehghan and Sotiris Nousias and Kiriakos N. Kutulakos and David B. Lindell},
  journal = {NeurIPS},
  year = {2023}
}
```

## Acknowledgments

We thank [NerfAcc](https://www.nerfacc.com/) for their implementation of Instant-NGP. 
