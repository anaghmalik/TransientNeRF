1. The captured scenes include: carving, boots, baskets, cinema, chef. 
The simulated scenes include lego, chair, ficus, hotdog, bench. 
These have different structures and loaders. 

In the root of the dataset folder you can also find the `intrinsics.npy` which is the set of captured rays and parameters used in training and `pulse_low_flux.mat` which is the calibrated laser pulse.


2. Training transform files.

For the *captured* scenes the training transforms are all named `transforms_train.json` and can be found under `<scene_name>/final_cams/<num_views>/transforms_train.json`. 

For the *simulated* scenes the training transforms are all named `transforms_train_v{i}.json` where `i` is the number of views (2, 3, 5). and can be found under `<scene_name>/transforms_train_v{i}.json`. 


3. Test transform files.

For the *captured* scenes the test transforms can be found under `<scene_name>/final_cams/test_jsons/transforms_test.json`. 

For the *simulated* scenes the test transforms can be found under `<scene_name>/transforms_test_final.json`. 


4. Downloading.

- You can download the data yourself either through the Dropbox download button, or by right-clicking a folder and selecting copy link address, then
```
wget "copied link" 
```
will start a download of the folder. 

5. (!!!) Using the dataset.

To use the dataset alongside its transforms please look at the loader in `loaders/loader_captured.py` in the [GitHub repository](https://github.com/anaghmalik/TransientNeRF). Most importantly to use the temporal *captured* data, you will have to resample the transient using the shift given in the `intrinsics.npy` file:

```
img_shape = (512, 512)
exposure_time= 299792458*4e-12
n_bins = 1500

x = (torch.arange(img_shape[0], device="cpu")-img_shape[0]//2+0.5)/(img_shape[0]//2-0.5)
y = (torch.arange(img_shape[0], device="cpu")-img_shape[0]//2+0.5)/(img_shape[0]//2-0.5)
z = torch.arange(n_bins*2, device="cpu").float()
X, Y, Z = torch.meshgrid(x, y, z, indexing="xy")
Z = Z*exposure_time/2
Z = Z - shift[0]
Z = Z*2/exposure_time
Z = (Z-n_bins*2//2+0.5)/(n_bins*2//2-0.5)
grid = torch.stack((Z, X, Y), dim=-1)[None, ...]
del X
del Y
del Z

rgb = torch.Tensor(rgb)[..., :3000].float().cpu()
rgb = torch.nn.functional.grid_sample(rgb[None, None, ...], grid, align_corners=True).squeeze().cpu()

```

where `shift` is the value from the `intrinsics.npy` file and `rgb` is the original loaded transient. 
