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
