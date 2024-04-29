import numpy as np 
import imageio 
import json 
import torch 
import sys 
import os
import configargparse
import ast
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loaders.utils import Rays
from utils import str2bool, load_args

def calc_psnr(img1, img2):
    # Calculate the mean squared error
    mse = np.mean((img1 - img2) ** 2)
    # Calculate the maximum possible pixel value (for data scaled between 0 and 1)
    max_pixel = 1.0
    # Calculate the PSNR
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

def get_rays(img_shape, c2w, K, device):
    OPENGL_CAMERA = True
    x, y = torch.meshgrid(
                torch.arange(img_shape, device=device),
                torch.arange(img_shape, device=device),
                indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()
    
    c2w = c2w.repeat(img_shape**2, 1, 1)
    camera_dirs = torch.nn.functional.pad(
            torch.stack(
                [
                    (x - K[0, 2] + 0.5) / K[0, 0],
                    (y - K[1, 2] + 0.5)
                    / K[1, 1]
                    * (-1.0 if OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
    directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
    origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
    viewdirs = directions / torch.linalg.norm(
        directions, dim=-1, keepdims=True
    )
    origins = torch.reshape(origins, (img_shape, img_shape, 3))
    viewdirs = torch.reshape(viewdirs, (img_shape, img_shape, 3))

    
    rays = Rays(origins=origins, viewdirs=viewdirs)

    return rays 


def read_json(json_path):
    f = open(json_path)
    positions = json.load(f)
    f.close()
    return positions

def generate_video(images, output_path, fps):
    # Determine the width and height of the images
    writer = imageio.get_writer(output_path, fps=fps)
    for image in images:
        writer.append_data(image)
    writer.close()

def calc_iou(rgb, gt_tran):
    intersection = np.minimum(rgb, gt_tran)
    union = np.maximum(rgb, gt_tran)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def load_eval_args():
    parser = configargparse.ArgumentParser()
    parser.add('-tc', '--test_config', 
        is_config_file=True, 
        default="./configs/test/captured/cinema_quantitative.ini", 
        help='Path to config file.'
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="cinema",
        # choices=[
        #     # nerf transient
        #     "lego",
        #     "chair",
        #     "drums",
        #     "ficus",
        #     "hotdog",
        #     "bench",
        #     "boar",
        #     "benches"
        # ],
        help="scene to evaluate the models on",
    )
    parser.add_argument(
        "--rep_number",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--step",
        type=int,
        default=300000,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
    )
    parser.add_argument(
        "--test_folder_path",
        type=str,
        default="test2",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/scratch/ondemand28/anagh/tnerf_release/multiview_transient/results/cinema_two_views_04-18_02:10:32",
    )
    parser.add_argument(
        "--data_folder_path",
        type=str,
        default="./data",
    )
    args = load_args(eval=True, parser=parser)
    return args

num2words = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 
             6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 
             11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', 
             15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen', 19: 'nineteen'}

if __name__=="__main__":
    pass