from datetime import datetime
import random
from typing import Optional
import ast
import configargparse
import os
import numpy as np
import torch
from loaders.utils import Rays, namedtuple_map
from nerfacc.estimators.occ_grid import OccGridEstimator
from nerfacc.grid import ray_aabb_intersect, traverse_grids
from misc.transient_volrend import ( 
rendering_transient_single_path)

from torch.utils.tensorboard import SummaryWriter
import shutil


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def render_transient(
    # scene
    radiance_field: torch.nn.Module,
    occupancy_grid: OccGridEstimator,
    rays: Rays,
    # rendering options
    near_plane = 0,
    far_plane = 2**15,
    render_step_size: float = 1e-3,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    # only useful for dnerf
    chunk = 8192*128, 
    use_normals = False, 
    args = None
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        n_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([n_rays] + list(r.shape[2:])), rays
        )
    else:
        n_rays, _ = rays_shape

    results = []
    
    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        rgbs, sigmas = radiance_field(positions, t_dirs)
        return rgbs, sigmas.squeeze(-1)

    
    for i in range(0, n_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        
        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = chunk_rays.origins[ray_indices]
            t_dirs = chunk_rays.viewdirs[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            sigmas = radiance_field.query_density(positions)
            return sigmas.squeeze(-1)
        
        ray_indices, t_starts, t_ends = occupancy_grid.sampling(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        rgb, opacity, depth, depth_variance, comp_weights, raw_rgbs = rendering_transient_single_path(
            t_starts=t_starts,
            t_ends=t_ends,
            ray_indices=ray_indices,
            n_rays=n_rays,
            # radiance field
            rgb_sigma_fn=rgb_sigma_fn,
            # rendering options
            render_bkgd=None,
            args = args
        )

        chunk_results_single = [rgb, opacity, depth, depth_variance, comp_weights, raw_rgbs, len(t_starts)]
        results.append(chunk_results_single)

    colors_single, opacities_single, depths_single, depths_variance, densities, raw_rgbs, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    
    normals_loss = 0 

    colors = torch.reshape(colors_single, (-1, args.n_bins, 3))

    return {'colors': colors.view((*rays_shape[:-1], -1)),
            'opacities': opacities_single.view((*rays_shape[:-1], -1)),
            'depths': depths_single.view((*rays_shape[:-1], -1)),
            'depths_variance' : depths_variance.view((*rays_shape[:-1], -1)),
            'n_rendering_samples': sum(n_rendering_samples),
            'normals_loss': normals_loss,
            'comp_weights': comp_weights, 
            "raw_rgbs":raw_rgbs}


def parse_list(arg):
    try:
        return ast.literal_eval(arg)
    except (SyntaxError, ValueError):
        raise configargparse.ArgumentTypeError(f"Invalid list format: {arg}")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


def load_args(eval = False, parser= None):
    # parser = configargparse.ArgumentParser()
    if not eval:
        parser = configargparse.ArgumentParser()

    parser.add('-c', '--my-config', 
        is_config_file=True, 
        default="./configs/train/simulated/bench_two_views.ini", 
        help='Path to config file.'
    )
    parser.add_argument(
        '--exp_name', 
        type=str, 
        default='lego_two_views', 
        help='Experiment name.'
    )
    parser.add_argument(
        "--aabb",
        nargs='+',
        type = lambda s: ast.literal_eval(s),
        default="[-1.5,-1.5,-1.5,1.5,1.5, 1.5]",
        help="AABB size.",
    )
    parser.add_argument(
        "--test_chunk_size",
        type=int,
        default=512,
        help="Test chunk size..",
    )
    parser.add_argument(
        "--num_rays_per_batch",
        type=int,
        default=512,
        help="Number of rays per batch.",
    )
    parser.add_argument(
        "--starting_rays_per_pixel",
        type=int,
        default=1,
        help="Starting rays per pixels.",
    )
    parser.add_argument(
        "--tfilter_sigma",
        type=int,
        default=3,
        help="Temporal filter standard deviation.",
    )
    parser.add_argument(
        "--space_carving",
        type=float,
        default=7*1e-3,
        help="Space carvig regaularization strength.",
    )
    # parser.add_argument(
    #     "--dataset_scale",
    #             type=int,
    #             default=46,
    #     help="Scale for all transient images.",
    # )
    parser.add_argument(
        "--rfilter_sigma",
        type=float,
        default=0.15,
        help="Spatial filter standard deviation.",
    )
    parser.add_argument(
        "--exposure_time",
        type=float,
        default=0.01,
        help="Exposure length per bin in meters.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--steps_til_checkpoint",
        type=int,
        default=20000,
        help="Steps per checkpoint.",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=1200,
        help="Number of bins.",
    )
    parser.add_argument(
        "--img_shape",
        type=int,
        default=512,
        help="Shape of training image.",
    )
    parser.add_argument(
        "--sample_as_per_distribution",
        action="store_true",
        help="Sample as per distribution or uniformly.",
    )
    parser.add_argument(
        "--render_n_samples",
        type=int,
        default=4096,
        help="Num samples per ray.",
    )
    parser.add_argument(
        "--exp",
        type=str2bool,
        default="true",
        help="Use double exp.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=300000,
        help="Max number of steps.",
    )
    parser.add_argument(
        "--near_plane",
        type=int,
        default=0,
        help="Near plane value.",
    )
    parser.add_argument(
        "--alpha_thre",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--far_plane",
        type=int,
        default=2**15,
        help="Far plane value.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="simulated",
        choices=["captured", "simulated"],
        help="Dataset being trained, captured or simulated.",
    )
    parser.add_argument(
        "--occ_thre",
                type=float,
                default=0.01,
        help="Occupancy threshold",
    )
    parser.add_argument(
        "--thold_warmup",
                type=int,
                default=-1,
        help="Warmup period for the occupancy threshold.",
    )
    parser.add_argument(
        "--final",
        type=str2bool,
        default="false",
        help="If final version or debug mode (creates dated folder).",
    )
    parser.add_argument(
        "--grid_resolution",
                type=int,
                default=128,
        help="Occgrid resolution.",
    )
    parser.add_argument(
        "--grid_nlvl",
                type=int,
                default=1,
        help="Number of grid levels.",
    )
    parser.add_argument(
        "--outpath",
                type=str,
                default="./results",
        help="Path to results folder.",
    )
    parser.add_argument(
        "--data_root_fp",
                type=str,
                default="./data/lego_data/lego_jsons/two_views",
        help="Root of dataset directory (where the transforms directory is).",
    )
    parser.add_argument(
        "--pulse_path",
                type=str,
                default="./datasets/pulse_low_flux.mat",
        help="Path to pulse for captured dataset.",
    )
    parser.add_argument(
        "--intrinsics",
                type=str,
                default="./data/lego_data/lego_jsons/two_views",
        help="Path to intrinsics for captured dataset",
    )
    parser.add_argument(
        "--pixels_to_plot",
        nargs='+',
        type = lambda s: ast.literal_eval(s),
        default=[(16, 16), (20, 16), (28, 25)],
        help="Pixels used for plotting in the summary.",
    )
    parser.add_argument(
        "--img_scale",
                type=int,
                default=100,
        help="Image scale used in summary.",
    )
    parser.add_argument(
        "--num_views",
                type=int,
                default=2,
        help="Number of views trained on.",
    )
    parser.add_argument(
        "--img_shape_test",
                type=int,
                default=64,
        help="Test image shape.",
    )
    parser.add_argument(
        "--seed",
                type=int,
                default=42,
        help="Seed.",
    )
    parser.add_argument(
        "--device",
                type=str,
                default="cuda:7",
        help="Device.",
    )
    parser.add_argument("--cone_angle", type=float, default=0.0)
    args = parser.parse_args()
    return args

def make_save_folder(args):
    now = datetime.now()
    now = now.strftime("%m-%d_%H:%M:%S")
    exp_name = args.exp_name + "_" + now
    outpath = os.path.join(args.outpath, exp_name)
    os.mkdir(outpath)
    shutil.copy(args.my_config, os.path.join(outpath, "params.txt"))
    
    with open(os.path.join(outpath, "params_full.txt"), "w") as out_file:
        param_list = []    
        for key, value in vars(args).items():
            if type(value) == list:
                value = [eval(f"{x}") for x in value]
            elif type(value) != int and type(value) != float:
                value = str(value)
                value = f"'{value}'"
            param_list.append("%s= %s" % (key, value))
         
        out_file.write('\n'.join(param_list))
    return outpath

def make_save_folder_final(args, optimizer, scheduler, radiance_field, occupancy_grid):
    outpath = os.path.join(args.outpath, args.exp_name)

    if not os.path.isdir(outpath):

        os.mkdir(outpath)
        with open(os.path.join(outpath, "params_full.txt"), "w") as out_file:
            param_list = []    
            for key, value in vars(args).items():
                if type(value) != int and type(value) != float:
                    value = str(value)
                    value = f"'{value}'"
                param_list.append("%s= %s" % (key, value))
            
            out_file.write('\n'.join(param_list))
        step = 0
        writer = SummaryWriter(log_dir=outpath)

    else:
        ckpt_path_var = os.path.join(outpath, 'variables.pth')
        ckpt = torch.load(ckpt_path_var)
        step = ckpt['step']

        ckpt_path_rf = os.path.join(outpath, 'radiance_field_%04d.pth' % (step))
        ckpt_path_oc = os.path.join(outpath, 'occupancy_grid_%04d.pth' % (step))
        ckpt_path_opt = os.path.join(outpath, 'optimizer_%04d.pth' % (step))
        ckpt_path_sch = os.path.join(outpath, 'scheduler_%04d.pth' % (step))

        ckpt = torch.load(ckpt_path_rf, map_location=args.device)
        radiance_field.load_state_dict(ckpt)
        radiance_field = radiance_field.to(args.device)

        ckpt = torch.load(ckpt_path_oc, map_location=args.device)
        occupancy_grid.load_state_dict(ckpt)
        occupancy_grid = occupancy_grid.to(args.device)

        ckpt = torch.load(ckpt_path_opt)
        optimizer.load_state_dict(ckpt)

        ckpt = torch.load(ckpt_path_sch)
        scheduler.load_state_dict(ckpt)
        print(f"previous checkpoint loaded; current step: {step}")
        writer = SummaryWriter(log_dir=outpath)
    
    return writer, step, outpath
    
if __name__=="__main__":
    pass