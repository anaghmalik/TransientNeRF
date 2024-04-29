import os
from misc.summary import write_summary_histogram
import time
import numpy as np
import torch
import tqdm
from misc.transient_volrend import torch_laser_kernel
from radiance_fields.ngp import NGPRadianceField
from torch.utils.tensorboard import SummaryWriter
import os
import torch.multiprocessing as mp
from multiprocessing import Value
from ctypes import c_longlong
from torch.utils.data import DataLoader
from nerfacc.estimators.occ_grid import OccGridEstimator
import math 
import scipy.io as sio

from utils import (
    make_save_folder,
    make_save_folder_final,
    set_random_seed,
    load_args, 
    render_transient
    )

if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn') # or 'forkserver'

def run():
    torch.cuda.empty_cache()
    args = load_args()
    device = args.device
    set_random_seed(args.seed)
    
    outpath = os.path.join(args.outpath, args.exp_name)

    
    aabb = torch.tensor(args.aabb, dtype=torch.float32, device=device)
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}

    # setup the dataset
    rfilter_sigma = args.rfilter_sigma
    img_shape = (args.img_shape, args.img_shape)
    img_shape_test = (args.img_shape_test, args.img_shape_test)
    max_steps = args.max_steps
    sample_as_per_distribution = args.sample_as_per_distribution
    target_sample_batch_size = 1 << 16

    if args.version == "simulated":
        from loaders.loader_synthetic import SubjectLoaderTransient as SubjectLoader
        test_dataset_kwargs = {"img_shape": img_shape_test, "have_images": True, "n_bins": args.n_bins, "color_bkgd_aug": "black", "rfilter_sigma":rfilter_sigma, "sample_as_per_distribution":sample_as_per_distribution }
        train_dataset_kwargs = {"img_shape": img_shape, "n_bins": args.n_bins, "color_bkgd_aug": "black", "rfilter_sigma":rfilter_sigma, "sample_as_per_distribution":sample_as_per_distribution}


        train_dataset = SubjectLoader(
            root_fp=args.data_root_fp,
            subject_id=args.exp_name,
            split="train",
            num_rays=target_sample_batch_size // args.render_n_samples,
            **train_dataset_kwargs,
            num_views=args.num_views
        )

        train_dataset.camtoworlds = train_dataset.camtoworlds.to(device)
        train_dataset.K = train_dataset.K.to(device)

        test_dataset = SubjectLoader(
            root_fp=args.data_root_fp,
            subject_id=args.exp_name,
            split="test",
            num_rays=None,
            **test_dataset_kwargs,
        )

        if test_dataset_kwargs["have_images"]:
            test_dataset.images = test_dataset.images.to(device)
        test_dataset.camtoworlds = test_dataset.camtoworlds.to(device)
        test_dataset.K = test_dataset.K.to(device)
    else:
        from loaders.loader_captured import LearnRays, SubjectLoaderTransientReal as SubjectLoader
        params = np.load(args.intrinsics, allow_pickle=True)[()]
        shift = params['shift'].numpy()
        rays = params['rays']


        data_root_fp = args.data_root_fp
        test_dataset_kwargs = {"img_shape": (args.img_shape_test, args.img_shape_test), "have_images": True, "n_bins": args.n_bins, "color_bkgd_aug": "black", "rfilter_sigma":rfilter_sigma, "sample_as_per_distribution":sample_as_per_distribution, "shift":shift}
        train_dataset_kwargs = {"img_shape": img_shape, "n_bins": args.n_bins, "color_bkgd_aug": "black", "rfilter_sigma":rfilter_sigma, "sample_as_per_distribution":sample_as_per_distribution, "shift":shift}

        train_dataset = SubjectLoader(
            root_fp=data_root_fp,
            subject_id=args.exp_name,
            split="train",
            num_rays=target_sample_batch_size // args.render_n_samples,
            **train_dataset_kwargs
                    )

        train_dataset.camtoworlds = train_dataset.camtoworlds.to(device)
        train_dataset.K = LearnRays(rays, device=device, img_shape=img_shape)
        train_dataset.K = train_dataset.K.to(device)


        test_dataset = SubjectLoader(
            root_fp=args.data_root_fp,
            subject_id=args.exp_name,
            split="test",
            num_rays=None,
            **test_dataset_kwargs,
        )

        if test_dataset_kwargs["have_images"]:
            test_dataset.images = test_dataset.images.to(device)
        
        test_dataset.camtoworlds = test_dataset.camtoworlds.to(device)
        test_dataset.K = LearnRays(rays, device=device, img_shape=(args.img_shape_test, args.img_shape_test))
        test_dataset.K = test_dataset.K.to(device)
        
        laser_pulse_dic = sio.loadmat(args.pulse_path)['out'].squeeze()
        laser_pulse = laser_pulse_dic
        laser_pulse = (laser_pulse[::2] + laser_pulse[1::2])/2
        lidx = np.argmax(laser_pulse)
        loffset = 50
        laser = laser_pulse[lidx-loffset:lidx+loffset+1]
        laser = laser / laser.sum()
        laser = laser[::-1]
        laser = torch.tensor(laser.copy(), device=device).float()
        laser_kernel = torch_laser_kernel(laser, device=device)
        args.laser_kernel = laser_kernel



    # setup the scene bounding box.
    scene_aabb = torch.tensor(args.aabb, dtype=torch.float32, device=device)
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / args.render_n_samples
    ).item()

    # setup the radiance field we want to train.
    grad_scaler = torch.cuda.amp.GradScaler(2**10)
    radiance_field = NGPRadianceField(
        use_viewdirs=True,
        aabb=args.aabb,
        unbounded=False,
        radiance_activation=torch.exp,
        args = args
    ).to("cpu")
    
    radiance_field = radiance_field.to(device)
    optimizer = torch.optim.Adam(
        radiance_field.parameters(), lr=args.lr, eps=1e-15
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max_steps // 2, max_steps * 3 // 4, max_steps * 9 // 10],
        gamma=0.33,
    )
    occupancy_grid = OccGridEstimator(
        roi_aabb=aabb, resolution=args.grid_resolution, levels=args.grid_nlvl
    ).to(device)


    # Make save folder.
    if args.final:
        writer, step, outpath = make_save_folder_final(args, optimizer, scheduler, radiance_field, occupancy_grid)
        args.outpath = outpath
    else:
        outpath = make_save_folder(args)
        args.outpath = outpath
        writer = SummaryWriter(log_dir=outpath)
        step = 0


    pbar = tqdm.tqdm(total=args.max_steps)

    while True:
        pbar.update(1)
        
        if args.version == "simulated":
            if step%1000==0:
                if train_dataset.rep <30:
                    train_dataset.rep += 2

        radiance_field.train()
        
        i = torch.randint(0, len(train_dataset), (1,)).item()
        data = train_dataset[i]
        rays = data["rays"]
        pixs = torch.reshape(data["pixels"][:int(rays.origins.shape[0]/train_dataset.rep)], (-1, args.n_bins, 3))

        def occ_eval_fn(x):
            density = radiance_field.query_density(x)
            return density * render_step_size

        # Warmup occupancy threshold.
        if args.version == "captured":
            if step<10000:
                occ_thre = 1e-6
            else:
                occ_thre = 1e-3
        else:
            occ_thre = args.occ_thre
        
        occupancy_grid.update_every_n_steps(
            step=step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=occ_thre,
        )
        
        out = render_transient(
            radiance_field,
            occupancy_grid,
            rays,
            near_plane=args.near_plane,
            far_plane=args.far_plane,
            render_step_size=render_step_size,
            cone_angle=args.cone_angle,
            alpha_thre=args.alpha_thre,
            use_normals = False, 
            args = args
        )


        rgb, acc, n_rendering_samples, comp_weights = [out[key] for key in ['colors', 'opacities', 'n_rendering_samples', "comp_weights"]]
        del out

        if n_rendering_samples == 0:
            continue

        num_rays = args.num_rays_per_batch
        train_dataset.update_num_rays(num_rays)


        alive_ray_mask = acc.squeeze(-1)>0
        alive_ray_mask = alive_ray_mask.reshape( train_dataset.rep, -1)
        alive_ray_mask = alive_ray_mask.sum(0).bool()


        rgba = torch.reshape(rgb, (-1, args.n_bins, 3))*data["weights"][:, None, None]
        comp_weights = comp_weights[pixs.sum(-1).repeat(train_dataset.rep, 1)<1e-7].mean()
        rgb = torch.zeros((int(rgba.shape[0]/train_dataset.rep), args.n_bins, 3), device=device)
        index = torch.arange(int(rgba.shape[0]/train_dataset.rep), device=device).repeat(train_dataset.rep)[:, None, None].expand(-1, args.n_bins, 3)
        rgb.scatter_add_(0, index.type(torch.int64), rgba)


        pixs = torch.log(pixs + 1)
        rgb = torch.log(rgb + 1)
        loss = torch.nn.functional.l1_loss(rgb[alive_ray_mask], pixs[alive_ray_mask]) + comp_weights*args.space_carving


        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        optimizer.step()
        scheduler.step()

        writer.add_scalar('Loss/train', loss.detach().cpu().numpy(), step)

        if not step % args.steps_til_checkpoint:
            torch.save(radiance_field.state_dict(), os.path.join(outpath, 'radiance_field_%04d.pth' % (step)))
            torch.save(occupancy_grid.state_dict(), os.path.join(outpath, 'occupancy_grid_%04d.pth' % (step)))
            torch.save(optimizer.state_dict(), os.path.join(outpath, 'optimizer_%04d.pth' % (step)))
            torch.save(scheduler.state_dict(), os.path.join(outpath, 'scheduler_%04d.pth' % (step)))
            torch.save({'step': step, "rays_per_pixel":train_dataset.rep}, os.path.join(outpath, 'variables.pth'))


        if not step % 1000:
            write_summary_histogram(radiance_field, occupancy_grid, writer, test_dataset, step, render_step_size, args)


        if step == max_steps:
            print("training stops")
            exit()

        step += 1
  

if __name__=="__main__":
    run()