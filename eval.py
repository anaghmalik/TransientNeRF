from gettext import npgettext
import os
import imageio
import torch
import matplotlib.pyplot as plt
from misc.eval_utils import calc_psnr as psnr_fn
from radiance_fields.ngp import NGPRadianceField
from nerfacc import OccGridEstimator
from utils import render_transient
import math
import numpy as np
from misc.dataset_utils import read_h5
from misc.eval_utils import read_json, load_eval_args, num2words
from misc.transient_volrend import torch_laser_kernel
from skimage.metrics import structural_similarity
import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg')
from utils import str2bool
import torch.nn.functional as F
import tqdm 
import scipy.io as sio
from scipy.ndimage import correlate1d
from loaders.loader_captured import LearnRays


def get_gt_depth(frame, camtoworld, data_root_fp):

    depth_folder = os.path.join(data_root_fp, "test")
    number = int(frame["file_path"].split("_")[-1])
    ax_flip = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

    try:
        fname = os.path.join(depth_folder, f"test_{number:03d}_depth_gt" + ".npy")
        pos3d = np.load(fname)
    except:
        fname = os.path.join(depth_folder, f"test_{number:03d}_depth_gt" + ".h5")
        pos3d = read_h5(fname)

    cam_pos = (ax_flip @ camtoworld)[:3, -1]

    depth = np.sqrt(((pos3d - cam_pos[None, None, :]) ** 2).sum(-1))

    return depth


@torch.no_grad()
def eval():
    args = load_eval_args()
    device = args.device

    ckpt_dir = args.checkpoint_dir

    # settings
    if args.version == "simulated":
        div_vals = {'lego': 1170299.068884274, 'chair': 1553458.125, 'ficus': 1135563.421617064, 'hotdog': 1078811.3736717035, 'bench': 202672.34663868704}
        view_scale = {'lego': {2: 137568.875, 3: 137568.875, 5: 141280.171875}, 'chair': {2: 147216.21875, 3: 147216.21875, 5: 147197.421875}, 'ficus': {2: 145397.578125, 3: 146952.296875, 5: 149278.296875}, 'hotdog': {2: 138315.34375, 3: 138315.34375, 5: 150297.515625}, 'bench': {2: 26818.662109375, 3: 26818.662109375, 5: 27648.828125}}
        view_scale = view_scale[args.scene][args.num_views]
        # view_scale = os.path.join(f"{args.data_root_fp}/max.npy")
        # view_scale = np.load(view_scale)

        dataset_scale = div_vals[args.scene]
        from loaders.loader_synthetic import SubjectLoaderTransient as SubjectLoader
    else:
        div_vals = {'boots': 4470.0, 'baskets': 6624.0, 'carving': 4975.0, 'chef': 9993.0, 'cinema': 8478.0,'food': 16857.0}
        view_scale = {'cinema': {2: 483.755615234375, 3: 483.755615234375, 5: 491.021728515625}, 'carving': {2: 299.24365234375, 3: 299.24365234375, 5: 323.334228515625}, 'boots': {2: 273.02276611328125, 3: 273.02276611328125, 5: 277.6478271484375}, 'food': {2: 553.1838989257812, 3: 553.1838989257812, 5: 561.6094970703125}, 'chef': {2: 493.50701904296875, 3: 493.50701904296875, 5: 548.0447998046875}, 'baskets': {2: 308.7045593261719, 3: 319.92572021484375, 5: 326.42620849609375}}
        view_scale = view_scale[args.scene][args.num_views]

        # view_scale = os.path.join(f"{args.data_root_fp}/max.npy")
        # view_scale = np.load(view_scale)
        dataset_scale = div_vals[args.scene]

        from loaders.loader_captured import SubjectLoaderTransientReal as SubjectLoader
        params = np.load(args.intrinsics, allow_pickle=True)[()]
        shift = params['shift'].numpy()
        rays = params['rays']
        data_root_fp = args.data_root_fp
        
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


    outpath = os.path.join(args.checkpoint_dir, "results")
    os.makedirs(outpath, exist_ok = True)
    positions = read_json(os.path.join(args.test_folder_path, f"transforms_{args.split}.json"))

    ckpt_path_rf = os.path.join(ckpt_dir, 'radiance_field_%04d.pth' % (args.step))
    ckpt_path_oc = os.path.join(ckpt_dir, 'occupancy_grid_%04d.pth' % (args.step))
    aabb = torch.tensor(args.aabb, dtype=torch.float32, device=device)
    img_shape = (512, 512)
    if args.version == "simulated":
        test_dataset_kwargs = {"img_shape": img_shape, "have_images": True, "n_bins": args.n_bins, "color_bkgd_aug": "black", "rfilter_sigma": args.rfilter_sigma}
    else:
        test_dataset_kwargs = {"img_shape": img_shape, "have_images": True, "n_bins": args.n_bins, "color_bkgd_aug": "black", "rfilter_sigma": args.rfilter_sigma, "shift":shift}

    render_step_size = (
            (aabb[3:] - aabb[:3]).max()
            * math.sqrt(3)
            / args.render_n_samples
    ).item()

    # load radiance field and occupancy grid
    occupancy_grid = OccGridEstimator(
        roi_aabb=aabb, resolution=args.grid_resolution, levels=args.grid_nlvl
    ).to(device)

    radiance_field = NGPRadianceField(
        use_viewdirs=True,
        aabb=aabb,
        unbounded=False,
        radiance_activation=torch.exp,
        args = args).to("cpu")
    radiance_field = radiance_field.to(device)

    ckpt = torch.load(ckpt_path_rf, map_location=device)
    radiance_field.load_state_dict(ckpt)
    radiance_field = radiance_field.to(device)

    ckpt = torch.load(ckpt_path_oc, map_location=device)
    occupancy_grid.load_state_dict(ckpt)
    occupancy_grid = occupancy_grid.to(device)

    # load test loader
    test_dataset = SubjectLoader(
        subject_id=f"{args.scene}",
        root_fp=args.test_folder_path,
        split=args.split,
        num_rays=None,
        **test_dataset_kwargs,
        testing=True,
        sample_as_per_distribution=args.sample_as_per_distribution
    )
    if args.version == "captured":
        test_dataset.K = LearnRays(rays, device=device, img_shape=img_shape)
        test_dataset.K = test_dataset.K.to(device)

    test_dataset.rep = 1
    # test_dataset.images = test_dataset.images.to(device)
    test_dataset.camtoworlds = test_dataset.camtoworlds.to(device)
    test_dataset.K = test_dataset.K.to(device)
    psnrs = []
    divs = []
    l1_errors_gt_rnd = []
    l1_errors_gt_rnd_pct = []
    ssims = []
    lpipss = []
    mses = []
    MSE_errors_gt_rnd = []
    MSE_errors_gt_rnd_pct = []
    
    # render transients and other outputs from network
    for i in range(len(test_dataset)):

        ind = int(positions["frames"][i]["file_path"].split("_")[-1])
        print(f"test image {ind}")

        rgb = np.zeros((img_shape[0], img_shape[1], args.n_bins, 3))
        depth = np.zeros((img_shape[0], img_shape[1]))
        depth_viz = np.zeros((img_shape[0], img_shape[1]))
        weights_sum = 0
        for j in tqdm.tqdm(range(args.rep_number)):

            data = test_dataset[i]
            pixels = data["pixels"].detach().cpu().numpy()
            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            sample_weights = data["weights"]
            del data

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


            depth += (out["depths"]*sample_weights[:, None]).reshape(img_shape[0], img_shape[1]).detach().cpu().numpy()
            depth_viz += (out["depths"]*sample_weights[:, None]*(out["opacities"]>0)).reshape(img_shape[0], img_shape[1]).detach().cpu().numpy()
            rgb += (out["colors"] * sample_weights[:, None]).reshape(img_shape[0], img_shape[1], args.n_bins, 3).detach().cpu().numpy()
            weights_sum += sample_weights.detach().cpu().numpy()


            del out

        rgb = rgb / weights_sum.reshape(img_shape[0], img_shape[1], 1, 1)
        depth = depth / weights_sum.reshape(img_shape[0], img_shape[1])
        pixels = pixels.reshape(img_shape[0], img_shape[1], args.n_bins, 3)
        depth_viz = depth_viz / weights_sum.reshape(img_shape[0], img_shape[1])

        # (a) rendering depth against gt
        if args.version == "simulated":
            exr_depth = get_gt_depth(positions["frames"][i], test_dataset.camtoworlds[i].cpu().numpy(), args.test_folder_path)
        elif args.version == "captured":
            lm = correlate1d(pixels[..., 0], laser.cpu().numpy(), axis=-1)
            exr_depth = np.argmax(lm, axis=-1)
            exr_depth = (exr_depth*2*299792458*4e-12)/2 #speed of light plus two times travel distance

        
        mask = (pixels.sum((-1, -2)) > 0)
        error_gt_rnd = np.abs(exr_depth - depth)
        percentage_error = (error_gt_rnd / (exr_depth + 1e-10))[mask]
        l1_errors_gt_rnd.append((error_gt_rnd * mask).mean())
        l1_errors_gt_rnd_pct.append(100 * percentage_error.mean())

        print(f"Error between ground truth depth and our rendered depth {error_gt_rnd[mask].mean()} ({100 * percentage_error.mean()}%)")

        error_gt_rnd = np.abs(exr_depth - depth)**2
        percentage_error = (error_gt_rnd / (exr_depth + 1e-10))[mask]
        MSE_errors_gt_rnd.append((error_gt_rnd * mask).mean())
        MSE_errors_gt_rnd_pct.append(100 * percentage_error.mean())

        # (c) psnr on rendered images
        rgb_image = rgb.sum(axis=-2)*view_scale/dataset_scale
        rgb_image = np.clip(rgb_image, 0, 1) ** (1 / 2.2)
        data_image = (pixels.sum(-2)*test_dataset.max.numpy()/dataset_scale) ** (1 / 2.2)
        if args.version == "simulated": vmin,vmax=2.5,5.5
        else: vmin,vmax=0.8,1.5
        
        plt.imsave(f"{outpath}/{args.scene}_{args.num_views}_{args.step}_test{ind}_depth.png", depth, cmap='inferno', vmin=vmin, vmax=vmax)
        plt.imsave(f"{outpath}/{args.scene}_{args.num_views}_{args.step}_test{ind}_depth_viz.png", depth_viz, cmap='inferno', vmin=vmin, vmax=vmax)

        np.save(f"{outpath}/{args.scene}_{args.num_views}_{args.step}_test{ind}_depth.png", depth)
        imageio.imwrite(f"{outpath}/{args.scene}_{args.num_views}_{args.step}_test{ind}_RGB.png", (rgb_image*255.0).astype(np.uint8))
        imageio.imwrite(f"{outpath}/{args.scene}_{args.num_views}_{args.step}_test{ind}_RGB_gt.png", (data_image*255.0).astype(np.uint8))

        mse_ = F.mse_loss(torch.from_numpy(data_image), torch.from_numpy(rgb_image))
        mses.append(mse_)
        print(f"Image mse {mse_}")

        psnr_ = psnr_fn(data_image, rgb_image)
        psnrs.append(psnr_)
        print(f"Image psnr {psnr_}")

        ssim_, _ = structural_similarity(data_image, rgb_image, full=True, channel_axis=2)
        ssims.append(ssim_)
        print(f"Image ssim {ssim_}")

        lpips_ = loss_fn_vgg(torch.from_numpy(data_image * 2 - 1).unsqueeze(-1).permute((3, 2, 0, 1)).to(torch.float32), torch.from_numpy(rgb_image * 2 - 1).unsqueeze(-1).permute((3, 2, 0, 1)).to(torch.float32))
        lpips_ = lpips_.detach().cpu().numpy().flatten()[0]
        lpipss.append(lpips_)
        print(f"Image LPIPS {lpips_}")

        torch.cuda.empty_cache()
        print("-----")

    print(f"Average PSNR: {sum(psnrs) / len(psnrs)}")
    print(f"Average SSIM: {sum(ssims) / len(ssims)}")
    print(f"Average LPIPS: {sum(lpipss) / len(lpipss)}")

    print(f"Average errors_gt_rnd: {sum(l1_errors_gt_rnd) / len(l1_errors_gt_rnd)}")

    np.savetxt(f"{outpath}/{args.scene}_{args.num_views}_end{ind}.txt",
               np.stack([np.array(mses), np.array(psnrs), np.array(ssims), np.array(lpipss),
                         np.array(l1_errors_gt_rnd), np.array(l1_errors_gt_rnd_pct)], axis=0))


if __name__ == "__main__":
    eval()