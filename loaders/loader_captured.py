import collections
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy
from mat73 import loadmat
from .utils import Rays
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from misc.dataset_utils import read_h5


def _load_renderings(root_fp: str, subject_id: str, split: str, have_images=True, img_shape=(256, 256)):
    """Load images from disk."""
    if not root_fp.startswith("/"):
        # allow relative path. e.g., "./data/nerf_synthetic/"
        root_fp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            root_fp,
        )

    data_dir = root_fp
    with open(
            os.path.join(data_dir, "transforms_{}.json".format(split)), "r"
    ) as fp:
        meta = json.load(fp)
    images = []
    camtoworlds = []

    if have_images:
        for i in range(len(meta["frames"])):
            frame = meta["frames"][i]
            number = int(frame["file_path"].split("_")[-1])
            fname = os.path.join(data_dir, f"{number:03d}" + ".png")

            # fname = os.path.join(data_dir, frame["file_path"] + ".png")
            rgba = imageio.imread(fname)
            camtoworlds.append(frame["transform_matrix"])
            images.append(rgba)

        images = np.stack(images, axis=0)
        camtoworlds = np.stack(camtoworlds, axis=0)

        h, w = images.shape[1:3]
    else:
        for i in range(len(meta["frames"])):
            frame = meta["frames"][i]
            camtoworlds.append(frame["transform_matrix"])

        camtoworlds = np.stack(camtoworlds, axis=0)

        h, w = img_shape

    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

    return images, camtoworlds, focal



def _load_renderings_transient_real(root_fp: str, subject_id: str, split: str, have_images=True, img_shape=(256, 256), n_bins=4096,   shift = 0):
    """Load images from disk."""

    data_dir = root_fp
    with open(
            os.path.join(data_dir, "transforms_{}.json".format(split)), "r"
    ) as fp:
        meta = json.load(fp)

    images = []
    camtoworlds = []

    exposure_time= 299792458*4e-12

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


    if have_images:
        tqdm.write('Loading data')
        for i in tqdm(range(len(meta["frames"]))):
            frame = meta["frames"][i]
            number = int(frame["file_path"].split("_")[-1])

            fname = os.path.join(os.path.join(data_dir, "../.."), f"transient{number:03d}.pt")
            rgba = torch.load(fname).to_dense()
            rgba = torch.Tensor(rgba)[..., :3000].float().cpu()
            # if img_shape[0]==256:
            #     rgba = (rgba[::2, ::2] + rgba[::2, 1::2] +  rgba[1::2, ::2]+ rgba[1::2, 1::2] )/4
            
            rgba = torch.nn.functional.grid_sample(rgba[None, None, ...], grid, align_corners=True).squeeze().cpu()
            rgba = (rgba[..., 1::2]+ rgba[..., ::2] )/2

            camtoworlds.append(frame["transform_matrix"])
            rgba = torch.clip(rgba, 0, None)
            rgba = rgba[..., None].repeat(1, 1, 1, 3)
            images.append(rgba)



        images = torch.stack(images, axis=0)
        max = torch.max(images)
        images /= torch.max(images)

        if split == "test":
            quotient = images.shape[1]//img_shape[0]
            times_downsample = int(np.log2(quotient))
        
            for i in range(times_downsample):
                images = (images[:, 1::2, ::2] + images[:, ::2, ::2] + images[:, 1::2, 1::2] + images[:, ::2, 1::2])/4

        camtoworlds = np.stack(camtoworlds, axis=0)

        h, w = images.shape[1:3]
    else:
        for i in range(len(meta["frames"])):
            frame = meta["frames"][i]
            camtoworlds.append(frame["transform_matrix"])

        camtoworlds = np.stack(camtoworlds, axis=0)
        max = 1
        h, w = img_shape
    

    return images, camtoworlds, max


class SubjectLoaderTransientReal(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "val", "trainval", "test"]
    SUBJECT_IDS = [
        "chair",
        "drums",
        "ficus",
        "hotdog",
        "lego",
        "materials",
        "mic",
        "ship",
    ]

    # WIDTH, HEIGHT = 64, 64
    NEAR, FAR = 0, 6
    OPENGL_CAMERA = True

    def __init__(
            self,
            subject_id: str,
            root_fp: str,
            split: str,
            color_bkgd_aug: str = "black",
            num_rays: int = None,
            near: float = None,
            far: float = None,
            batch_over_images: bool = True,
            have_images=True,
            img_shape=(256, 256),
            n_bins=10000, 
            rfilter_sigma=0.15,
            sample_as_per_distribution = True,
            shift = 0.3, 
            testing =False
    ):
        super().__init__()
        #assert split in self.SPLITS, "%s" % split
        # assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        assert color_bkgd_aug in ["white", "black", "random"]
        self.sample_as_per_distribution = sample_as_per_distribution
        self.rfilter_sigma = rfilter_sigma
        self.HEIGHT, self.WIDTH = img_shape
        self.split = split
        self.testing = testing
        self.num_rays = num_rays
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far
        self.training = (num_rays is not None) and (
                split in ["train", "trainval"]
        )
        self.shift = shift
        self.testing = testing
        self.rep = 1
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        self.have_images = have_images
        self.n_bins = n_bins
        shift = shift

        if split == "trainval":
            _images_train, _camtoworlds_train, _focal_train = _load_renderings_transient_real(
                root_fp, subject_id, "train", n_bins=self.n_bins, shift=shift
            )
            _images_val, _camtoworlds_val, _focal_val = _load_renderings_transient_real(
                root_fp, subject_id, "val", n_bins=self.n_bins, shift=shift
            )
            self.images = np.concatenate([_images_train, _images_val])
            self.camtoworlds = np.concatenate(
                [_camtoworlds_train, _camtoworlds_val]
            )
            self.focal = _focal_train
            self.images = torch.from_numpy(self.images).to(torch.float32)

            # ste for transient
            self.images = torch.reshape(self.images, (-1, self.HEIGHT, self.WIDTH, self.n_bins*3))


        elif have_images:
            self.images, self.camtoworlds, self.focal = _load_renderings_transient_real(
                root_fp, subject_id, split, n_bins=self.n_bins, shift=shift, img_shape=img_shape
            )
            self.images =self.images.to(torch.float32)
            assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)
        else:
            _, self.camtoworlds, self.focal = _load_renderings(
                root_fp, subject_id, split, have_images=have_images, img_shape=img_shape
            )
        
        self.max = self.focal

        self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
        self.camtoworlds[:, :3, 3] = self.camtoworlds[:, :3, 3]
        # self.K = LearnRays(params["rays"], img_shape=(self.WIDTH, self.HEIGHT))



    def __len__(self):
        return len(self.camtoworlds)

    # @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]
        # pixels, alpha = torch.split(rgba, [3, 1], dim=-1)

        if rgba is not None:
            pixels = rgba.to(self.camtoworlds.device)
        else:
            pixels = rgba
        

        if self.color_bkgd_aug == "random":
            color_bkgd = torch.rand(3, device=self.camtoworlds.device)
        elif self.color_bkgd_aug == "white":
            color_bkgd = torch.ones(3, device=self.camtoworlds.device)
        elif self.color_bkgd_aug == "black":
            color_bkgd = torch.zeros(3, device=self.camtoworlds.device)

        # pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }


    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index, rep=None, num_rays=None):
        """Fetch the data (it maybe cached for multiple batches)."""
        if num_rays==None:
            num_rays = self.num_rays
        if rep==None:
            rep = self.rep
        


        if self.training:
            if self.batch_over_images:
                image_id = torch.randint(
                    0,
                    len(self.images),
                    size=(num_rays,),
                    device=self.images.device,
                )
            else:
                image_id = [index]
            x = torch.randint(
                0, self.WIDTH, size=(num_rays,), device="cpu"
            )
            y = torch.randint(
                0, self.HEIGHT, size=(num_rays,), device="cpu"
            )
            x = x.repeat(rep)
            y = y.repeat(rep)
            image_id = image_id.repeat(rep)


            rgba = self.images[image_id, y, x]  # (num_rays, 4)

        elif self.testing:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH, device="cpu"),
                torch.arange(self.HEIGHT, device="cpu"),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()
            x = x.repeat(rep)
            y = y.repeat(rep)
            # image_id = image_id.repeat(rep)
            if self.have_images:
                rgba = self.images[image_id, y, x]  # (num_rays, 4)
            else:
                rgba = None 
        elif self.have_images:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH, device=self.camtoworlds.device),
                torch.arange(self.HEIGHT, device=self.camtoworlds.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()
            rgba = self.images[image_id, y, x]  # (num_rays, 4)
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH, device=self.camtoworlds.device),
                torch.arange(self.HEIGHT, device=self.camtoworlds.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        
        scale = self.rfilter_sigma
        c2w = self.camtoworlds[image_id]

        bounds_max = [4*scale]*x.shape[0]
        loc = 0
        if self.training:
            s_x, s_y, weights = spatial_filter(x, y, sigma=scale, rep = self.rep, prob_dithering=self.sample_as_per_distribution)
            s_x = (torch.clip(x + torch.from_numpy(s_x), 0, self.WIDTH-1).to(self.camtoworlds.device)).to(torch.float32)
            s_y = (torch.clip(y + torch.from_numpy(s_y), 0, self.HEIGHT-1).to(self.camtoworlds.device)).to(torch.float32)
            weights = torch.Tensor(weights).to(self.camtoworlds.device)

        elif self.testing:
            s_x, s_y, weights = spatial_filter(x, y, sigma=scale, rep = self.rep, prob_dithering=self.sample_as_per_distribution)
            s_x = (torch.clip(x + torch.from_numpy(s_x), 0, self.WIDTH-1).to(self.camtoworlds.device)).to(torch.float32)
            s_y = (torch.clip(y + torch.from_numpy(s_y), 0, self.HEIGHT-1).to(self.camtoworlds.device)).to(torch.float32)
            weights = torch.Tensor(weights).to(self.camtoworlds.device)
        else: 
            s_x = x
            s_y = y



        camera_dirs = self.K(s_x, s_y)

        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        if self.training:
            origins = torch.reshape(origins, (-1, 3))
            viewdirs = torch.reshape(viewdirs, (-1, 3))
            # here
            rgba = torch.reshape(rgba, (-1,self.n_bins*3))
        elif self.testing:
            origins = torch.reshape(origins, (-1, 3))
            viewdirs = torch.reshape(viewdirs, (-1, 3))
            # here
            if self.have_images:
                rgba = torch.reshape(rgba, (-1,self.n_bins*3))

        elif self.have_images:
            origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
            viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))
            rgba = torch.reshape(rgba, (self.HEIGHT, self.WIDTH, self.n_bins * 3))
        else:
            origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
            viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))
            rgba = None

        rays = Rays(origins=origins, viewdirs=viewdirs)
        if self.training or self.testing:
            return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            "weights":weights
        }

        return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
        }





class LearnRays(torch.nn.Module):
    def __init__(self, rays, device ="cuda:0", img_shape = (256, 256)):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnRays, self).__init__()
        self.device = device
        self.init_c2w = None
        self.img_shape = img_shape

        x = np.arange(32, 480)
        X, Y = np.meshgrid(x, x)

        tar_x = np.arange(0, 512)
        tar_X, tar_Y = np.meshgrid(tar_x, tar_x)
        # rays = rays.detach().cpu().numpy()

        ray_x = scipy.interpolate.interpn((x, x), rays[32:-32, 32:-32, 0].transpose(1, 0), np.stack([tar_X, tar_Y], axis=-1).squeeze().flatten(), bounds_error = False, fill_value=None).reshape(512, 512)
        ray_y = scipy.interpolate.interpn((x, x), rays[32:-32, 32:-32, 1].transpose(1, 0), np.stack([tar_X, tar_Y], axis=-1).squeeze().flatten(), bounds_error = False, fill_value=None).reshape(512, 512)
        ray_z = scipy.interpolate.interpn((x, x), rays[32:-32, 32:-32, 2].transpose(1, 0), np.stack([tar_X, tar_Y], axis=-1).squeeze().flatten(), bounds_error = False, fill_value=None).reshape(512, 512)

        rays = torch.from_numpy(np.stack([ray_x, ray_y, ray_z], axis=-1)).to(self.device)

        quotient = rays.shape[1]//img_shape[0]
        times_downsample = int(np.log2(quotient))
    
        for i in range(times_downsample):
            rays = (rays[1::2, ::2] + rays[::2, ::2] + rays[1::2, 1::2] + rays[::2, 1::2])/4

        rays = rays/torch.linalg.norm(rays, dim=-1, keepdims=True)
        self.rays = rays
        # self.rays = torch.nn.Parameter(rays, requires_grad=learn_rays)    

    def forward(self, x0, y0):
        """input coord = (n, 2)
        rays = (512, 512, 3)
        """
        rays = self.rays
        x1, y1 = torch.floor(x0.float()), torch.floor(y0.float())
        x2, y2 = x1+1, y1+1
        """
        Perform bilinear interpolation to estimate the value of the function f(x, y)
        at the continuous point (x0, y0), given that f is known at integer values of x, y.
        """
        # if (y1>self.img_shape[0]-1).any() or (x1>self.img_shape[0]-1).any():
        #     print("hello")
        x1, y1 = torch.clip(x1, 0, self.img_shape[0]-1), torch.clip(y1, 0, self.img_shape[0]-1)

        # x2, y2 = torch.clip(x2, 0, self.img_shape[0]-1), torch.clip(y2, 0, self.img_shape[0]-1)

        # Compute the weights for the interpolation
        wx1 = ((x2 - x0) / (x2 - x1 + 1e-8))[:, None]
        wx2 = ((x0 - x1) / (x2 - x1 + 1e-8))[:, None]
        wy1 = ((y2 - y0) / (y2 - y1 + 1e-8))[:, None]
        wy2 = ((y0 - y1) / (y2 - y1 + 1e-8))[:, None]

        x1, y1, x2, y2 = x1.long(), y1.long(), x2.long(), y2.long()
        x2, y2 = torch.clip(x2, 0, self.img_shape[0] - 1), torch.clip(y2, 0, self.img_shape[0] - 1)

        # Compute the interpolated value of f(x, y) at (x0, y0)
        f_interp = wx1 * wy1 * rays[y1, x1] + \
                wx1 * wy2 * rays[y2, x1] + \
                wx2 * wy1 * rays[y1, x2] + \
                wx2 * wy2 * rays[y2, x2]

        f_interp = f_interp/torch.linalg.norm(f_interp, dim=-1, keepdims=True) 
        return f_interp.float()


def spatial_filter(x, y, sigma, rep, prob_dithering=True):
    pdf_fn = lambda x: np.exp(-x/(2*sigma**2)) - np.exp(-16)
    if prob_dithering:
        bounds_max = [4*sigma]*x.shape[0]
        loc = 0
        s_x = scipy.stats.truncnorm.rvs((-np.array(bounds_max)-loc)/sigma, (np.array(bounds_max)-loc)/sigma, loc=loc, scale=sigma)
        s_y = scipy.stats.truncnorm.rvs((-np.array(bounds_max)-loc)/sigma, (np.array(bounds_max)-loc)/sigma, loc=loc, scale=sigma)
        weights = np.ones_like(s_x)*1/rep
    
    else:
        s_x = np.random.uniform(low=-4*sigma, high=4*sigma, size=(rep, x.shape[0]//rep))
        s_y = np.random.uniform(low=-4*sigma, high=4*sigma, size=(rep, x.shape[0]//rep))
        dists = (s_x**2 + s_y**2)
        weights = pdf_fn(dists)
        weights = weights/weights.sum(0)[None, :]
        s_x = s_x.flatten()
        s_y = s_y.flatten()
        weights = weights.flatten()
    
    return s_x, s_y, weights
