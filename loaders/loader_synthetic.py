import collections
import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import scipy
import zipfile
from .utils import Rays
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from misc.dataset_utils import read_h5
from tqdm import tqdm

def _load_renderings(root_fp: str, subject_id: str, split: str, have_images=True, img_shape=(256, 256)):
    """Load images from disk."""
    # if not root_fp.startswith("/"):
    #     # allow relative path. e.g., "./data/nerf_synthetic/"
    #     root_fp = os.path.join(
    #         os.path.dirname(os.path.abspath(__file__)),
    #         "..",
    #         "..",
    #         root_fp,
    #     )

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


def _load_renderings_transient(root_fp: str, subject_id: str, split: str, num_views= None, have_images=True, img_shape=(256, 256), gamma=False):
    """Load images from disk."""
    # if not root_fp.startswith("/"):
    #     # allow relative path. e.g., "./data/nerf_synthetic/"
    #     root_fp = os.path.join(
    #         os.path.dirname(os.path.abspath(__file__)),
    #         "..",
    #         "..",
    #         root_fp,
    #     )

    data_dir = root_fp
    if split == "train": tname = f"train_v{num_views}"
    else: tname = split

    with open(
            os.path.join(data_dir, "transforms_{}.json".format(tname)), "r"
    ) as fp:
        meta = json.load(fp)
    images = []
    camtoworlds = []

    if have_images:
        for i in tqdm(range(len(meta["frames"]))):
            frame = meta["frames"][i]
            number = int(frame["file_path"].split("_")[-1])

            try:
                files_dir = os.path.join(data_dir, split)
                fname = os.path.join(files_dir, f"{split}_{number:03d}" + ".h5")
                rgba = read_h5(fname)
            except:     
                try:
                    files_dir = os.path.join(data_dir, "test")
                    fname = os.path.join(files_dir, f"test_{number:03d}" + ".h5")
                    rgba = read_h5(fname)
                except:         
                    try:
                        files_dir = os.path.join(data_dir, "test")
                        fname = os.path.join(files_dir, f"test_{number:03d}" + ".h5")
                        archive = zipfile.ZipFile(f"{fname}.zip")
                        file = archive.open(f"test_{number:03d}" + ".h5")
                        rgba = read_h5(file)
                        file.close()
                    except:
                        pass



            rgba = rgba[..., :3]

            if gamma:
                print("using gamma")
                rgba_sum = rgba.sum(-2)
                rgba_sum_normalized = rgba_sum/rgba_sum.max()
                rgba_sum_norm_gamma = rgba_sum_normalized**(1/2.2)
                rgba = (rgba*rgba_sum_norm_gamma[..., None, :])/(rgba_sum[..., None, :]+1e-10)

            camtoworlds.append(frame["transform_matrix"])
            rgba = torch.clip(torch.Tensor(rgba), 0, None)
            images.append(torch.Tensor(rgba))


        images = torch.stack(images, axis=0)

        if split == "test":
            quotient = images.shape[1]//img_shape[0]
            times_downsample = int(np.log2(quotient))
        
            for i in range(times_downsample):
                images = (images[:, 1::2, ::2] + images[:, ::2, ::2] + images[:, 1::2, 1::2] + images[:, ::2, 1::2])/4


        if not gamma:
            #np.save(os.path.join(data_dir, "max.npy"), torch.max(images).numpy())
            max = torch.max(images)
            images /= torch.max(images)

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

    return images, camtoworlds, focal, max



class SubjectLoaderTransient(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "val", "trainval", "test"]

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
            testing=False, 
            rfilter_sigma=0.3, 
            scene=None, 
            sample_as_per_distribution = True, 
            gamma=False,
            num_views = None
    ):
        super().__init__()
        self.testing = testing 
        # assert split in self.SPLITS, "%s" % split
        assert color_bkgd_aug in ["white", "black", "random"]
        self.sample_as_per_distribution = sample_as_per_distribution

        self.HEIGHT, self.WIDTH = img_shape
        self.split = split
        self.num_rays = num_rays
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far
        self.training = (num_rays is not None) and (
                split in ["train", "trainval"]
        )
        self.rep = 0
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        self.have_images = have_images
        self.rfilter_sigma = rfilter_sigma
        self.n_bins = n_bins
        if split == "trainval":
            _images_train, _camtoworlds_train, _focal_train = _load_renderings_transient(
                root_fp, subject_id, "train", gamma=gamma
            )
            _images_val, _camtoworlds_val, _focal_val = _load_renderings_transient(
                root_fp, subject_id, "val", gamma=gamma
            )
            self.images = np.concatenate([_images_train, _images_val])
            self.camtoworlds = np.concatenate(
                [_camtoworlds_train, _camtoworlds_val]
            )
            self.focal = _focal_train
            self.images = torch.from_numpy(self.images).to(torch.float32)

            # ste for transient
            self.images = torch.reshape(self.images, (-1, self.HEIGHT, self.WIDTH, self.n_bins*3))
            # assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)

        elif have_images:
            self.images, self.camtoworlds, self.focal, self.max = _load_renderings_transient(
                root_fp, subject_id, split, gamma=gamma, img_shape=img_shape, num_views=num_views
            )
            self.images = self.images.to(torch.float32)
            assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)
        else:
            _, self.camtoworlds, self.focal = _load_renderings(
                root_fp, subject_id, split, have_images=have_images, img_shape=img_shape, num_views=num_views
            )

        self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
        self.K = torch.tensor(
            [
                [self.focal, 0, self.WIDTH / 2.0],
                [0, self.focal, self.HEIGHT / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )  # (3, 3)

    def __len__(self):
        return len(self.camtoworlds)

    @torch.no_grad()
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
                0, self.WIDTH, size=(num_rays,), device=self.images.device
            )
            y = torch.randint(
                0, self.HEIGHT, size=(num_rays,), device=self.images.device
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
            try:
                rgba = self.images[image_id, y, x]  # (num_rays, 4)
            except: rgba=None

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
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        bounds_max = [4*scale]*x.shape[0]
        loc = 0
        if self.training:
            s_x, s_y, weights = spatial_filter(x, y, sigma=scale, rep = self.rep, prob_dithering=self.sample_as_per_distribution)
            s_x = (torch.clip(x + torch.from_numpy(s_x), 0, self.WIDTH).to(self.camtoworlds.device)).to(torch.float32)
            s_y = (torch.clip(y + torch.from_numpy(s_y), 0, self.HEIGHT).to(self.camtoworlds.device)).to(torch.float32)
            weights = torch.Tensor(weights).to(self.camtoworlds.device)
            #s_x = x.to(self.camtoworlds.device).to(torch.float32)
            #s_y = y.to(self.camtoworlds.device).to(torch.float32)

        elif self.testing:
            s_x, s_y, weights = spatial_filter(x, y, sigma=scale, rep = self.rep, prob_dithering=self.sample_as_per_distribution, normalize=False)
            s_x = (torch.clip(x + torch.from_numpy(s_x), 0, self.WIDTH).to(self.camtoworlds.device)).to(torch.float32)
            s_y = (torch.clip(y + torch.from_numpy(s_y), 0, self.HEIGHT).to(self.camtoworlds.device)).to(torch.float32)
            weights = torch.Tensor(weights).to(self.camtoworlds.device)
            #s_x = x.to(self.camtoworlds.device).to(torch.float32)
            #s_y = y.to(self.camtoworlds.device).to(torch.float32)
        else: 
            s_x = x
            s_y = y

        camera_dirs = F.pad(
            torch.stack(
                [
                    (s_x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (s_y - self.K[1, 2] + 0.5)
                    / self.K[1, 1]
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  

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
            try: rgba = torch.reshape(rgba, (-1,self.n_bins*3))
            except: rgba = None

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


def spatial_filter(x, y, sigma, rep, prob_dithering=True, normalize=True):
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
        if normalize:
            weights = weights/weights.sum(0)[None, :]
        s_x = s_x.flatten()
        s_y = s_y.flatten()
        weights = weights.flatten()
    
    return s_x, s_y, weights