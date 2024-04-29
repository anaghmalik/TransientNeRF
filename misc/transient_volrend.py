from typing import Callable, Dict, Optional, Tuple
import torch
from torch import Tensor
from nerfacc.pack import pack_info
from nerfacc.scan import exclusive_prod, exclusive_sum
from torch_scatter import scatter_max
import math 


def rendering_transient_single_path(
    # ray marching results
    t_starts: Tensor,
    t_ends: Tensor,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
    # radiance field
    rgb_sigma_fn: Optional[Callable] = None,
    # rendering options
    render_bkgd: Optional[Tensor] = None,
    args = None
):

    # Query sigma and color with gradients
    data = rgb_sigma_fn(t_starts, t_ends, ray_indices.long())
    rgbs, sigmas = data
    dists = (t_starts + t_ends)/2
    
    if args.exp:
        rgbs = torch.exp(rgbs)-1


    # Rendering: compute weights and ray indices.
    weights_non_squared, transmittance, alphas = render_weight_from_density(
            t_starts,
            t_ends,
            sigmas,
            ray_indices=ray_indices,
            n_rays=n_rays
    )
    
    # modelling squared transmittance 
    # alphas = 1 - torch.exp(-sigmas * (t_ends - t_starts))
    weights = (weights_non_squared ** 2 / (alphas.squeeze() + 1e-9))

    # r**2 fall off 
    src = weights[:, None] * rgbs
    src = src/(dists[:, None].detach()**2 + 1e-10)

    if args.version == "simulated":
        # this code bins the output samples into a tensor of size [n_rays, n_bins, 3]
        tfilter_sigma = args.tfilter_sigma
        bin_mapping, dist_weights = mapping_dist_to_bin_mitsuba(dists, args.n_bins, args.exposure_time, c=1, sigma=tfilter_sigma)
        src = (dist_weights[..., None] * src[:, None, :]).flatten(0, 1)
        colors = torch.zeros((n_rays * args.n_bins, 3), device=weights.device)
        index = ((torch.repeat_interleave(ray_indices, 8*tfilter_sigma) * args.n_bins) + bin_mapping.flatten().long())[:, None].expand(-1, 3).long()
        colors.scatter_add_(0, index, src)
        colors = colors.view(n_rays, args.n_bins, 3)
        bin_numbers_floor, bin_numbers_ceil, alpha = mapping_dist_to_bin(dists, args.n_bins, args.exposure_time)
        index_f = ((ray_indices * args.n_bins) + bin_numbers_floor.long())[:, None].expand(-1, 3).long()
        index = index_f
    
    
    if args.version == "captured":
        bin_numbers_floor, bin_numbers_ceil, _ = mapping_dist_to_bin(dists, args.n_bins, args.exposure_time)    
        colors = torch.zeros((n_rays * args.n_bins, 3), device=weights.device)    
        index = ((ray_indices * args.n_bins) + bin_numbers_floor.long())[:, None].expand(-1, 3).long()   
        colors.scatter_add_(0, index, src)    
        colors = colors.view(n_rays, args.n_bins, 3)
        colors = convolve_colour(colors, args.laser_kernel, n_bins=args.n_bins)



    # do the same for the sigmas
    comp_weights = torch.zeros((n_rays * args.n_bins, 1), device=weights.device)
    comp_weights.scatter_add_(0, index[:, [0]], weights_non_squared[:, None])
    comp_weights = comp_weights.reshape(n_rays, args.n_bins)


    opacities = accumulate_along_rays(
        weights, values=None, ray_indices=ray_indices, n_rays=n_rays
    )

    
    out, argmax = scatter_max(weights_non_squared, ray_indices, out=torch.zeros(n_rays, device=ray_indices.device))

    if t_starts.shape[0]!=0:
        argmax[argmax==weights.shape[0]] = weights.shape[0]-1
        depths = (t_starts+t_ends)[argmax]/2
    else:
        depths = out[:, None]

    to_accum_var = ((t_ends+t_starts)/2 - depths[ray_indices])**2
    depths_variance = accumulate_along_rays(
        weights_non_squared,
        ray_indices=ray_indices,
        values=to_accum_var[:, None],
        n_rays=n_rays,
    )
    depths_variance = depths_variance/(opacities+1e-10)


    return colors, opacities, depths, depths_variance, comp_weights, rgbs


def mapping_dist_to_bin_mitsuba(dists, n_bins, exposure_time, c=1, sigma=5):
    times = 2 * dists / c
    ratio = times / exposure_time
    ranges = torch.arange(0, 8 * sigma, device=dists.device)[None, :].repeat(ratio.shape[0], 1)
    bin_mapping = (torch.ceil(ratio-4*sigma))[:, None]+ranges
    ranges = bin_mapping - ratio[:, None]
    dist_weights = torch.exp(-ranges**2/(2*sigma**2))-math.exp(-8)

    dist_weights[(bin_mapping<0) ] = 0
    dist_weights[(bin_mapping>n_bins) ] = 0

    bin_mapping = torch.clip(bin_mapping, 0, n_bins-1)
    dist_weights = (dist_weights.T/(dist_weights.sum(-1)[: None]+1e-10)).T
    return bin_mapping, dist_weights


def mapping_dist_to_bin(dists, n_bins, exposure_time, c=1):
    times = 2 * dists / c
    #  (torch.randn(times.shape[0])*7).to("cuda")
    ratio = times / exposure_time
    alpha = (torch.ceil(ratio) - ratio) / (torch.ceil(ratio) - torch.floor(ratio) + 1e-10)

    bin_numbers_floor = torch.floor(ratio)
    bin_numbers_ceil = torch.ceil(ratio)
    # if torch.max(bin_numbers)>bin_length:
    #     print("hello")
    bin_numbers_floor = torch.clip(bin_numbers_floor, 0, n_bins - 1)
    bin_numbers_ceil = torch.clip(bin_numbers_ceil, 0, n_bins - 1)

    return bin_numbers_floor, bin_numbers_ceil, alpha


def render_transmittance_from_alpha(
    alphas: Tensor,
    packed_info: Optional[Tensor] = None,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
    prefix_trans: Optional[Tensor] = None,
) -> Tensor:
    """Compute transmittance :math:`T_i` from alpha :math:`\\alpha_i`.

    .. math::
        T_i = \\prod_{j=1}^{i-1}(1-\\alpha_j)

    This function supports both batched and flattened input tensor. For flattened input tensor, either
    (`packed_info`) or (`ray_indices` and `n_rays`) should be provided.

    Args:
        alphas: The opacity values of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
        packed_info: A tensor of shape (n_rays, 2) that specifies the start and count
            of each chunk in the flattened samples, with in total n_rays chunks.
            Useful for flattened input.
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        prefix_trans: The pre-computed transmittance of the samples. Tensor with shape (all_samples,).

    Returns:
        The rendering transmittance with the same shape as `alphas`.

    Examples:

    .. code-block:: python

        >>> alphas = torch.tensor([0.4, 0.8, 0.1, 0.8, 0.1, 0.0, 0.9], device="cuda")
        >>> ray_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2], device="cuda")
        >>> transmittance = render_transmittance_from_alpha(alphas, ray_indices=ray_indices)
        tensor([1.0, 0.6, 0.12, 1.0, 0.2, 1.0, 1.0])
    """
    if ray_indices is not None and packed_info is None:
        packed_info = pack_info(ray_indices, n_rays)

    trans = exclusive_prod(1 - alphas, packed_info)
    if prefix_trans is not None:
        trans *= prefix_trans
    return trans

def convolve_colour(color, kernel, n_bins):
    color = color.transpose(1, 2).reshape(-1, n_bins)
    color = kernel(color[:, None, :]).squeeze()
    color = color.reshape(-1, 3, n_bins).transpose(1, 2)
    return color

def torch_laser_kernel(laser, device='cuda'):
    m = torch.nn.Conv1d(1, 1, laser.shape[0], padding=(laser.shape[0] - 1) // 2, padding_mode="zeros", device=device)
    m.weight.requires_grad = False
    m.bias.requires_grad = False
    m.bias *= 0
    m.weight = torch.nn.Parameter(laser[None, None, ...])
    return m

def render_transmittance_from_density(
    t_starts: Tensor,
    t_ends: Tensor,
    sigmas: Tensor,
    packed_info: Optional[Tensor] = None,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
    prefix_trans: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Compute transmittance :math:`T_i` from density :math:`\\sigma_i`.

    .. math::
        T_i = exp(-\\sum_{j=1}^{i-1}\\sigma_j\delta_j)
    
    This function supports both batched and flattened input tensor. For flattened input tensor, either
    (`packed_info`) or (`ray_indices` and `n_rays`) should be provided.

    Args:
        t_starts: Where the frustum-shape sample starts along a ray. Tensor with \
            shape (all_samples,) or (n_rays, n_samples).
        t_ends: Where the frustum-shape sample ends along a ray. Tensor with \
            shape (all_samples,) or (n_rays, n_samples).
        sigmas: The density values of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
        packed_info: A tensor of shape (n_rays, 2) that specifies the start and count
            of each chunk in the flattened samples, with in total n_rays chunks.
            Useful for flattened input.
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        prefix_trans: The pre-computed transmittance of the samples. Tensor with shape (all_samples,).

    Returns:
        The rendering transmittance and opacities, both with the same shape as `sigmas`.

    Examples:
    
    .. code-block:: python

        >>> t_starts = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device="cuda")
        >>> t_ends = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], device="cuda")
        >>> sigmas = torch.tensor([0.4, 0.8, 0.1, 0.8, 0.1, 0.0, 0.9], device="cuda")
        >>> ray_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2], device="cuda")
        >>> transmittance, alphas = render_transmittance_from_density(
        >>>     t_starts, t_ends, sigmas, ray_indices=ray_indices)
        transmittance: [1.00, 0.67, 0.30, 1.00, 0.45, 1.00, 1.00]
        alphas: [0.33, 0.55, 0.095, 0.55, 0.095, 0.00, 0.59]

    """
    if ray_indices is not None and packed_info is None:
        packed_info = pack_info(ray_indices, n_rays)

    sigmas_dt = sigmas * (t_ends - t_starts)
    alphas = 1.0 - torch.exp(-sigmas_dt)
    trans = torch.exp(-exclusive_sum(sigmas_dt, packed_info))
    if prefix_trans is not None:
        trans *= prefix_trans
    return trans, alphas


def render_weight_from_alpha(
    alphas: Tensor,
    packed_info: Optional[Tensor] = None,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
    prefix_trans: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Compute rendering weights :math:`w_i` from opacity :math:`\\alpha_i`.

    .. math::
        w_i = T_i\\alpha_i, \\quad\\textrm{where}\\quad T_i = \\prod_{j=1}^{i-1}(1-\\alpha_j)

    This function supports both batched and flattened input tensor. For flattened input tensor, either
    (`packed_info`) or (`ray_indices` and `n_rays`) should be provided.

    Args:
        alphas: The opacity values of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
        packed_info: A tensor of shape (n_rays, 2) that specifies the start and count
            of each chunk in the flattened samples, with in total n_rays chunks.
            Useful for flattened input.
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        prefix_trans: The pre-computed transmittance of the samples. Tensor with shape (all_samples,).

    Returns:
        The rendering weights and transmittance, both with the same shape as `alphas`.

    Examples:

    .. code-block:: python

        >>> alphas = torch.tensor([0.4, 0.8, 0.1, 0.8, 0.1, 0.0, 0.9], device="cuda")
        >>> ray_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2], device="cuda")
        >>> weights, transmittance = render_weight_from_alpha(alphas, ray_indices=ray_indices)
        weights: [0.4, 0.48, 0.012, 0.8, 0.02, 0.0, 0.9])
        transmittance: [1.00, 0.60, 0.12, 1.00, 0.20, 1.00, 1.00]

    """
    trans = render_transmittance_from_alpha(
        alphas, packed_info, ray_indices, n_rays, prefix_trans
    )
    weights = trans * alphas
    return weights, trans


def render_weight_from_density(
    t_starts: Tensor,
    t_ends: Tensor,
    sigmas: Tensor,
    packed_info: Optional[Tensor] = None,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
    prefix_trans: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute rendering weights :math:`w_i` from density :math:`\\sigma_i` and interval :math:`\\delta_i`.

    .. math::
        w_i = T_i(1 - exp(-\\sigma_i\delta_i)), \\quad\\textrm{where}\\quad T_i = exp(-\\sum_{j=1}^{i-1}\\sigma_j\delta_j)

    This function supports both batched and flattened input tensor. For flattened input tensor, either
    (`packed_info`) or (`ray_indices` and `n_rays`) should be provided.

    Args:
        t_starts: The start time of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
        t_ends: The end time of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
        sigmas: The density values of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
        packed_info: A tensor of shape (n_rays, 2) that specifies the start and count
            of each chunk in the flattened samples, with in total n_rays chunks.
            Useful for flattened input.
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        prefix_trans: The pre-computed transmittance of the samples. Tensor with shape (all_samples,).

    Returns:
        The rendering weights, transmittance and opacities, both with the same shape as `sigmas`.

    Examples:

    .. code-block:: python

        >>> t_starts = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device="cuda")
        >>> t_ends = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], device="cuda")
        >>> sigmas = torch.tensor([0.4, 0.8, 0.1, 0.8, 0.1, 0.0, 0.9], device="cuda")
        >>> ray_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2], device="cuda")
        >>> weights, transmittance, alphas = render_weight_from_density(
        >>>     t_starts, t_ends, sigmas, ray_indices=ray_indices)
        weights: [0.33, 0.37, 0.03, 0.55, 0.04, 0.00, 0.59]
        transmittance: [1.00, 0.67, 0.30, 1.00, 0.45, 1.00, 1.00]
        alphas: [0.33, 0.55, 0.095, 0.55, 0.095, 0.00, 0.59]

    """
    trans, alphas = render_transmittance_from_density(
        t_starts, t_ends, sigmas, packed_info, ray_indices, n_rays, prefix_trans
    )
    weights = trans * alphas
    return weights, trans, alphas


@torch.no_grad()
def render_visibility_from_alpha(
    alphas: Tensor,
    packed_info: Optional[Tensor] = None,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
    prefix_trans: Optional[Tensor] = None,
) -> Tensor:
    """Compute visibility from opacity :math:`\\alpha_i`.

    In this function, we first compute the transmittance from the sample opacity. The
    transmittance is then used to filter out occluded samples. And opacity is used to
    filter out transparent samples. The function returns a boolean tensor indicating
    which samples are visible (`transmittance > early_stop_eps` and `opacity > alpha_thre`).

    This function supports both batched and flattened input tensor. For flattened input tensor, either
    (`packed_info`) or (`ray_indices` and `n_rays`) should be provided.

    Args:
        alphas: The opacity values of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
        packed_info: A tensor of shape (n_rays, 2) that specifies the start and count
            of each chunk in the flattened samples, with in total n_rays chunks.
            Useful for flattened input.
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        early_stop_eps: The early stopping threshold on transmittance.
        alpha_thre: The threshold on opacity.
        prefix_trans: The pre-computed transmittance of the samples. Tensor with shape (all_samples,).

    Returns:
        A boolean tensor indicating which samples are visible. Same shape as `alphas`.

    Examples:

    .. code-block:: python

        >>> alphas = torch.tensor([0.4, 0.8, 0.1, 0.8, 0.1, 0.0, 0.9], device="cuda")
        >>> ray_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2], device="cuda")
        >>> transmittance = render_transmittance_from_alpha(alphas, ray_indices=ray_indices)
        tensor([1.0, 0.6, 0.12, 1.0, 0.2, 1.0, 1.0])
        >>> visibility = render_visibility_from_alpha(
        >>>     alphas, ray_indices=ray_indices, early_stop_eps=0.3, alpha_thre=0.2)
        tensor([True,  True, False,  True, False, False,  True])

    """
    trans = render_transmittance_from_alpha(
        alphas, packed_info, ray_indices, n_rays, prefix_trans
    )
    vis = trans >= early_stop_eps
    if alpha_thre > 0:
        vis = vis & (alphas >= alpha_thre)
    return vis


@torch.no_grad()
def render_visibility_from_density(
    t_starts: Tensor,
    t_ends: Tensor,
    sigmas: Tensor,
    packed_info: Optional[Tensor] = None,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
    prefix_trans: Optional[Tensor] = None,
) -> Tensor:
    """Compute visibility from density :math:`\\sigma_i` and interval :math:`\\delta_i`.

    In this function, we first compute the transmittance and opacity from the sample density. The
    transmittance is then used to filter out occluded samples. And opacity is used to
    filter out transparent samples. The function returns a boolean tensor indicating
    which samples are visible (`transmittance > early_stop_eps` and `opacity > alpha_thre`).

    This function supports both batched and flattened input tensor. For flattened input tensor, either
    (`packed_info`) or (`ray_indices` and `n_rays`) should be provided.

    Args:
        alphas: The opacity values of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
        packed_info: A tensor of shape (n_rays, 2) that specifies the start and count
            of each chunk in the flattened samples, with in total n_rays chunks.
            Useful for flattened input.
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        early_stop_eps: The early stopping threshold on transmittance.
        alpha_thre: The threshold on opacity.
        prefix_trans: The pre-computed transmittance of the samples. Tensor with shape (all_samples,).

    Returns:
        A boolean tensor indicating which samples are visible. Same shape as `alphas`.

    Examples:

    .. code-block:: python

        >>> t_starts = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device="cuda")
        >>> t_ends = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], device="cuda")
        >>> sigmas = torch.tensor([0.4, 0.8, 0.1, 0.8, 0.1, 0.0, 0.9], device="cuda")
        >>> ray_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2], device="cuda")
        >>> transmittance, alphas = render_transmittance_from_density(
        >>>     t_starts, t_ends, sigmas, ray_indices=ray_indices)
        transmittance: [1.00, 0.67, 0.30, 1.00, 0.45, 1.00, 1.00]
        alphas: [0.33, 0.55, 0.095, 0.55, 0.095, 0.00, 0.59]
        >>> visibility = render_visibility_from_density(
        >>>     t_starts, t_ends, sigmas, ray_indices=ray_indices, early_stop_eps=0.3, alpha_thre=0.2)
        tensor([True,  True, False,  True, False, False,  True])

    """
    trans, alphas = render_transmittance_from_density(
        t_starts, t_ends, sigmas, packed_info, ray_indices, n_rays, prefix_trans
    )
    vis = trans >= early_stop_eps
    if alpha_thre > 0:
        vis = vis & (alphas >= alpha_thre)
    return vis


def accumulate_along_rays(
    weights: Tensor,
    values: Optional[Tensor] = None,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
) -> Tensor:
    """Accumulate volumetric values along the ray.

    This function supports both batched inputs and flattened inputs with
    `ray_indices` and `n_rays` provided.

    Note:
        This function is differentiable to `weights` and `values`.

    Args:
        weights: Weights to be accumulated. If `ray_indices` not provided,
            `weights` must be batched with shape (n_rays, n_samples). Else it
            must be flattened with shape (all_samples,).
        values: Values to be accumulated. If `ray_indices` not provided,
            `values` must be batched with shape (n_rays, n_samples, D). Else it
            must be flattened with shape (all_samples, D). None means
            we accumulate weights along rays. Default: None.
        ray_indices: Ray indices of the samples with shape (all_samples,).
            If provided, `weights` must be a flattened tensor with shape (all_samples,)
            and values (if not None) must be a flattened tensor with shape (all_samples, D).
            Default: None.
        n_rays: Number of rays. Should be provided together with `ray_indices`. Default: None.

    Returns:
        Accumulated values with shape (n_rays, D). If `values` is not given we return
        the accumulated weights, in which case D == 1.

    Examples:

    .. code-block:: python

        # Rendering: accumulate rgbs, opacities, and depths along the rays.
        colors = accumulate_along_rays(weights, rgbs, ray_indices, n_rays)
        opacities = accumulate_along_rays(weights, None, ray_indices, n_rays)
        depths = accumulate_along_rays(
            weights,
            (t_starts + t_ends)[:, None] / 2.0,
            ray_indices,
            n_rays,
        )
        # (n_rays, 3), (n_rays, 1), (n_rays, 1)
        print(colors.shape, opacities.shape, depths.shape)

    """
    if values is None:
        src = weights[..., None]
    else:
        assert values.dim() == weights.dim() + 1
        assert weights.shape == values.shape[:-1]
        src = weights[..., None] * values
    if ray_indices is not None:
        assert n_rays is not None, "n_rays must be provided"
        assert weights.dim() == 1, "weights must be flattened"
        outputs = torch.zeros(
            (n_rays, src.shape[-1]), device=src.device, dtype=src.dtype
        )
        outputs.index_add_(0, ray_indices, src)
    else:
        outputs = torch.sum(src, dim=-2)
    return outputs


def accumulate_along_rays_(
    weights: Tensor,
    values: Optional[Tensor] = None,
    ray_indices: Optional[Tensor] = None,
    outputs: Optional[Tensor] = None,
) -> None:
    """Accumulate volumetric values along the ray.

    Inplace version of :func:`accumulate_along_rays`.
    """
    if weights.shape[0] == 0:
        return 0
    if values is None:
        src = weights[..., None]
    else:
        assert values.dim() == weights.dim() + 1
        assert weights.shape == values.shape[:-1]
        src = weights[..., None] * values
    if ray_indices is not None:
        # assert weights.dim() == 1, "weights must be flattened"
        # assert (
        #     outputs.dim() == 2 and outputs.shape[-1] == src.shape[-1]
        # ), "outputs must be of shape (n_rays, D)"
        outputs.index_add_(0, ray_indices, src)
    else:
        outputs.add_(src.sum(dim=-2))


def shift_transient_grid_sample_3d(transient, depth, exposure_time, n_bins):
    x_dim = transient.shape[0]
    bins_move = depth/exposure_time
    if x_dim%2 == 0:
        x = (torch.arange(x_dim, device=transient.device)-x_dim//2+0.5)/(x_dim//2-0.5)
    else:
        x = (torch.arange(x_dim, device=transient.device)-x_dim//2)/(x_dim//2)

    if x_dim == 1:
        x = torch.zeros_like(x)
        
    z = torch.arange(n_bins, device=transient.device).float()
    X, Z = torch.meshgrid(x, z, indexing="ij")
    Z = Z - bins_move
    Z[Z<0] = n_bins+1
    Z = (Z-n_bins//2+0.5)/(n_bins//2-0.5)
    grid = torch.stack((Z, X), dim=-1)[None, ...]
    shifted_transient = torch.nn.functional.grid_sample(transient.permute(2, 0, 1)[None], grid, align_corners=True).squeeze(0).permute(1, 2, 0)
    return shifted_transient

