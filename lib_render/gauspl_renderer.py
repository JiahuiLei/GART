# ! Warning, this wrapper is based on the original gaussian-splatting

import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from sh_utils import eval_sh
import time
import torch
import numpy as np
import torch.nn.functional as F


def render_cam_pcl(
    xyz,
    frame,
    scale,
    opacity,
    color_feat,
    H,
    W,
    CAM_K,
    verbose=False,
    active_sph_order=0,
    bg_color=[1.0, 1.0, 1.0],
):
    # ! Camera is at origin, every input is in camera coordinate space

    S = torch.zeros_like(frame)
    S[:, 0, 0] = scale[:, 0]
    S[:, 1, 1] = scale[:, 1]
    S[:, 2, 2] = scale[:, 2]
    actual_covariance = frame @ (S**2) @ frame.permute(0, 2, 1)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    device = xyz.device
    screenspace_points = (
        torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device=xyz.device) + 0
    )
    # screenspace_points.retain_grad()
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # * Specially handle the non-centered camera, using first padding and finally crop
    if abs(H // 2 - CAM_K[1, 2]) > 1.0 or abs(W // 2 - CAM_K[0, 2]) > 1.0:
        center_handling_flag = True
        left_w, right_w = CAM_K[0, 2], W - CAM_K[0, 2]
        top_h, bottom_h = CAM_K[1, 2], H - CAM_K[1, 2]
        new_W = int(2 * max(left_w, right_w))
        new_H = int(2 * max(top_h, bottom_h))
    else:
        center_handling_flag = False
        new_W, new_H = W, H

    # Set up rasterization configuration
    FoVx = focal2fov(CAM_K[0, 0], new_W)
    FoVy = focal2fov(CAM_K[1, 1], new_H)
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)

    # TODO: Check dynamic gaussian repos and original gaussian repo, they use projection matrix to handle non-centered K, not using this stupid padding like me
    viewmatrix = torch.from_numpy(getWorld2View2(np.eye(3), np.zeros(3)).transpose(0, 1)).to(device)
    projection_matrix = (
        getProjectionMatrix(znear=0.01, zfar=1.0, fovX=FoVx, fovY=FoVy).transpose(0, 1).to(device)
    )
    full_proj_transform = (viewmatrix.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = viewmatrix.inverse()[3, :3]

    raster_settings = GaussianRasterizationSettings(
        image_height=new_H,
        image_width=new_W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.tensor(bg_color, dtype=torch.float32, device=device),
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=full_proj_transform,
        sh_degree=0,  # ! use pre-compute color!
        campos=camera_center,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = xyz
    means2D = screenspace_points
    # opacity = torch.ones_like(means3D[:, 0]) * sigma

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    # JH
    cov3D_precomp = strip_lowerdiag(actual_covariance)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # xyz are in camera frame, so the dir in camera frame is just their normalized direction
    dir_cam = F.normalize(xyz, dim=-1)
    # P_w = Frame @ P_local
    dir_local = torch.einsum("nji,nj->ni", frame, dir_cam)  # note the transpose
    dir_local = F.normalize(
        dir_local, dim=-1
    )  # If frame is not SO(3) but Affinity, have to normalize
    N = len(color_feat)
    shs_view = color_feat.reshape(N, -1, 3)  # N, Deg, Channels
    sh2rgb = eval_sh(active_sph_order, shs_view.permute(0, 2, 1), dir_local)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    # colors_precomp = color_feat

    # Rasterize visible Gaussians to image, obtain their radii (on screen).

    start_time = time.time()
    ret = rasterizer(
        means3D=means3D.float(),
        means2D=means2D.float(),
        shs=None,
        colors_precomp=colors_precomp.float(),
        opacities=opacity.float(),
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp.float(),
    )
    if len(ret) == 2:
        rendered_image, radii = ret
        depth, alpha = None, None
    elif len(ret) == 4:
        rendered_image, radii, depth, alpha = ret
    else:
        raise ValueError(f"Unexpected return value from rasterizer with len={len(ret)}")
    if verbose:
        print(
            f"render time: {(time.time() - start_time)*1000:.3f}ms",
        )
    ret = {
        "rgb": rendered_image,
        "dep": depth,
        "alpha": alpha,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
    if center_handling_flag:
        for k in ["rgb", "dep", "alpha"]:
            if ret[k] is None:
                continue
            if left_w > right_w:
                ret[k] = ret[k][:, :, :W]
            else:
                ret[k] = ret[k][:, :, -W:]
            if top_h > bottom_h:
                ret[k] = ret[k][:, :H, :]
            else:
                ret[k] = ret[k][:, -H:, :]
    return ret


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty
