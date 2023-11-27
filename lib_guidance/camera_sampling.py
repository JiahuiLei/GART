import random
import numpy as np
import torch
import math


def sample_camera(
    global_step=1,
    n_view=4,
    real_batch_size=1,
    random_azimuth_range=[-180.0, 180.0],
    random_elevation_range=[0.0, 30.0],
    eval_elevation_deg=15,
    camera_distance_range=[0.8, 1.0],  # relative
    fovy_range=[15, 60],
    zoom_range=[1.0, 1.0],
    progressive_until=0,
    relative_radius=True,
):
    # camera_perturb = 0.0
    # center_perturb = 0.0
    # up_perturb: 0.0

    # ! from uncond.py
    # ThreeStudio has progressive increase of camera poses, from eval to random
    r = min(1.0, global_step / (progressive_until + 1))
    elevation_range = [
        (1 - r) * eval_elevation_deg + r * random_elevation_range[0],
        (1 - r) * eval_elevation_deg + r * random_elevation_range[1],
    ]
    azimuth_range = [
        (1 - r) * 0.0 + r * random_azimuth_range[0],
        (1 - r) * 0.0 + r * random_azimuth_range[1],
    ]

    # sample elevation angles
    if random.random() < 0.5:
        # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
        elevation_deg = (
            torch.rand(real_batch_size) * (elevation_range[1] - elevation_range[0])
            + elevation_range[0]
        ).repeat_interleave(n_view, dim=0)
        elevation = elevation_deg * math.pi / 180
    else:
        # otherwise sample uniformly on sphere
        elevation_range_percent = [
            (elevation_range[0] + 90.0) / 180.0,
            (elevation_range[1] + 90.0) / 180.0,
        ]
        # inverse transform sampling
        elevation = torch.asin(
            2
            * (
                torch.rand(real_batch_size)
                * (elevation_range_percent[1] - elevation_range_percent[0])
                + elevation_range_percent[0]
            )
            - 1.0
        ).repeat_interleave(n_view, dim=0)
        elevation_deg = elevation / math.pi * 180.0

    # sample azimuth angles from a uniform distribution bounded by azimuth_range
    # ensures sampled azimuth angles in a batch cover the whole range
    azimuth_deg = (
        torch.rand(real_batch_size).reshape(-1, 1) + torch.arange(n_view).reshape(1, -1)
    ).reshape(-1) / n_view * (azimuth_range[1] - azimuth_range[0]) + azimuth_range[0]
    azimuth = azimuth_deg * math.pi / 180

    ######## Different from original ########
    # sample fovs from a uniform distribution bounded by fov_range
    fovy_deg = (
        torch.rand(real_batch_size) * (fovy_range[1] - fovy_range[0]) + fovy_range[0]
    ).repeat_interleave(n_view, dim=0)
    fovy = fovy_deg * math.pi / 180

    # sample distances from a uniform distribution bounded by distance_range
    camera_distances = (
        torch.rand(real_batch_size) * (camera_distance_range[1] - camera_distance_range[0])
        + camera_distance_range[0]
    ).repeat_interleave(n_view, dim=0)
    if relative_radius:
        scale = 1 / torch.tan(0.5 * fovy)
        camera_distances = scale * camera_distances

    # zoom in by decreasing fov after camera distance is fixed
    zoom = (
        torch.rand(real_batch_size) * (zoom_range[1] - zoom_range[0]) + zoom_range[0]
    ).repeat_interleave(n_view, dim=0)
    fovy = fovy * zoom
    fovy_deg = fovy_deg * zoom
    ###########################################

    # convert spherical coordinates to cartesian coordinates
    # right hand coordinate system, x back, y right, z up
    # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
    camera_positions = torch.stack(
        [
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation),
        ],
        dim=-1,
    )

    azimuth, elevation
    # build opencv camera
    z = -torch.stack(
        [
            torch.cos(elevation) * torch.cos(azimuth),
            torch.cos(elevation) * torch.sin(azimuth),
            torch.sin(elevation),
        ],
        -1,
    )  # nview, 3
    # up is 0,0,1
    x = torch.cross(z, torch.tensor([0.0, 0.0, 1.0], device=z.device).repeat(n_view, 1), -1)
    y = torch.cross(z, x, -1)

    R_wc = torch.stack([x, y, z], dim=2)  # nview, 3, 3, col is basis
    t_wc = camera_positions

    T_wc = torch.eye(4, device=R_wc.device).repeat(n_view, 1, 1)
    T_wc[:, :3, :3] = R_wc
    T_wc[:, :3, 3] = t_wc

    return T_wc, fovy_deg  # B,4,4, B


def opencv2blender(T):
    ret = T.clone()
    # y,z are negative
    ret[:, :, 1] *= -1
    ret[:, :, 2] *= -1
    return ret


def naive_sample_T_oc(
    theta_range=[-np.pi, np.pi], phi_range=[-np.pi / 6, np.pi / 3], r_range=[0.9, 1.1]
):
    # ! May have bug!
    # in opencv convention
    theta = np.random.uniform(*theta_range)
    phi = np.random.uniform(*phi_range)
    r = np.random.uniform(*r_range)

    T_oc = cam_from_angle(theta, phi, r)
    return T_oc


def cam_from_angle(theta, phi, r):
    sph_point = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])
    _z = -sph_point
    _x = np.cross(_z, np.array([0, 0, 1]))
    _y = np.cross(_z, _x)
    R_oc = np.stack([_x / np.linalg.norm(_x), _y / np.linalg.norm(_y), _z], axis=1)

    # t_oc = np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]) * r
    t_oc = sph_point * r
    T_oc = np.eye(4)
    T_oc[:3, :3] = R_oc
    T_oc[:3, 3] = t_oc
    return T_oc


def fov2K(fov=90, H=256, W=256):
    if isinstance(fov, torch.Tensor):
        f = H / (2 * torch.tan(fov / 2 * np.pi / 180))
        K = torch.eye(3).repeat(fov.shape[0], 1, 1).to(fov)
        K[:, 0, 0], K[:, 0, 2] = f, W / 2.0
        K[:, 1, 1], K[:, 1, 2] = f, H / 2.0
        return K.clone()
    else:
        f = H / (2 * np.tan(fov / 2 * np.pi / 180))
        K = np.eye(3)
        K[0, 0], K[0, 2] = f, W / 2.0
        K[1, 1], K[1, 2] = f, H / 2.0
        return K.copy()


if __name__ == "__main__":
    T_cv, fovy = sample_camera(global_step=1.0, progressive_until=10, n_view=1)
    print(fovy)
    T_bl = opencv2blender(T_cv)
