import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import trimesh
from pytorch3d.transforms import (
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    quaternion_to_matrix,
    matrix_to_quaternion,
)
import logging, time
import os

def init_qso_naive(N, opacity_base_logit, scale_base_logit):
    o = nn.Parameter(torch.ones(N, 1) * opacity_base_logit)
    s = torch.ones(N, 3) * scale_base_logit
    q = torch.zeros(N, 4)
    q[:, 0] = 1.0
    return q, s, o


def init_xyz_near_mesh(v_init, faces, n_init, init_std):
    tmesh = trimesh.Trimesh(v_init, faces, process=False)
    mesh_pts, _ = trimesh.sample.sample_surface_even(tmesh, n_init)
    noise = np.random.randn(*mesh_pts.shape) * init_std
    xyz_init = torch.from_numpy(mesh_pts + noise)
    return xyz_init.float()


def init_xyz_on_mesh(v_init, faces, subdivide_num):
    # * xyz
    denser_v, denser_f = v_init.detach().cpu().numpy(), faces
    for i in range(subdivide_num):
        denser_v, denser_f = trimesh.remesh.subdivide(denser_v, denser_f)
    body_mesh = trimesh.Trimesh(denser_v, denser_f, process=False)
    v_init = torch.as_tensor(denser_v, dtype=torch.float32)
    return v_init, body_mesh


def init_qso_on_mesh(
    body_mesh,
    scale_init_factor,
    thickness_init_factor,
    max_scale,
    min_scale,
    s_inv_act,
    opacity_base_logit,
):
    # * Quaternion
    # each column is a basis vector
    # the local frame is z to normal, xy on the disk
    normal = body_mesh.vertex_normals.copy()
    v_init = torch.as_tensor(body_mesh.vertices.copy())
    faces = torch.as_tensor(body_mesh.faces.copy())

    uz = torch.as_tensor(normal, dtype=torch.float32)
    rand_dir = torch.randn_like(uz)
    ux = F.normalize(torch.cross(uz, rand_dir, dim=-1), dim=-1)
    uy = F.normalize(torch.cross(uz, ux, dim=-1), dim=-1)
    frame = torch.stack([ux, uy, uz], dim=-1)  # N,3,3
    ret_q = matrix_to_quaternion(frame)

    # * Scaling
    xy = v_init[faces[:, 1]] - v_init[faces[:, 0]]
    xz = v_init[faces[:, 2]] - v_init[faces[:, 0]]
    area = torch.norm(torch.cross(xy, xz, dim=-1), dim=-1) / 2
    vtx_nn_area = torch.zeros_like(v_init[:, 0])
    for i in range(3):
        vtx_nn_area.scatter_add_(0, faces[:, i], area / 3.0)
    radius = torch.sqrt(vtx_nn_area / np.pi)
    # radius = torch.clamp(radius * scale_init_factor, max=max_scale, min=min_scale)
    # ! 2023.11.22, small eps
    radius = torch.clamp(
        radius * scale_init_factor, max=max_scale - 1e-4, min=min_scale + 1e-4
    )
    thickness = radius * thickness_init_factor
    # ! 2023.11.22, small eps
    thickness = torch.clamp(thickness, max=max_scale - 1e-4, min=min_scale + 1e-4)
    radius_logit = s_inv_act(radius)
    thickness_logit = s_inv_act(thickness)
    ret_s = torch.stack([radius_logit, radius_logit, thickness_logit], dim=-1)

    ret_o = torch.ones_like(v_init[:, :1]) * opacity_base_logit
    return ret_q, ret_s, ret_o


def get_on_mesh_init_geo_values(
    template,
    on_mesh_subdivide,
    scale_init_factor,
    thickness_init_factor,
    max_scale,
    min_scale,
    s_inv_act,
    opacity_init_logit,
):
    v, f = template.get_init_vf()
    x, mesh = init_xyz_on_mesh(v, f, on_mesh_subdivide)
    q, s, o = init_qso_on_mesh(
        mesh,
        scale_init_factor,
        thickness_init_factor,
        max_scale,
        min_scale,
        s_inv_act,
        opacity_init_logit,
    )
    return x, q, s, o


def get_near_mesh_init_geo_values(
    template, scale_base_logit, opacity_base_logit, random_init_num, random_init_std
):
    v, f = template.get_init_vf()
    x = init_xyz_near_mesh(v, f, random_init_num, random_init_std)
    q, s, o = init_qso_naive(len(x), opacity_base_logit, scale_base_logit)
    return x, q, s, o


def get_inside_mesh_init_geo_values(
    template, scale_base_logit, opacity_base_logit, random_init_num, buffer_factor=10
):
    v, f = template.get_init_vf()

    tmesh = trimesh.Trimesh(v, f, process=False)

    bounds = tmesh.bounds  # [2,3]
    # generate random points inside the bounds
    os.makedirs("cache", exist_ok=True)
    mode = template.name
    cache_fn = f"cache/{mode}_{template.cano_pose_type}_{random_init_num}.npz"
    
    if os.path.exists(cache_fn):
        x = np.load(cache_fn)["x"]
    else:
        x = (
            torch.rand(random_init_num * buffer_factor, 3) * (bounds[1] - bounds[0])
            + bounds[0]
        )
        logging.info(f"Init with trimesh contained examination {len(x)} pts ...")
        start_time = time.time()
        contained = tmesh.contains(x)
        logging.info(
            "Init inside with trimesh contained function done in {}s".format(
                time.time() - start_time
            )
        )
        x = x[contained][:random_init_num]
        np.savez_compressed(cache_fn, x=x.numpy().astype(np.float16))
    x = torch.as_tensor(x, dtype=torch.float32)
    logging.info(f"Init {len(x)} Components inside the mesh")
    q, s, o = init_qso_naive(len(x), opacity_base_logit, scale_base_logit)
    return x, q, s, o
